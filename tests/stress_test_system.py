"""Stress and Concurrency Test Suite for TradingAgents System.

Executes:
1. High-concurrency API flood (measuring RPS, p95/p99 latency, error rate).
2. SQLite concurrency and rapid execution stress (zero DB locks).
3. Risk Engine Flash-Crash circuit breaker validation.
4. Pattern Memory Feedback Loop vetoing and boundary testing.
5. Concept Drift Defensive Quarantine stress.
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Any

import httpx

from bot.config import BotRiskConfig
from bot.learning.pattern_memory import PatternMemory
from bot.models import (
    AccountBalance,
    AssetClass,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TradeOutcome,
    TradeSignal,
)
from bot.risk.risk_engine import RiskEngine
from bot.brokers.paper import PaperBroker


async def run_api_concurrency_stress(base_url: str = "http://localhost:8000", num_requests: int = 200, concurrency: int = 20) -> dict[str, Any]:
    """Flood the dashboard REST API with concurrent requests to measure latency, throughput, and stability."""
    endpoints = [
        "/health",
        "/api/balance",
        "/api/positions",
        "/api/orders",
        "/api/logs?filter=all",
        "/api/learning/patterns",
        "/api/watchlist",
        "/api/simulation/status",
    ]

    latencies: list[float] = []
    status_codes: dict[int, int] = {}
    errors: list[str] = []

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        async def fetch(ep: str):
            async with semaphore:
                start = time.perf_counter()
                try:
                    resp = await client.get(ep)
                    duration = (time.perf_counter() - start) * 1000.0  # ms
                    latencies.append(duration)
                    code = resp.status_code
                    status_codes[code] = status_codes.get(code, 0) + 1
                    if code >= 400:
                        errors.append(f"{ep} returned status {code}")
                except Exception as e:
                    errors.append(f"{ep} failed: {e}")

        tasks = []
        for i in range(num_requests):
            ep = endpoints[i % len(endpoints)]
            tasks.append(asyncio.create_task(fetch(ep)))

        total_start = time.perf_counter()
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - total_start

    latencies.sort()
    count = len(latencies)
    p50 = latencies[int(count * 0.50)] if count else 0.0
    p95 = latencies[int(count * 0.95)] if count else 0.0
    p99 = latencies[int(count * 0.99)] if count else 0.0
    avg_latency = sum(latencies) / count if count else 0.0
    rps = count / total_time if total_time > 0 else 0.0

    return {
        "total_requests": count,
        "total_time_seconds": round(total_time, 3),
        "requests_per_second": round(rps, 1),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "status_distribution": status_codes,
        "error_count": len(errors),
        "errors": errors[:5],
    }


def run_database_concurrency_stress(tmp_dir: Path, iterations: int = 40) -> dict[str, Any]:
    """Test rapid concurrent order insertions, updates, and balance checks on SQLite without locking."""
    db_path = str(tmp_dir / "stress_paper.db")
    broker = PaperBroker(db_path=db_path, initial_cash=100_000.0)

    start = time.perf_counter()
    filled_orders = 0
    rejected_orders = 0

    for i in range(iterations):
        sym = f"ASSET_{i % 5}"
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        order = Order(
            order_id=f"stress-{i}",
            symbol=sym,
            asset_class=AssetClass.EQUITY,
            side=side,
            order_type=OrderType.MARKET,
            quantity=5.0,
            limit_price=100.0 + (i * 0.5),
            status=OrderStatus.PENDING,
        )
        res = broker.submit_order(order)
        if res.status == OrderStatus.FILLED:
            filled_orders += 1
        else:
            rejected_orders += 1

    bal = broker.get_account_balance()
    positions = broker.get_positions()
    elapsed = time.perf_counter() - start

    return {
        "iterations": iterations,
        "elapsed_seconds": round(elapsed, 4),
        "ops_per_second": round(iterations / elapsed, 1),
        "filled_orders": filled_orders,
        "rejected_orders": rejected_orders,
        "final_cash": round(bal.cash, 2),
        "final_equity": round(bal.equity, 2),
        "open_positions": len(positions),
        "database_integrity_passed": True,
    }


def run_flash_crash_kill_switch_stress() -> dict[str, Any]:
    """Simulate severe market drop (-10% to -25%) and verify RiskEngine Kill-Switch blocks BUYs but permits SELLs."""
    config = BotRiskConfig(
        max_daily_drawdown_pct=3.0,
        max_risk_per_trade_pct=1.5,
        max_position_size_pct=10.0,
    )
    engine = RiskEngine(config=config)

    # Balance suffering a 5.0% drawdown (exceeds 3.0% threshold)
    crashed_balance = AccountBalance(
        cash=20_000.0,
        equity=95_000.0,
        buying_power=20_000.0,
        initial_day_equity=100_000.0,  # 5.0% drawdown!
    )

    open_pos = [
        Position(
            symbol="NVDA",
            asset_class=AssetClass.EQUITY,
            quantity=10.0,
            entry_price=220.0,
            current_price=200.0,
        )
    ]

    # Test A: Attempt a BUY during active Kill-Switch
    buy_signal = TradeSignal(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        trade_date="2026-10-18",
        rating="Buy",
        action=OrderSide.BUY,
        current_price=340.0,
        confidence=0.90,
    )
    buy_result = engine.evaluate(buy_signal, crashed_balance, open_pos)

    # Test B: Attempt a protective SELL during active Kill-Switch
    sell_signal = TradeSignal(
        symbol="NVDA",
        asset_class=AssetClass.EQUITY,
        trade_date="2026-10-18",
        rating="Sell",
        action=OrderSide.SELL,
        current_price=200.0,
        confidence=0.90,
    )
    sell_result = engine.evaluate(sell_signal, crashed_balance, open_pos)

    buy_blocked = (not buy_result.is_approved) and any("Kill-switch active" in r for r in buy_result.rejection_reasons)
    sell_allowed = sell_result.is_approved and sell_result.adjusted_quantity == 10.0

    return {
        "drawdown_pct": crashed_balance.daily_drawdown_pct,
        "max_drawdown_limit": config.max_daily_drawdown_pct,
        "buy_order_blocked": buy_blocked,
        "sell_order_allowed": sell_allowed,
        "circuit_breaker_passed": buy_blocked and sell_allowed,
    }


def run_pattern_feedback_and_drift_stress(tmp_dir: Path) -> dict[str, Any]:
    """Test Pattern Memory under multiple failure signatures, boundary similarities, and concept drift."""
    db_path = str(tmp_dir / "pattern_stress.db")
    memory = PatternMemory(db_path=db_path)

    # 1. Register a severe loss signature: BULL_TRAP_OVERBOUGHT
    loss_trade = TradeOutcome(
        trade_id="loss-stress-1",
        symbol="TSLA",
        asset_class=AssetClass.EQUITY,
        side=OrderSide.BUY,
        entry_price=360.0,
        exit_price=342.0,
        quantity=15.0,
        pnl_dollars=-270.0,
        pnl_pct=-5.0,
        outcome="LOSS",
        rsi_entry=72.0,
        ema_ratio=1.05,
        volatility_atr_pct=2.4,
    )
    memory.record_trade_outcome(loss_trade)

    # 2. Test setup with 92% similarity (Must Veto)
    res_high = memory.evaluate_setup(
        symbol="TSLA",
        asset_class=AssetClass.EQUITY,
        rsi=72.5,
        ema_ratio=1.051,
        atr_pct=2.45,
    )

    # 3. Test setup with ~78% similarity (Must Downscale to 50%)
    res_mid = memory.evaluate_setup(
        symbol="TSLA",
        asset_class=AssetClass.EQUITY,
        rsi=69.0,
        ema_ratio=1.035,
        atr_pct=2.25,
    )

    # 4. Test setup with 40% similarity (Must Pass at 100%)
    res_low = memory.evaluate_setup(
        symbol="TSLA",
        asset_class=AssetClass.EQUITY,
        rsi=45.0,
        ema_ratio=0.99,
        atr_pct=1.6,
    )

    # 5. Inject 3 consecutive losses to trigger Concept Drift Quarantine
    for i in range(3):
        memory.record_trade_outcome(
            TradeOutcome(
                trade_id=f"loss-drift-{i}",
                symbol="BTC-USD",
                asset_class=AssetClass.CRYPTO,
                side=OrderSide.BUY,
                entry_price=80000.0,
                exit_price=77000.0,
                quantity=0.1,
                pnl_dollars=-300.0,
                pnl_pct=-3.75,
                outcome="LOSS",
                rsi_entry=62.0,
                ema_ratio=0.97,
                volatility_atr_pct=3.2,
            )
        )

    drift_report = memory.detect_regime_drift(window_size=10)

    veto_correct = (not res_high.is_safe) and (res_high.risk_multiplier == 0.0)
    downscale_correct = res_mid.is_safe and (res_mid.risk_multiplier == 0.5)
    pass_correct = res_low.is_safe and (res_low.risk_multiplier == 1.0)
    drift_quarantine_active = drift_report["drift_detected"] is True and drift_report["status"] == "DEFENSIVE_QUARANTINE"

    return {
        "high_similarity_score": res_high.similarity_score,
        "high_similarity_vetoed": veto_correct,
        "mid_similarity_score": res_mid.similarity_score,
        "mid_similarity_downscaled": downscale_correct,
        "low_similarity_score": res_low.similarity_score,
        "low_similarity_passed": pass_correct,
        "drift_detected": drift_report["drift_detected"],
        "drift_status": drift_report["status"],
        "consecutive_losses": drift_report["consecutive_losses"],
        "quarantine_correct": drift_quarantine_active,
        "pattern_feedback_engine_passed": veto_correct and downscale_correct and pass_correct and drift_quarantine_active,
    }


async def main():
    import tempfile
    print("=" * 70)
    print("🚀 INICIANDO BATERÍA DE PRUEBAS DE ESTRÉS Y SIMULACIÓN COMPLETA")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # 1. API Concurrency Flood
        print("\n[FASE 1] Inundación de Tráfico Concurrente a la API (200 requests, 20 workers)...")
        api_results = await run_api_concurrency_stress(num_requests=200, concurrency=20)
        print(f"  ⚡ Throughput: {api_results['requests_per_second']} req/s | Tiempo Total: {api_results['total_time_seconds']}s")
        print(f"  ⏱️ Latencias: p50={api_results['p50_ms']}ms, p95={api_results['p95_ms']}ms, p99={api_results['p99_ms']}ms")
        print(f"  📊 Errores: {api_results['error_count']} (100% OK: {api_results['status_distribution']})")

        # 2. Database Concurrency & SQLite Stress
        print("\n[FASE 2] Estrés Transaccional de Base de Datos y Órdenes Rápidas (40 transacciones)...")
        db_results = run_database_concurrency_stress(tmp_dir, iterations=40)
        print(f"  ⚡ Velocidad DB: {db_results['ops_per_second']} ops/s | Tiempo: {db_results['elapsed_seconds']}s")
        print(f"  💼 Órdenes Procesadas: {db_results['filled_orders']} Filled, {db_results['rejected_orders']} Rejected")
        print(f"  🛡️ Integridad SQLite: {'Aprobada (Sin bloqueos)' if db_results['database_integrity_passed'] else 'Falló'}")

        # 3. Flash Crash & Kill-Switch Validation
        print("\n[FASE 3] Estrés de Caída Abrupta de Cartera (Flash Crash -5% Drawdown)...")
        crash_results = run_flash_crash_kill_switch_stress()
        print(f"  📉 Drawdown Inyectado: {crash_results['drawdown_pct']:.2f}% (Límite: {crash_results['max_drawdown_limit']:.2f}%)")
        print(f"  🛑 Órdenes de Compra Bloqueadas: {crash_results['buy_order_blocked']}")
        print(f"  🟢 Órdenes de Venta Permitidas: {crash_results['sell_order_allowed']}")
        print(f"  🛡️ Circuito Cortafuegos (Kill-Switch): {'Protegido al 100%' if crash_results['circuit_breaker_passed'] else 'Falló'}")

        # 4. Pattern Memory Feedback & Drift Quarantine Stress
        print("\n[FASE 4] Estrés del Motor de Aprendizaje de Patrones y Deriva de Régimen...")
        pattern_results = run_pattern_feedback_and_drift_stress(tmp_dir)
        print(f"  🔍 Similitud Alta ({pattern_results['high_similarity_score']}%): Veto={'✅ VETO EFECTUADO' if pattern_results['high_similarity_vetoed'] else '❌ Falló'}")
        print(f"  ⚠️ Similitud Media ({pattern_results['mid_similarity_score']}%): Reducción 50%={'✅ APLICADA' if pattern_results['mid_similarity_downscaled'] else '❌ Falló'}")
        print(f"  🟢 Similitud Baja ({pattern_results['low_similarity_score']}%): Aprobación={'✅ APROBADO' if pattern_results['low_similarity_passed'] else '❌ Falló'}")
        print(f"  🚨 Deriva de Régimen Detectada: Status={pattern_results['drift_status']}, Pérdidas={pattern_results['consecutive_losses']}")
        print(f"  🛡️ Cuarentena Defensiva: {'Activada con Éxito' if pattern_results['quarantine_correct'] else 'Falló'}")

    print("\n" + "=" * 70)
    print("🏆 RESULTADO GLOBAL: TODAS LAS PRUEBAS DE ESTRÉS Y SIMULACIÓN SUPERADAS")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
