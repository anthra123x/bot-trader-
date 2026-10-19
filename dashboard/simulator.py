"""Autonomous Simulation Engine for TradingAgents Dashboard.

Simulates the complete multi-agent workflow (analysts, debates, risk engine,
order execution, dynamic price fluctuations, take-profit/stop-loss exits,
and portfolio mark-to-market).
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from bot.brokers.paper import PaperBroker
from bot.learning.pattern_memory import PatternMemory
from bot.models import AssetClass, Order, OrderSide, OrderStatus, OrderType, TradeOutcome

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Simulates realistic autonomous trading floor deliberations and paper executions."""

    def __init__(self, log_buffer: Any, order_history: list[dict[str, Any]]):
        self.log_buffer = log_buffer
        self.order_history = order_history
        self.is_running = False
        self._task: asyncio.Task | None = None
        self.paper_broker = PaperBroker(db_path="data/simulation_paper.db", initial_cash=100_000.0)
        self.pattern_memory = PatternMemory(db_path="data/pattern_memory.db")
        self.watchlist = ["NVDA", "BTC-USD", "AAPL", "ETH-USD", "TSLA", "MSFT"]
        
        # Current realistic simulated live prices
        self.current_prices: dict[str, float] = {
            "NVDA": 214.50,
            "BTC-USD": 81650.0,
            "AAPL": 344.00,
            "ETH-USD": 2410.0,
            "TSLA": 355.00,
            "MSFT": 529.00,
        }
        # Keep track of targets for open positions: {symbol: {"tp": float, "sl": float}}
        self._position_targets: dict[str, dict[str, float]] = {}

        # Performance stats
        self.closed_trades_count = 0
        self.winning_trades_count = 0
        self.realized_pnl_total = 0.0

    def log(self, message: str) -> None:
        """Push a message to the shared circular log buffer."""
        self.log_buffer.append(message)
        logger.info(message)

    def start(self) -> None:
        """Start the simulation loop in the background."""
        if self.is_running:
            return
        self.is_running = True
        self.log("[SISTEMA] 🚀 MODO SIMULACIÓN EN VIVO ACTIVADO (Supervisión Cuantitativa)")
        self._task = asyncio.create_task(self._simulation_loop())

    def stop(self) -> None:
        """Stop the simulation loop."""
        self.is_running = False
        if self._task:
            self._task.cancel()
        self.log("[SISTEMA] ⏸️ Modo Simulación pausado.")

    def reset(self) -> None:
        """Reset the simulation environment and account balance."""
        self.paper_broker.reset(100_000.0)
        self.closed_trades_count = 0
        self.winning_trades_count = 0
        self.realized_pnl_total = 0.0
        self._position_targets.clear()
        self.order_history.clear()
        self.current_prices = {
            "NVDA": 214.50,
            "BTC-USD": 81650.0,
            "AAPL": 344.00,
            "ETH-USD": 2410.0,
            "TSLA": 355.00,
            "MSFT": 529.00,
        }
        self.log("[SISTEMA] 🔄 Cartera de simulación reiniciada a $100,000.00 USD (Efectivo 100%).")

    def tick_market(self) -> None:
        """Simulate realistic live ticker price movement for all watchlist assets."""
        for sym in self.watchlist:
            price = self.current_prices.get(sym, 100.0)
            is_crypto = "-USD" in sym or "/USD" in sym
            vol = random.uniform(0.004, 0.012) if is_crypto else random.uniform(0.002, 0.007)
            direction = 1 if random.random() < 0.54 else -1
            delta = price * vol * direction
            self.current_prices[sym] = round(max(1.0, price + delta), 2)

    def get_price(self, symbol: str) -> float:
        """Return the current consistent simulated market price."""
        if symbol not in self.current_prices:
            self.current_prices[symbol] = 100.0
        return self.current_prices[symbol]

    async def _simulation_loop(self) -> None:
        """Continuous cyclic analysis, dynamic market ticking, and trading simulation."""
        ticker_index = 0
        while self.is_running:
            # Step A: Update live market ticker prices
            self.tick_market()

            # Step B: Evaluate open positions for Take-Profit / Stop-Loss exits
            await self._evaluate_open_positions_exits()

            # Step C: Select next symbol for multi-agent analysis
            symbol = self.watchlist[ticker_index % len(self.watchlist)]
            ticker_index += 1

            try:
                await self._simulate_asset_cycle(symbol)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log(f"[ERROR] Excepción en simulación para {symbol}: {e}")

            # Sleep between analysis cycles
            await asyncio.sleep(4)

    async def _evaluate_open_positions_exits(self) -> None:
        """Check all currently open positions against TP/SL levels."""
        positions = self.paper_broker.get_positions()
        for pos in positions:
            current_price = self.get_price(pos.symbol)
            pnl_dollars = (current_price - pos.entry_price) * pos.quantity
            pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100.0

            targets = self._position_targets.get(pos.symbol, {})
            tp_price = targets.get("tp", pos.entry_price * 1.045)  # default +4.5%
            sl_price = targets.get("sl", pos.entry_price * 0.965)  # default -3.5%

            # Take Profit triggered
            if current_price >= tp_price or pnl_pct >= 4.0:
                self.log(
                    f"[Gestión de Salida] 🎯 TAKE-PROFIT ALCANZADO en {pos.symbol}: "
                    f"+${pnl_dollars:,.2f} (+{pnl_pct:.2f}%). Ejecutando venta..."
                )
                await self._execute_sell(pos.symbol, pos.quantity, current_price, "TAKE_PROFIT", pnl_dollars)
                self._position_targets.pop(pos.symbol, None)
                await asyncio.sleep(0.5)

            # Stop Loss triggered
            elif current_price <= sl_price or pnl_pct <= -3.2:
                self.log(
                    f"[Gestión de Salida] 🛡️ STOP-LOSS ACTIVADO en {pos.symbol}: "
                    f"-${abs(pnl_dollars):,.2f} ({pnl_pct:.2f}%). Cortando posición..."
                )
                await self._execute_sell(pos.symbol, pos.quantity, current_price, "STOP_LOSS", pnl_dollars)
                self._position_targets.pop(pos.symbol, None)
                await asyncio.sleep(0.5)

    async def _execute_sell(self, symbol: str, quantity: float, price: float, reason: str, pnl: float) -> None:
        """Execute a sell order to close position."""
        is_crypto = "-USD" in symbol or "/USD" in symbol
        order = Order(
            order_id=f"exit-{str(uuid.uuid4())[:6]}",
            symbol=symbol,
            asset_class=AssetClass.CRYPTO if is_crypto else AssetClass.EQUITY,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            limit_price=price,
            status=OrderStatus.PENDING,
        )
        executed = self.paper_broker.submit_order(order)
        self.log(f"[Broker] ⚡ Orden FILLED: SELL {quantity} {symbol} a ${executed.filled_price:,.2f} ({reason})")

        self.closed_trades_count += 1
        if pnl >= 0:
            self.winning_trades_count += 1
        self.realized_pnl_total += pnl

        # ML4T Feedback Loop: Record outcome into pattern memory
        entry_calc = price - (pnl / quantity) if quantity > 0 else price
        pnl_calc_pct = (pnl / (entry_calc * quantity) * 100.0) if (entry_calc * quantity) > 0 else 0.0
        self.pattern_memory.record_trade_outcome(
            TradeOutcome(
                trade_id=executed.order_id,
                symbol=symbol,
                asset_class=AssetClass.CRYPTO if is_crypto else AssetClass.EQUITY,
                side=OrderSide.SELL,
                entry_price=entry_calc,
                exit_price=price,
                quantity=quantity,
                pnl_dollars=pnl,
                pnl_pct=pnl_calc_pct,
                outcome="WIN" if pnl >= 0 else "LOSS",
                rsi_entry=round(random.uniform(42.0, 74.0), 1),
                ema_ratio=round(random.uniform(0.95, 1.05), 3),
                volatility_atr_pct=round(random.uniform(1.8, 3.8), 2),
            )
        )

        win_rate = (self.winning_trades_count / self.closed_trades_count) * 100.0
        self.log(f"[Rendimiento] 📊 Trades cerrados: {self.closed_trades_count} | Win Rate: {win_rate:.1f}% | PnL Realizado: ${self.realized_pnl_total:,.2f}")

        self.order_history.insert(0, {
            "order_id": executed.order_id,
            "symbol": executed.symbol,
            "side": executed.side.value,
            "quantity": executed.quantity,
            "status": executed.status.value,
            "filled_price": executed.filled_price,
            "limit_price": executed.limit_price,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    async def _simulate_asset_cycle(self, symbol: str) -> None:
        """Simulate multi-agent deliberation and defensive execution for a symbol."""
        current_price = self.get_price(symbol)
        is_crypto = "-USD" in symbol or "/USD" in symbol
        asset_class = AssetClass.CRYPTO if is_crypto else AssetClass.EQUITY

        balance = self.paper_broker.get_account_balance()
        positions = self.paper_broker.get_positions()
        existing_pos = next((p for p in positions if p.symbol == symbol), None)
        existing_market_val = existing_pos.market_value if existing_pos else 0.0

        self.log(f"[ORQUESTADOR] 🔎 Ciclo de análisis para {symbol} (${current_price:,.2f})")
        await asyncio.sleep(1.0)

        # 1. Market Analyst
        rsi_val = round(random.uniform(38.0, 72.0), 1)
        trend = "alcista" if rsi_val > 50 else "bajista / consolidación"
        self.log(f"[Market Analyst] {symbol}: RSI en {rsi_val}, estructura {trend}. Soportes y medias móviles validados.")
        await asyncio.sleep(1.0)

        # 2. Sentiment Analyst
        sentiment_pct = random.randint(52, 88)
        self.log(f"[Sentiment Analyst] {symbol}: Sentimiento social {sentiment_pct}% Bullish en StockTwits/Reddit.")
        await asyncio.sleep(1.0)

        # 3. News Analyst
        catalysts = [
            "Flujo institucional neto comprador y proyecciones de demanda elevadas.",
            "Condiciones macro favorables y baja presión vendedora.",
            "Rebote técnico sobre zona de consolidación institucional.",
        ]
        self.log(f"[News Analyst] {symbol}: {random.choice(catalysts)}")
        await asyncio.sleep(1.0)

        # 4. Fundamentals Analyst
        self.log(f"[Fundamentals Analyst] {symbol}: Crecimiento de ingresos y salud de balance alineados.")
        await asyncio.sleep(1.0)

        # 5. Bull vs Bear Debate
        self.log(f"[Bull Researcher] Tesis: Fuerte absorción de oferta y ratio riesgo/beneficio favorable.")
        await asyncio.sleep(0.8)
        self.log(f"[Bear Researcher] Objeción: Controlar volatilidad con Stop-Loss innegociable.")
        await asyncio.sleep(0.8)

        # 6. Research Manager Synthesis
        action = random.choice([OrderSide.BUY, OrderSide.BUY, OrderSide.HOLD])
        if action == OrderSide.HOLD:
            self.log(f"[Research Manager] Síntesis para {symbol}: Señal neutral / esperando confirmación. Recomendación: HOLD.")
            return

        self.log(f"[Research Manager] Síntesis para {symbol}: Convicción compradora confirmada. Recomendación: BUY.")
        await asyncio.sleep(0.8)

        # 7. Trader Agent Proposal
        sl_pct = random.uniform(3.2, 4.2)
        tp_pct = random.uniform(5.5, 8.5)
        stop_loss = round(current_price * (1 - sl_pct / 100.0), 2)
        take_profit = round(current_price * (1 + tp_pct / 100.0), 2)

        self.log(f"[Trader Agent] Propuesta: BUY. SL: ${stop_loss:,.2f} (-{sl_pct:.1f}%), TP: ${take_profit:,.2f} (+{tp_pct:.1f}%).")
        await asyncio.sleep(1.0)

        # 8. Defensive Risk Engine Checks
        # Rule 0: Pattern Memory & Mistake Feedback Check (ML4T Feedback Loop)
        pattern_check = self.pattern_memory.evaluate_setup(
            symbol=symbol,
            asset_class=asset_class,
            rsi=rsi_val,
            ema_ratio=1.03 if trend == "alcista" else 0.95,
            atr_pct=random.uniform(1.8, 3.2),
        )
        if not pattern_check.is_safe:
            self.log(f"[Pattern Memory] 🛡️ VETO POR PATRÓN FALLIDO: {pattern_check.warning_message}")
            self.log(f"[Pattern Memory] 💡 Lección de retroalimentación: '{pattern_check.learned_lesson}'")
            return
        elif pattern_check.risk_multiplier < 1.0:
            self.log(f"[Pattern Memory] ⚠️ {pattern_check.warning_message}")

        # Rule A: Minimum Cash Reserve check (maintain at least 15% cash)
        if balance.cash < (balance.equity * 0.15):
            self.log(f"[Risk Engine] ⚠️ RECHAZADO: Reserva de efectivo baja (${balance.cash:,.2f} < 15% de cartera). Priorizando liquidez.")
            return

        # Rule B: Concentration Cap (maximum 10% of portfolio in any single symbol)
        max_alloc_dollars = balance.equity * 0.10
        if existing_market_val >= max_alloc_dollars:
            self.log(f"[Risk Engine] ⚠️ RECHAZADO: Exposición en {symbol} ya alcanza ${existing_market_val:,.2f} (Tope máx 10%).")
            return

        remaining_alloc = max_alloc_dollars - existing_market_val
        alloc_dollars = min(remaining_alloc, random.uniform(4000.0, 7500.0), balance.cash * 0.5)

        if asset_class == AssetClass.CRYPTO:
            qty = round(alloc_dollars / current_price, 4)
        else:
            qty = max(1.0, float(int(alloc_dollars / current_price)))

        if qty <= 0:
            self.log(f"[Risk Engine] ⚠️ RECHAZADO: Tamaño calculado insignificante.")
            return

        self.log(f"[Risk Engine] ✅ APROBADO: Asignación ${qty * current_price:,.2f} ({qty} unid). Stop Loss protegido a ${stop_loss:,.2f}.")
        await asyncio.sleep(1.0)

        # 9. Broker Execution
        order = Order(
            order_id=f"sim-{str(uuid.uuid4())[:6]}",
            symbol=symbol,
            asset_class=asset_class,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=qty,
            limit_price=current_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            status=OrderStatus.PENDING,
        )

        executed = self.paper_broker.submit_order(order)
        if executed.status == OrderStatus.FILLED:
            self.log(f"[Broker] ⚡ Orden FILLED: BUY {qty} {symbol} a ${executed.filled_price:,.2f}")
            # Register TP and SL levels for exit monitoring
            self._position_targets[symbol] = {"tp": take_profit, "sl": stop_loss}

            self.order_history.insert(0, {
                "order_id": executed.order_id,
                "symbol": executed.symbol,
                "side": executed.side.value,
                "quantity": executed.quantity,
                "status": executed.status.value,
                "filled_price": executed.filled_price,
                "limit_price": executed.limit_price,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            self.log(f"[Broker] ❌ Orden rechazada: {executed.rejection_reason}")
