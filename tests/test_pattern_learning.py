"""Unit tests for PatternMemory, Mistake Diagnostics, and Feedback Loops."""

from pathlib import Path
import pytest

from bot.config import BotRiskConfig
from bot.learning.pattern_memory import PatternMemory
from bot.models import (
    AccountBalance,
    AssetClass,
    OrderSide,
    TradeOutcome,
    TradeSignal,
)
from bot.risk.risk_engine import RiskEngine


@pytest.fixture
def temp_memory(tmp_path: Path) -> PatternMemory:
    db_file = tmp_path / "test_patterns.db"
    return PatternMemory(str(db_file))


def test_diagnose_mistake_overbought():
    sig, lesson = PatternMemory.diagnose_mistake(
        side=OrderSide.BUY,
        rsi=72.0,
        ema_ratio=1.05,
        atr_pct=2.0,
        pnl_pct=-4.2,
    )
    assert sig == "BULL_TRAP_OVERBOUGHT"
    assert "sobrecompra" in lesson.lower()


def test_diagnose_mistake_falling_knife():
    sig, lesson = PatternMemory.diagnose_mistake(
        side=OrderSide.BUY,
        rsi=45.0,
        ema_ratio=0.92,
        atr_pct=2.5,
        pnl_pct=-5.1,
    )
    assert sig == "COUNTER_TREND_FALLING_KNIFE"
    assert "ema-50" in lesson.lower()


def test_record_trade_and_evaluate_setup_veto(temp_memory: PatternMemory):
    # 1. Record a severe loss with overbought setup
    loss_trade = TradeOutcome(
        trade_id="trade-001",
        symbol="NVDA",
        asset_class=AssetClass.EQUITY,
        side=OrderSide.BUY,
        entry_price=220.0,
        exit_price=209.0,
        quantity=20.0,
        pnl_dollars=-220.0,
        pnl_pct=-5.0,
        outcome="LOSS",
        rsi_entry=70.0,
        ema_ratio=1.04,
        volatility_atr_pct=2.1,
        market_regime="BULL_EXHAUSTION",
    )
    temp_memory.record_trade_outcome(loss_trade)

    # 2. Evaluate a new prospective trade that has an almost identical setup
    result = temp_memory.evaluate_setup(
        symbol="NVDA",
        asset_class=AssetClass.EQUITY,
        rsi=70.5,
        ema_ratio=1.04,
        atr_pct=2.15,
    )

    # Must detect high similarity and veto
    assert not result.is_safe
    assert result.risk_multiplier == 0.0
    assert result.matched_mistake_tag == "BULL_TRAP_OVERBOUGHT"
    assert result.similarity_score >= 80.0
    assert "BULL_TRAP_OVERBOUGHT" in (result.warning_message or "")


def test_evaluate_setup_safe_when_different(temp_memory: PatternMemory):
    loss_trade = TradeOutcome(
        trade_id="trade-002",
        symbol="BTC-USD",
        asset_class=AssetClass.CRYPTO,
        side=OrderSide.BUY,
        entry_price=85000.0,
        exit_price=81000.0,
        quantity=0.1,
        pnl_dollars=-400.0,
        pnl_pct=-4.7,
        outcome="LOSS",
        rsi_entry=75.0,
        ema_ratio=1.10,
        volatility_atr_pct=5.5,
    )
    temp_memory.record_trade_outcome(loss_trade)

    # Now evaluate a healthy, pull-back setup (RSI=48, EMA=1.01, low ATR)
    result = temp_memory.evaluate_setup(
        symbol="BTC-USD",
        asset_class=AssetClass.CRYPTO,
        rsi=48.0,
        ema_ratio=1.01,
        atr_pct=1.8,
    )

    assert result.is_safe
    assert result.risk_multiplier == 1.0


def test_regime_drift_detection(temp_memory: PatternMemory):
    # Add 4 consecutive losses
    for i in range(4):
        temp_memory.record_trade_outcome(
            TradeOutcome(
                trade_id=f"loss-{i}",
                symbol="ETH-USD",
                asset_class=AssetClass.CRYPTO,
                side=OrderSide.BUY,
                entry_price=2500.0,
                exit_price=2400.0,
                quantity=1.0,
                pnl_dollars=-100.0,
                pnl_pct=-4.0,
                outcome="LOSS",
                rsi_entry=60.0,
                ema_ratio=0.98,
                volatility_atr_pct=3.0,
            )
        )

    drift = temp_memory.detect_regime_drift(window_size=10)
    assert drift["drift_detected"] is True
    assert drift["status"] == "DEFENSIVE_QUARANTINE"
    assert drift["consecutive_losses"] == 4
    assert drift["win_rate_pct"] == 0.0


def test_risk_engine_integration_with_pattern_memory(temp_memory: PatternMemory):
    # Insert mistake into memory
    temp_memory.record_trade_outcome(
        TradeOutcome(
            trade_id="trade-bad",
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.BUY,
            entry_price=350.0,
            exit_price=332.0,
            quantity=10.0,
            pnl_dollars=-180.0,
            pnl_pct=-5.14,
            outcome="LOSS",
            rsi_entry=55.0,
            ema_ratio=1.01,
            volatility_atr_pct=2.2,
        )
    )

    config = BotRiskConfig(max_risk_per_trade_pct=1.5, max_position_size_pct=10.0)
    engine = RiskEngine(config=config, pattern_memory=temp_memory)

    balance = AccountBalance(cash=100_000.0, equity=100_000.0, buying_power=100_000.0, initial_day_equity=100_000.0)
    signal = TradeSignal(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        trade_date="2026-10-18",
        rating="Buy",
        action=OrderSide.BUY,
        current_price=350.0,
        stop_loss=336.0,
        confidence=0.75,
    )

    result = engine.evaluate(signal, balance, [])
    # Should be rejected because the setup matches the loss pattern in memory
    assert not result.is_approved
    assert any("VETO DE APRENDIZAJE" in r for r in result.rejection_reasons)
