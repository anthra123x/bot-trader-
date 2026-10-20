"""Unit and integration tests for Phase 1 Neural Architecture Foundation:
- TradeFeatureSnapshot serialization & models
- RewardCalculator continuous mathematical reward function
- RiskEngine trough/peak excursion tracking (MAE / MFE)
- PatternMemory schema migration & enriched historical matrix
- Runner trade outcome neural feedback integration
"""

import os
from pathlib import Path
import pytest

from bot.config import BotRiskConfig
from bot.learning.pattern_memory import PatternMemory
from bot.learning.reward_calculator import REWARD_CALCULATOR, RewardCalculator
from bot.models import AssetClass, OrderSide, TradeFeatureSnapshot, TradeOutcome
from bot.risk.risk_engine import RiskEngine


def test_trade_feature_snapshot_defaults_and_serialization():
    snapshot = TradeFeatureSnapshot(
        rsi_14=62.5,
        ema_ratio_50=1.03,
        ema_ratio_200=1.08,
        atr_pct=2.4,
        band_pct=0.75,
        hurst_exponent=0.62,
        volume_delta_flow=145.2,
        is_cvd_divergent=True,
        l2_imbalance_ratio=1.45,
        spread_pct=0.04,
        market_regime="BULL_MOMENTUM",
        prudence_level="OFFENSIVE",
        fear_greed_value=72,
        sentiment_score=0.85,
        hour_utc=14,
        day_of_week=1,
        mtf_confluence="ALIGNED",
        screener_score=78.5,
        screener_setup="BULLISH_CVD_ABSORPTION",
        position_pct_of_equity=4.5,
        existing_positions_count=2,
    )
    d = snapshot.model_dump()
    assert d["rsi_14"] == 62.5
    assert d["hurst_exponent"] == 0.62
    assert d["is_cvd_divergent"] is True
    assert d["l2_imbalance_ratio"] == 1.45
    assert d["market_regime"] == "BULL_MOMENTUM"


def test_reward_calculator_clean_win():
    # Big win with almost zero drawdown (MAE = 0.2%), peak = 4.5%, exit = 4.2%
    res = REWARD_CALCULATOR.calculate_breakdown(
        pnl_pct=4.2,
        volatility_atr_pct=2.0,
        mae_pct=0.2,
        mfe_pct=4.5,
        stop_loss_pct=2.5,
        duration_hours=2.0,
    )
    assert res.total_reward > 0.65
    assert res.pnl_component > 0.8
    assert res.mae_component > 0.6
    assert res.efficiency_component > 0.7


def test_reward_calculator_poor_exit_efficiency():
    # Trade was up +8% at peak, but held until it crashed down to +0.5%
    res_poor = REWARD_CALCULATOR.calculate_breakdown(
        pnl_pct=0.5,
        volatility_atr_pct=2.0,
        mae_pct=1.0,
        mfe_pct=8.0,
        stop_loss_pct=2.5,
        duration_hours=5.0,
    )
    # Trade that captured +2.5% when peak was +2.6%
    res_good = REWARD_CALCULATOR.calculate_breakdown(
        pnl_pct=2.5,
        volatility_atr_pct=2.0,
        mae_pct=0.3,
        mfe_pct=2.6,
        stop_loss_pct=2.5,
        duration_hours=2.0,
    )
    assert res_good.total_reward > res_poor.total_reward
    assert res_good.efficiency_component > res_poor.efficiency_component


def test_reward_calculator_stop_loss_hit():
    # Full stop loss hit: pnl = -2.5%, mae = 2.5%
    res = REWARD_CALCULATOR.calculate_breakdown(
        pnl_pct=-2.5,
        volatility_atr_pct=2.0,
        mae_pct=2.5,
        mfe_pct=0.2,
        stop_loss_pct=2.5,
        duration_hours=1.5,
    )
    assert res.total_reward < -0.6
    assert res.pnl_component < -0.7
    assert res.mae_component <= -0.9


def test_reward_calculator_stagnation_penalty():
    # Stagnant flat trade (+0.05%) held for 28 hours
    res_stagnant = REWARD_CALCULATOR.calculate_breakdown(
        pnl_pct=0.05,
        volatility_atr_pct=2.0,
        mae_pct=0.3,
        mfe_pct=0.4,
        stop_loss_pct=2.5,
        duration_hours=28.0,
    )
    assert res_stagnant.opportunity_penalty < 0.0


def test_risk_engine_excursion_tracking_mae_mfe(tmp_path: Path):
    db_file = tmp_path / "test_risk_engine.db"
    pm = PatternMemory(str(db_file))
    engine = RiskEngine(config=BotRiskConfig(), pattern_memory=pm)

    # 1. Register entry for SOL-USD at $180.0
    engine.register_position_entry("SOL-USD", entry_price=180.0)

    # 2. Price fluctuates: drops to $176.0 (trough), then rallies to $195.0 (peak)
    engine.update_position_extremes("SOL-USD", current_price=176.0, entry_price=180.0)
    engine.update_position_extremes("SOL-USD", current_price=195.0, entry_price=180.0)
    engine.update_position_extremes("SOL-USD", current_price=191.0, entry_price=180.0)

    # 3. Calculate excursions at exit price $191.0 for qty = 5.0
    excursions = engine.get_position_excursions("SOL-USD", entry_price=180.0, current_price=191.0, quantity=5.0)

    # Peak was $195.0 -> MFE = 195 - 180 = $15.0 per unit -> $75.0 total, MFE% = 8.33%
    assert round(excursions["peak_price"], 2) == 195.0
    assert round(excursions["mfe_dollars"], 2) == 75.0
    assert round(excursions["mfe_pct"], 2) == pytest.approx(8.33, rel=1e-2)

    # Trough was $176.0 -> MAE = 180 - 176 = $4.0 per unit -> $20.0 total, MAE% = 2.22%
    assert round(excursions["trough_price"], 2) == 176.0
    assert round(excursions["mae_dollars"], 2) == 20.0
    assert round(excursions["mae_pct"], 2) == pytest.approx(2.22, rel=1e-2)

    # 4. Clean exit pops state
    engine.register_trade_exit("SOL-USD", is_loss=False)
    norm_sym = "SOLUSD"
    assert norm_sym not in engine._peak_prices
    assert norm_sym not in engine._trough_prices


def test_pattern_memory_enriched_schema_and_matrix(tmp_path: Path):
    db_file = tmp_path / "test_neural_patterns.db"
    pm = PatternMemory(str(db_file))

    snapshot = TradeFeatureSnapshot(
        rsi_14=58.0,
        ema_ratio_50=1.02,
        ema_ratio_200=1.05,
        atr_pct=1.8,
        band_pct=0.6,
        hurst_exponent=0.58,
        volume_delta_flow=85.0,
        is_cvd_divergent=False,
        l2_imbalance_ratio=1.2,
        spread_pct=0.03,
        market_regime="BULL_CONTINUATION",
        prudence_level="BALANCED",
        fear_greed_value=60,
        sentiment_score=0.4,
        hour_utc=15,
        day_of_week=2,
        mtf_confluence="ALIGNED",
        screener_score=68.0,
        screener_setup="MOMENTUM_PULLBACK",
        position_pct_of_equity=3.0,
        existing_positions_count=1,
    )

    outcome = TradeOutcome(
        trade_id="neural-001",
        symbol="BTC-USD",
        asset_class=AssetClass.CRYPTO,
        side=OrderSide.BUY,
        entry_price=90000.0,
        exit_price=93500.0,
        quantity=0.1,
        pnl_dollars=350.0,
        pnl_pct=3.89,
        outcome="WIN",
        duration_hours=4.5,
        rsi_entry=58.0,
        ema_ratio=1.02,
        volatility_atr_pct=1.8,
        market_regime="BULL_CONTINUATION",
        mae_dollars=45.0,
        mae_pct=0.50,
        mfe_dollars=380.0,
        mfe_pct=4.22,
        reward_score=0.742,
        exit_reason="TRAILING_STOP",
        features=snapshot,
    )

    pm.record_trade_outcome(outcome)

    matrix = pm.get_historical_feature_matrix(limit=10)
    assert len(matrix) >= 1
    rec = matrix[0]
    assert rec["trade_id"] == "neural-001"
    assert rec["reward_score"] == pytest.approx(0.742, abs=1e-3)
    assert rec["mae_pct"] == pytest.approx(0.50, abs=1e-3)
    assert rec["mfe_pct"] == pytest.approx(4.22, abs=1e-3)
    assert rec["exit_reason"] == "TRAILING_STOP"
    assert rec["features"]["hurst_exponent"] == 0.58
    assert rec["features"]["screener_setup"] == "MOMENTUM_PULLBACK"
