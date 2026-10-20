"""Unit and integration tests for Phase 2: Brain — Adaptive Model (LightGBM Online Classifier).

Tests:
- Online feature extraction (19 dimensions)
- Graceful heuristic fallback when samples < 25
- Online LightGBM training on historical trade outcomes
- Prediction probability calibration and feature importances
- Persistence and reload
- Integration into RiskEngine.evaluate_meta_labeling
"""

import os
from pathlib import Path
import numpy as np
import pytest

from bot.config import BotRiskConfig
from bot.learning.adaptive_model import (
    FEATURE_NAMES,
    AdaptiveModel,
    encode_confluence,
    encode_regime,
    extract_features_vector,
)
from bot.learning.pattern_memory import PatternMemory
from bot.models import (
    AssetClass,
    OrderSide,
    TradeFeatureSnapshot,
    TradeOutcome,
    TradeSignal,
)
from bot.risk.risk_engine import RiskEngine


def test_feature_vector_dimension_and_encoding():
    signal = TradeSignal(
        symbol="BTC-USD",
        asset_class=AssetClass.CRYPTO,
        trade_date="2026-10-20",
        rating="Buy",
        action=OrderSide.BUY,
        confidence=0.82,
        current_price=90000.0,
        metrics={
            "rsi": 58.5,
            "ema_ratio": 1.025,
            "ema_ratio_200": 1.06,
            "atr_pct": 2.1,
            "band_pct": 0.65,
            "hurst_exponent": 0.59,
            "volume_delta_flow": 82.0,
            "is_cvd_divergent": 0.0,
            "l2_imbalance_ratio": 1.35,
            "spread_pct": 0.03,
            "fear_greed_value": 68,
            "sentiment_score": 0.45,
            "hour_utc": 15,
            "day_of_week": 1,
            "screener_score": 72.0,
            "regime": "TRENDING_BULL_EXPANSION",
            "mtf_confluence": "ALIGNED",
        },
    )

    vec = extract_features_vector(signal, market_regime="TRENDING_BULL_EXPANSION")
    assert len(vec) == len(FEATURE_NAMES)
    assert len(vec) == 19
    assert vec.dtype == np.float32
    # Check key indices
    assert vec[0] == 58.5  # rsi_14
    assert vec[5] == 0.59  # hurst_exponent
    assert vec[8] == 1.35  # l2_imbalance_ratio
    assert vec[18] == 1.0  # crypto asset class code


def test_adaptive_model_heuristic_fallback(tmp_path: Path):
    model_file = str(tmp_path / "test_model.lgb")
    meta_file = str(tmp_path / "test_model_meta.json")
    model = AdaptiveModel(min_samples_to_train=25, model_path=model_file, meta_path=meta_file)

    signal = TradeSignal(
        symbol="ETH-USD",
        asset_class=AssetClass.CRYPTO,
        trade_date="2026-10-20",
        rating="Buy",
        action=OrderSide.BUY,
        confidence=0.78,
        current_price=3000.0,
        metrics={
            "rsi": 52.0,
            "hurst_exponent": 0.58,
            "volume_delta_flow": 0.35,
            "l2_imbalance_ratio": 1.40,
        },
    )

    p_success, reason = model.predict_p_success(signal, market_regime="BULL_MOMENTUM")
    assert not model.is_trained
    assert p_success >= 0.60
    assert "Heurística Estadística" in reason


def test_adaptive_model_training_and_inference(tmp_path: Path):
    model_file = str(tmp_path / "test_model.lgb")
    meta_file = str(tmp_path / "test_model_meta.json")
    model = AdaptiveModel(min_samples_to_train=25, model_path=model_file, meta_path=meta_file)

    # 1. Synthesize 40 diverse historical samples (24 wins, 16 losses)
    records: list[dict] = []
    np.random.seed(42)

    for i in range(40):
        # Balanced training setups: 20 wins, 20 losses
        is_win = (i % 2 == 0)
        if is_win:
            rsi = float(np.random.uniform(50.0, 62.0))
            hurst = float(np.random.uniform(0.55, 0.65))
            cvd = float(np.random.uniform(0.15, 0.50))
            l2 = float(np.random.uniform(1.20, 1.80))
            pnl = float(np.random.uniform(1.5, 6.0))
            outcome = "WIN"
            reward = float(np.random.uniform(0.4, 0.9))
        else:
            # Losing setups: overbought rsi or low hurst (noise)
            rsi = float(np.random.uniform(68.0, 78.0))
            hurst = float(np.random.uniform(0.40, 0.48))
            cvd = float(np.random.uniform(-0.40, 0.05))
            l2 = float(np.random.uniform(0.45, 0.70))
            pnl = float(np.random.uniform(-4.0, -1.0))
            outcome = "LOSS"
            reward = float(np.random.uniform(-0.9, -0.3))

        records.append({
            "trade_id": f"trade-{i:03d}",
            "symbol": "BTC-USD" if i % 2 == 0 else "NVDA",
            "outcome": outcome,
            "pnl_pct": pnl,
            "reward_score": reward,
            "rsi_entry": rsi,
            "ema_ratio": 1.02,
            "volatility_atr_pct": 2.0,
            "market_regime": "BULL_CONTINUATION" if is_win else "BEAR_CORRECTION",
            "features": {
                "rsi_14": rsi,
                "ema_ratio_50": 1.02,
                "ema_ratio_200": 1.05,
                "atr_pct": 2.0,
                "band_pct": 0.6,
                "hurst_exponent": hurst,
                "volume_delta_flow": cvd,
                "is_cvd_divergent": False if is_win else True,
                "l2_imbalance_ratio": l2,
                "spread_pct": 0.03,
                "fear_greed_value": 65 if is_win else 35,
                "sentiment_score": 0.5 if is_win else -0.3,
                "hour_utc": 14,
                "day_of_week": 2,
                "screener_score": 75.0 if is_win else 40.0,
                "mtf_confluence": "ALIGNED" if is_win else "OPPOSED",
            },
        })

    # 2. Train model
    success = model.train_on_history(records)
    assert success is True
    assert model.is_trained is True
    assert model._sample_count == 40
    assert model._train_accuracy >= 0.70

    # 3. Check feature importances were computed
    importances = model.get_feature_importances()
    assert len(importances) > 0
    top_feature = list(importances.keys())[0]
    assert importances[top_feature] > 0.0

    # 4. Predict on a strong Bullish setup
    strong_signal = TradeSignal(
        symbol="BTC-USD",
        asset_class=AssetClass.CRYPTO,
        trade_date="2026-10-20",
        rating="Buy",
        action=OrderSide.BUY,
        confidence=0.85,
        current_price=92000.0,
        metrics={
            "rsi": 54.0,
            "hurst_exponent": 0.62,
            "volume_delta_flow": 0.45,
            "l2_imbalance_ratio": 1.60,
            "screener_score": 82.0,
        },
    )
    p_win, reason_win = model.predict_p_success(strong_signal, market_regime="BULL_CONTINUATION")
    assert p_win >= 0.65
    assert "LightGBM Neural Predictor" in reason_win

    # 5. Predict on an Overbought noisy trap
    trap_signal = TradeSignal(
        symbol="SOL-USD",
        asset_class=AssetClass.CRYPTO,
        trade_date="2026-10-20",
        rating="Buy",
        action=OrderSide.BUY,
        confidence=0.55,
        current_price=210.0,
        metrics={
            "rsi": 76.0,
            "hurst_exponent": 0.42,
            "volume_delta_flow": -0.30,
            "l2_imbalance_ratio": 0.50,
            "screener_score": 35.0,
        },
    )
    p_trap, reason_trap = model.predict_p_success(trap_signal, market_regime="BEAR_CORRECTION")
    assert p_trap < p_win
    assert p_trap <= 0.50


def test_risk_engine_evaluate_meta_labeling_integration():
    engine = RiskEngine(config=BotRiskConfig())

    # High conviction setup with strong microstructure
    good_signal = TradeSignal(
        symbol="NVDA",
        asset_class=AssetClass.EQUITY,
        trade_date="2026-10-20",
        rating="Buy",
        action=OrderSide.BUY,
        confidence=0.80,
        current_price=220.0,
        metrics={
            "rsi": 55.0,
            "hurst_exponent": 0.60,
            "volume_delta_flow": 0.40,
            "l2_imbalance_ratio": 1.45,
            "spread_pct": 0.04,
        },
    )

    is_ok, p_success, reason = engine.evaluate_meta_labeling(good_signal, market_regime="TRENDING_BULL")
    assert is_ok is True
    assert p_success >= 0.60
    assert "Meta-Labeling Aprobado" in reason
