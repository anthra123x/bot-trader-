"""Unit and integration tests for Phase 3: Evolution (ParamOptimizer & PortfolioIntelligence).

Tests:
- Rolling Sharpe, Sortino, Max Drawdown, Expectancy calculations
- Per-symbol dynamic capital allocation multipliers
- Portfolio degradation & Circuit Breaker HALT trigger
- Bayesian parameter optimization via Optuna TPE
- Dynamic application of calibrated parameters to BotRiskConfig
- RiskEngine sizing integration with PortfolioIntelligence
"""

import os
from pathlib import Path
import numpy as np
import pytest

import unittest.mock
from bot.config import BotConfig, BotRiskConfig
from bot.learning.param_optimizer import PARAM_OPTIMIZER, ParamOptimizer
from bot.learning.portfolio_intelligence import (
    PORTFOLIO_INTELLIGENCE,
    PortfolioIntelligence,
)
from bot.models import (
    AccountBalance,
    AssetClass,
    OrderSide,
    PatternCheckResult,
    TradeSignal,
)
from bot.risk.risk_engine import RiskEngine


def test_portfolio_intelligence_empty():
    pi = PortfolioIntelligence()
    rep = pi.compute_metrics([])
    assert rep.total_trades_analyzed == 0
    assert rep.health_status == "STABLE"
    assert rep.sizing_multiplier == 1.0


def test_portfolio_intelligence_metrics_and_symbols():
    pi = PortfolioIntelligence(lookback_trades=30)
    # 20 trades for BTC-USD (mostly wins: +4%, +3%, -1%, +5%)
    records = []
    for i in range(20):
        is_win = (i % 4 != 0)
        pnl = 3.5 if is_win else -1.2
        records.append({
            "symbol": "BTC-USD",
            "pnl_pct": pnl,
            "outcome": "WIN" if is_win else "LOSS",
        })

    # 10 trades for SOL-USD (mostly losses: -3.0%, -2.5%, +0.5%)
    for i in range(10):
        is_win = (i % 4 == 0)
        pnl = 1.0 if is_win else -2.5
        records.append({
            "symbol": "SOL-USD",
            "pnl_pct": pnl,
            "outcome": "WIN" if is_win else "LOSS",
        })

    rep = pi.compute_metrics(records)
    assert rep.total_trades_analyzed == 30
    assert rep.rolling_win_rate > 50.0
    assert rep.rolling_sharpe > 0.0
    assert rep.rolling_sortino > 0.0
    assert rep.mathematical_expectancy > 0.0

    # BTC-USD should have an allocation boost (> 1.0)
    btc_mult = pi.get_symbol_allocation_multiplier("BTC-USD")
    assert btc_mult >= 1.25

    # SOL-USD should have an allocation penalty (< 1.0)
    sol_mult = pi.get_symbol_allocation_multiplier("SOL-USD")
    assert sol_mult <= 0.60


def test_portfolio_intelligence_circuit_breaker_halt():
    pi = PortfolioIntelligence(lookback_trades=20)
    # Severe losing streak: 18 losses of -3.5%, 2 small wins
    records = []
    for i in range(20):
        pnl = -3.5 if i < 18 else 0.5
        records.append({
            "symbol": "ETH-USD",
            "pnl_pct": pnl,
            "outcome": "LOSS" if i < 18 else "WIN",
        })

    rep = pi.compute_metrics(records)
    assert rep.health_status == "HALT_RECOMMENDED"
    assert rep.sizing_multiplier == 0.0
    assert len(rep.warnings) > 0
    assert "Circuit breaker" in rep.warnings[0]


def test_param_optimizer_bayesian_tuning(tmp_path: Path):
    cache_file = str(tmp_path / "test_optimized_params.json")
    optimizer = ParamOptimizer(min_samples_to_optimize=15, cache_path=cache_file)

    # Synthesize 30 trades with MAE, MFE, RSI, and screener scores
    records = []
    for i in range(30):
        is_win = (i % 2 == 0)
        records.append({
            "rsi_entry": 56.0 if is_win else 68.0,
            "mae_pct": 0.8 if is_win else 3.2,
            "mfe_pct": 5.5 if is_win else 0.8,
            "pnl_pct": 4.0 if is_win else -2.5,
            "features": {"screener_score": 75.0 if is_win else 45.0},
        })

    best_params = optimizer.optimize_parameters(records, n_trials=25)
    assert len(best_params) > 0
    assert "default_stop_loss_pct" in best_params
    assert "min_risk_reward_ratio" in best_params
    assert 1.5 <= best_params["default_stop_loss_pct"] <= 4.5
    assert 1.8 <= best_params["min_risk_reward_ratio"] <= 3.5

    # Test apply to BotRiskConfig
    cfg = BotRiskConfig()
    applied = optimizer.apply_to_config(cfg, best_params)
    assert applied >= 5
    assert cfg.default_stop_loss_pct == best_params["default_stop_loss_pct"]
    assert cfg.min_risk_reward_ratio == best_params["min_risk_reward_ratio"]


def test_risk_engine_integration_with_portfolio_intelligence(tmp_path: Path):
    engine = RiskEngine(config=BotRiskConfig())

    # Pre-configure PortfolioIntelligence with an optimal BTC-USD track record
    records = [
        {"symbol": "BTC-USD", "pnl_pct": 4.0, "outcome": "WIN"},
        {"symbol": "BTC-USD", "pnl_pct": 3.0, "outcome": "WIN"},
        {"symbol": "BTC-USD", "pnl_pct": -1.0, "outcome": "LOSS"},
        {"symbol": "BTC-USD", "pnl_pct": 5.0, "outcome": "WIN"},
    ]
    PORTFOLIO_INTELLIGENCE.compute_metrics(records)

    bal = AccountBalance(cash=100_000.0, equity=100_000.0, buying_power=100_000.0)
    signal = TradeSignal(
        symbol="BTC-USD",
        asset_class=AssetClass.CRYPTO,
        trade_date="2026-10-20",
        rating="Buy",
        action=OrderSide.BUY,
        confidence=0.85,
        current_price=90_000.0,
        metrics={
            "rsi": 52.0,
            "hurst_exponent": 0.60,
            "volume_delta_flow": 0.40,
            "l2_imbalance_ratio": 1.40,
        },
    )

    with unittest.mock.patch.object(
        engine.pattern_memory,
        "evaluate_setup",
        return_value=PatternCheckResult(is_safe=True, risk_multiplier=1.0),
    ):
        res = engine.evaluate(signal, bal, [])

    assert res.is_approved is True
    assert res.adjusted_quantity > 0
    # Warning should indicate Alpha Edge boost
    edge_warnings = [w for w in res.risk_warnings if "Alpha Edge" in w]
    assert len(edge_warnings) > 0
