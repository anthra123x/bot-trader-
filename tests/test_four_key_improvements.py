"""Unit & Integration tests for the 4 Key Improvements identified in the Performance Report.

1. Filtro de Calidad de Activo por Win Rate Histórico: Excluir activos con Win Rate < 40% de las últimas 10 operaciones.
2. Time-Stop Más Agresivo para Altcoins: Evacuar altcoins (no BTC/ETH/SOL) estancadas <= +0.5% a las 4h.
3. Dynamic Sizing por Sharpe Histórico: Asignar 2x-3x más capital a activos con alto Sharpe (ej. ETH) vs activos sangrantes/ruidosos.
4. Tighten Stop-Loss para Altcoins: Reducir stop loss default a 1.5% (max 2.5%) para altcoins vs 2.5% (max 4.0%) para majors.
"""

import time
from datetime import datetime, timezone
import pytest

from bot.config import BotRiskConfig
from bot.learning.pattern_memory import PatternMemory
from bot.learning.portfolio_intelligence import PortfolioIntelligence
from bot.models import AccountBalance, AssetClass, OrderSide, TradeOutcome, TradeSignal
from bot.risk.risk_engine import RiskEngine
from bot.symbols import is_altcoin, is_major_crypto


class TestMejora1AssetQualityFilter:
    """Tests for Improvement 1: Historical Win Rate Quality Filter."""

    def test_asset_with_poor_winrate_is_disqualified(self):
        memory = PatternMemory(db_path=":memory:")
        # Record 2 losses for LTC-USD
        for i in range(2):
            memory.record_trade_outcome(TradeOutcome(
                trade_id=f"loss-ltc-{i}",
                symbol="LTC-USD",
                asset_class=AssetClass.CRYPTO,
                side=OrderSide.BUY,
                entry_price=60.0,
                exit_price=58.0,
                quantity=1.0,
                pnl_dollars=-2.0,
                pnl_pct=-3.33,
                outcome="LOSS",
                rsi_entry=55.0,
                ema_ratio=1.0,
                volatility_atr_pct=2.0,
                market_regime="CHOPPY",
                created_at=datetime.now(timezone.utc),
            ))

        track = memory.get_asset_track_record("LTC-USD", lookback=10, min_trades=2, min_win_rate_pct=40.0)
        assert track["trade_count"] == 2
        assert track["win_rate_pct"] == 0.0
        assert track["is_disqualified"] is True

        quality_eval = memory.evaluate_asset_quality("LTC-USD")
        assert quality_eval.is_safe is False
        assert quality_eval.matched_mistake_tag == "LOW_HISTORICAL_WIN_RATE"

    def test_asset_with_insufficient_history_is_not_disqualified(self):
        memory = PatternMemory(db_path=":memory:")
        # Only 1 trade: should not disqualify prematurely
        memory.record_trade_outcome(TradeOutcome(
            trade_id="loss-btc-1",
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            side=OrderSide.BUY,
            entry_price=80000.0,
            exit_price=79000.0,
            quantity=0.01,
            pnl_dollars=-10.0,
            pnl_pct=-1.25,
            outcome="LOSS",
            rsi_entry=55.0,
            ema_ratio=1.0,
            volatility_atr_pct=2.0,
            market_regime="CHOPPY",
            created_at=datetime.now(timezone.utc),
        ))

        track = memory.get_asset_track_record("BTC-USD", lookback=10, min_trades=2, min_win_rate_pct=40.0)
        assert track["trade_count"] == 1
        assert track["is_disqualified"] is False

    def test_risk_engine_vetoes_disqualified_asset(self):
        memory = PatternMemory(db_path=":memory:")
        # 3 trades: 0 wins, 3 losses for LINK-USD
        for i in range(3):
            memory.record_trade_outcome(TradeOutcome(
                trade_id=f"loss-link-{i}",
                symbol="LINK-USD",
                asset_class=AssetClass.CRYPTO,
                side=OrderSide.BUY,
                entry_price=12.0,
                exit_price=11.5,
                quantity=10.0,
                pnl_dollars=-5.0,
                pnl_pct=-4.17,
                outcome="LOSS",
                rsi_entry=50.0,
                ema_ratio=1.0,
                volatility_atr_pct=2.0,
                market_regime="CHOPPY",
                created_at=datetime.now(timezone.utc),
            ))

        config = BotRiskConfig(enable_asset_win_rate_filter=True, min_asset_win_rate_pct=40.0, min_asset_history_trades=2)
        engine = RiskEngine(config=config, pattern_memory=memory)

        balance = AccountBalance(cash=10000.0, buying_power=10000.0, equity=10000.0, initial_day_equity=10000.0)
        signal = TradeSignal(
            symbol="LINK-USD",
            trade_date="2026-10-21",
            rating="BUY",
            action=OrderSide.BUY,
            confidence=0.85,
            target_price=14.0,
            stop_loss=11.8,
            current_price=12.0,
            asset_class=AssetClass.CRYPTO,
            metrics={"regime": "BULL_TREND", "rsi": 45.0, "atr": 0.24, "ema_ratio": 1.02},
        )

        res = engine.validate_trade(signal, balance, positions=[])
        assert res.is_approved is False
        assert any("bajo Win Rate histórico" in r for r in res.rejection_reasons)


class TestMejora2AltcoinAggressiveTimeStop:
    """Tests for Improvement 2: Aggressive 4h Time-Stop for Altcoins."""

    def test_symbol_classification(self):
        assert is_major_crypto("BTC-USD") is True
        assert is_major_crypto("ETH-USD") is True
        assert is_major_crypto("SOL-USD") is True
        assert is_altcoin("BTC-USD") is False
        assert is_altcoin("ETH-USD") is False
        assert is_altcoin("SOL-USD") is False

        assert is_altcoin("LINK-USD") is True
        assert is_altcoin("DOGE-USD") is True
        assert is_altcoin("LTC-USD") is True
        assert is_altcoin("NVDA") is False

    def test_altcoin_stagnant_at_4h_evacuated(self):
        config = BotRiskConfig(
            enable_time_stop_rotation=True,
            enable_altcoin_aggressive_time_stop=True,
            altcoin_time_stop_hours=4.0,
            altcoin_time_stop_min_profit_pct=0.5,
        )
        engine = RiskEngine(config=config)
        now = time.time()
        entry_time = now - (4.2 * 3600.0)  # 4.2 hours ago

        engine.register_position_entry("LINK-USD", entry_price=12.0, timestamp=entry_time)

        # Current price 12.03 (+0.25% gain <= +0.5% required): must evacuate
        should_exit, reason = engine.evaluate_time_stop_exit(
            symbol="LINK-USD",
            current_price=12.03,
            entry_price=12.0,
            current_qty=100.0,
        )
        assert should_exit is True
        assert "Capital Rotation Time-Stop" in reason
        assert "Altcoin LINK-USD" in reason

    def test_altcoin_with_good_gain_does_not_evacuate(self):
        config = BotRiskConfig(
            enable_time_stop_rotation=True,
            enable_altcoin_aggressive_time_stop=True,
            altcoin_time_stop_hours=4.0,
            altcoin_time_stop_min_profit_pct=0.5,
            stagnant_range_pct=1.2,
        )
        engine = RiskEngine(config=config)
        now = time.time()
        entry_time = now - (4.2 * 3600.0)

        engine.register_position_entry("LINK-USD", entry_price=12.0, timestamp=entry_time)

        # Current price 12.20 (+1.67% gain > +0.5% and > 1.2%): healthy breakout, stay in trade
        should_exit, reason = engine.evaluate_time_stop_exit(
            symbol="LINK-USD",
            current_price=12.20,
            entry_price=12.0,
            current_qty=100.0,
        )
        assert should_exit is False
        assert reason == ""


class TestMejora3DynamicSizingBySharpe:
    """Tests for Improvement 3: Dynamic Sizing scaled by Sharpe Ratio."""

    def test_high_sharpe_asset_receives_up_to_2x_capital(self):
        pi = PortfolioIntelligence(lookback_trades=20)
        # 10 consistent winning trades for ETH-USD (+1.2% each, very low std)
        records = []
        for i in range(10):
            records.append({
                "symbol": "ETH-USD",
                "pnl_pct": 1.20 + (i % 2) * 0.1,
                "outcome": "WIN",
            })
        # 5 loss trades for LINK-USD (-2.5% each)
        for i in range(5):
            records.append({
                "symbol": "LINK-USD",
                "pnl_pct": -2.50,
                "outcome": "LOSS",
            })

        pi.compute_metrics(records)

        eth_mult = pi.get_symbol_allocation_multiplier("ETH-USD")
        link_mult = pi.get_symbol_allocation_multiplier("LINK-USD")

        assert eth_mult == 2.00  # High Sharpe powerhouse
        assert link_mult == 0.50  # Bleeding asset penalty
        assert eth_mult / link_mult >= 3.0  # 4x capital ratio

    def test_normalized_symbol_lookup(self):
        pi = PortfolioIntelligence(lookback_trades=10)
        records = [{
            "symbol": "ETH-USD",
            "pnl_pct": 1.5,
            "outcome": "WIN",
        } for _ in range(5)]
        pi.compute_metrics(records)

        # Lookup with ETHUSD should find ETH-USD
        mult = pi.get_symbol_allocation_multiplier("ETHUSD")
        assert mult >= 1.25


class TestMejora4TightenedAltcoinStopLoss:
    """Tests for Improvement 4: Tighten Stop-Loss for Altcoins."""

    def test_altcoin_gets_tightened_stop_loss(self):
        config = BotRiskConfig(
            default_stop_loss_pct=2.5,
            max_stop_loss_pct=4.0,
            altcoin_default_stop_loss_pct=1.5,
            altcoin_max_stop_loss_pct=2.5,
        )
        engine = RiskEngine(config=config)

        # Altcoin returns 1.5% default and 2.5% max
        assert engine.get_default_stop_loss_pct("LINK-USD") == 1.5
        assert engine.get_default_stop_loss_pct("DOGE-USD") == 1.5
        assert engine.get_max_stop_loss_pct("LTC-USD") == 2.5

        # Major crypto returns 2.5% default and 4.0% max
        assert engine.get_default_stop_loss_pct("ETH-USD") == 2.5
        assert engine.get_default_stop_loss_pct("BTC-USD") == 2.5
        assert engine.get_max_stop_loss_pct("SOL-USD") == 4.0

        # Equity returns 2.5% default and 4.0% max
        assert engine.get_default_stop_loss_pct("NVDA") == 2.5
        assert engine.get_max_stop_loss_pct("NVDA") == 4.0

    def test_risk_engine_clamps_wide_altcoin_stop_to_2_5_pct(self):
        config = BotRiskConfig(
            default_stop_loss_pct=2.5,
            max_stop_loss_pct=4.0,
            altcoin_default_stop_loss_pct=1.5,
            altcoin_max_stop_loss_pct=2.5,
        )
        memory = PatternMemory(db_path=":memory:")
        engine = RiskEngine(config=config, pattern_memory=memory)
        balance = AccountBalance(cash=10000.0, buying_power=10000.0, equity=10000.0, initial_day_equity=10000.0)

        # Signal for DOGE with 3.5% stop loss (too wide for altcoin limit of 2.5%)
        # Current price: 0.10, requested SL: 0.0965 (-3.5%)
        signal = TradeSignal(
            symbol="DOGE-USD",
            trade_date="2026-10-21",
            rating="BUY",
            action=OrderSide.BUY,
            confidence=0.80,
            target_price=0.12,
            stop_loss=0.0965,
            current_price=0.10,
            asset_class=AssetClass.CRYPTO,
            metrics={"regime": "BULL_TREND", "rsi": 45.0, "atr": 0.002, "ema_ratio": 1.02},
        )

        res = engine.validate_trade(signal, balance, positions=[])
        assert res.is_approved is True
        # Must have clamped stop loss to 2.5% ($0.0975)
        assert res.stop_loss_price == pytest.approx(0.0975, abs=0.0001)
        assert any("Clamped to max allowable 2.5%" in w for w in res.risk_warnings)
