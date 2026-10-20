"""Unit tests for the 4 Quantitative Performance Levers.

Palanca A: Dynamic Hot Momentum Pool Discovery (HotMomentumScanner)
Palanca B: Multi-Timeframe Confluence (QuantScreener MTF)
Palanca C: Adaptive Volatility ATR Elastic Take-Profits (RiskEngine)
Palanca D: Order Book Imbalance / Level 2 Depth Analysis (OrderBookAnalyzer)
"""

from unittest.mock import MagicMock, patch
import pytest

from bot.config import BotConfig, BotRiskConfig
from bot.models import AccountBalance, AssetClass, OrderSide, TradeSignal
from bot.risk.order_book_analyzer import OrderBookAnalysis, OrderBookAnalyzer
from bot.risk.quant_screener import QuantScreener
from bot.risk.risk_engine import RiskEngine
from bot.scanner.hot_scanner import HotMomentumScanner


class TestPalancaADynamicHotScanner:
    """Tests for Palanca A: Hot Momentum Pool Discovery."""

    def test_hot_scanner_fallback_on_empty_credentials(self):
        scanner = HotMomentumScanner()
        core = ["BTC-USD", "ETH-USD"]
        mock_broker = MagicMock()
        mock_broker._headers.return_value = {}
        with patch.dict("os.environ", {}, clear=True):
            universe = scanner.discover_hot_universe(mock_broker, core, top_n=5)
            assert universe == core

    @patch("bot.scanner.hot_scanner.requests.get")
    def test_hot_scanner_ranking_and_merge(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "snapshots": {
                "SOL/USD": {
                    "dailyBar": {"c": 160.0, "v": 100000, "vw": 158.0},
                    "prevDailyBar": {"c": 140.0},  # +14.28%
                },
                "DOGE/USD": {
                    "dailyBar": {"c": 0.15, "v": 2000000, "vw": 0.14},
                    "prevDailyBar": {"c": 0.12},  # +25.0%
                },
                "BTC/USD": {
                    "dailyBar": {"c": 68000.0, "v": 5000, "vw": 67500.0},
                    "prevDailyBar": {"c": 67000.0},  # +1.49%
                },
            }
        }
        mock_get.return_value = mock_resp

        scanner = HotMomentumScanner()
        core = ["BTC-USD", "ETH-USD"]
        mock_broker = MagicMock()
        mock_broker._headers.return_value = {"Authorization": "Bearer test"}

        universe = scanner.discover_hot_universe(mock_broker, core, top_n=2, force_refresh=True)

        # DOGE-USD and SOL-USD should rank top based on momentum * liquidity
        assert "DOGE-USD" in universe[:2] or "SOL-USD" in universe[:2]
        # Core symbols should be preserved
        assert "BTC-USD" in universe
        assert "ETH-USD" in universe


class TestPalancaBMultiTimeframeConfluence:
    """Tests for Palanca B: MTF Confluence Filter."""

    def test_screener_mtf_confluence_initialization(self):
        screener = QuantScreener(enable_mtf_confluence=True, mtf_bonus_score=20.0, mtf_penalty_score=25.0)
        assert screener.enable_mtf_confluence is True
        assert screener.mtf_bonus_score == 20.0
        assert screener.mtf_penalty_score == 25.0

    @patch("bot.risk.quant_screener._verified_rows")
    def test_screener_mtf_bonus_and_metrics(self, mock_rows):
        import pandas as pd
        import numpy as np

        # Create mock price history with clear uptrend (macro bullish) and recent dip
        n = 80
        dates = pd.date_range("2026-01-01", periods=n)
        prices = np.linspace(100, 200, n)
        # Induce short-term dip at the end to trigger pullback
        prices[-3:] = [195, 190, 188]
        df = pd.DataFrame({
            "open": prices,
            "high": prices + 2,
            "low": prices - 2,
            "close": prices,
            "volume": [10000] * n,
        }, index=dates)
        mock_rows.return_value = df

        screener = QuantScreener(enable_mtf_confluence=True, conviction_threshold=30.0)
        res = screener.screen("SOL-USD", "2026-10-18")

        assert "atr_pct" in res.metrics
        assert "mtf_confluence" in res.metrics
        assert res.metrics["close"] == 188.0


class TestPalancaCAdaptiveVolatilityATRTargets:
    """Tests for Palanca C: Adaptive Volatility ATR Elastic Take-Profits."""

    def setup_method(self):
        self.config = BotRiskConfig(
            enable_adaptive_atr_targets=True,
            partial_take_profit_threshold_pct=2.2,
            stage2_take_profit_threshold_pct=4.5,
            stage3_take_profit_threshold_pct=8.0,
            atr_tp_multiplier_stage1=1.5,
            atr_tp_multiplier_stage2=3.0,
            atr_tp_multiplier_stage3=5.0,
        )
        self.engine = RiskEngine(self.config)

    def test_low_volatility_preserves_minimum_floors(self):
        # Entry with low ATR (e.g. 1.0% ATR)
        self.engine.register_entry_context("BTC-USD", {"atr_pct": 1.0})
        # Thresholds should remain at 2.2% floor (since 1.0% * 1.5 = 1.5% < 2.2%)
        # At +2.0% profit, stage 0 should NOT trigger
        qualifies, qty = self.engine.evaluate_partial_take_profit(
            symbol="BTC-USD",
            current_price=102.0,
            entry_price=100.0,
            current_qty=1.0,
        )
        assert qualifies is False

        # At +2.3% profit, stage 0 triggers at guaranteed floor
        qualifies, qty = self.engine.evaluate_partial_take_profit(
            symbol="BTC-USD",
            current_price=102.3,
            entry_price=100.0,
            current_qty=1.0,
        )
        assert qualifies is True

    def test_high_volatility_expands_elastic_thresholds(self):
        # Entry with high ATR (e.g. 4.0% ATR on SOL/DOGE)
        self.engine.register_entry_context("SOL-USD", {"atr_pct": 4.0})
        # Stage 1 elastic threshold = max(2.2%, 4.0% * 1.5) = 6.0%!
        # At +3.0% profit, position does NOT sell early; allows runner to breathe
        qualifies, qty = self.engine.evaluate_partial_take_profit(
            symbol="SOL-USD",
            current_price=103.0,
            entry_price=100.0,
            current_qty=10.0,
        )
        assert qualifies is False

        # At +6.2% profit, triggers stage 1 harvesting maximum gains
        qualifies, qty = self.engine.evaluate_partial_take_profit(
            symbol="SOL-USD",
            current_price=106.2,
            entry_price=100.0,
            current_qty=10.0,
        )
        assert qualifies is True

    def test_get_position_targets_stretches_with_atr(self):
        self.engine.register_entry_context("SOL-USD", {"atr_pct": 3.5})
        targets = self.engine.get_position_targets("SOL-USD", entry_price=100.0)
        # Stage 2 ATR stretch = 3.5 * 3.0 = 10.5% -> TP = $110.50
        assert targets["take_profit"] >= 110.0


class TestPalancaDOrderBookImbalance:
    """Tests for Palanca D: Order Book Imbalance / L2 Depth Analysis."""

    def test_analyzer_caches_and_handles_equity_safely(self):
        analyzer = OrderBookAnalyzer()
        res = analyzer.analyze("AAPL")
        assert res.imbalance_ratio == 1.0
        assert res.bias == "NEUTRAL"

    @patch("bot.risk.order_book_analyzer.requests.get")
    def test_analyzer_detects_absorption_and_overhang(self, mock_get):
        analyzer = OrderBookAnalyzer()
        mock_broker = MagicMock()
        mock_broker._headers.return_value = {"Authorization": "Bearer test"}

        # 1. Bullish Absorption: High Bids vs Low Asks
        resp_bull = MagicMock()
        resp_bull.status_code = 200
        resp_bull.json.return_value = {
            "orderbooks": {
                "BTC/USD": {
                    "b": [{"p": 68000, "s": 10.0}],  # $680,000 bids
                    "a": [{"p": 68001, "s": 2.0}],   # $136,000 asks (5.0x ratio)
                }
            }
        }
        mock_get.return_value = resp_bull
        res_bull = analyzer.analyze("BTC-USD", broker=mock_broker)
        assert res_bull.bias == "BULLISH_ABSORPTION"
        assert res_bull.imbalance_ratio >= 1.5

        # 2. Seller Overhang: Low Bids vs Huge Asks
        analyzer._cache.clear()
        resp_bear = MagicMock()
        resp_bear.status_code = 200
        resp_bear.json.return_value = {
            "orderbooks": {
                "BTC/USD": {
                    "b": [{"p": 68000, "s": 1.0}],   # $68,000 bids
                    "a": [{"p": 68001, "s": 10.0}],  # $680,000 asks (0.10x ratio)
                }
            }
        }
        mock_get.return_value = resp_bear
        res_bear = analyzer.analyze("BTC-USD", broker=mock_broker)
        assert res_bear.bias == "SELLER_OVERHANG"
        assert res_bear.imbalance_ratio <= 0.60

    def test_risk_engine_clamps_size_on_seller_overhang(self):
        config = BotRiskConfig(
            enable_order_book_imbalance=True,
            max_seller_overhang_ratio=0.60,
            max_position_size_pct=20.0,
            enable_meta_labeling=False,
        )
        engine = RiskEngine(config)
        balance = AccountBalance(cash=10000.0, equity=10000.0, buying_power=10000.0)

        # Baseline signal without overhang
        normal_signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="BUY",
            action=OrderSide.BUY,
            confidence=1.0,
            current_price=100.0,
            stop_loss=96.0,
            metrics={"l2_imbalance_ratio": 1.5},
        )
        res_normal = engine.evaluate(normal_signal, balance, positions=[])
        assert res_normal.is_approved is True

        # Signal with severe seller overhang (0.40x)
        overhang_signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="BUY",
            action=OrderSide.BUY,
            confidence=1.0,
            current_price=100.0,
            stop_loss=96.0,
            metrics={"l2_imbalance_ratio": 0.40},
        )
        res_overhang = engine.evaluate(overhang_signal, balance, positions=[])
        assert res_overhang.is_approved is True
        # Quantity should be reduced to 55% of normal
        assert res_overhang.adjusted_quantity < res_normal.adjusted_quantity
        assert pytest.approx(res_overhang.adjusted_quantity, rel=1e-2) == round(res_normal.adjusted_quantity * 0.55, 6)

    def test_runner_identifies_discovered_crypto_assets(self):
        from bot.runner import BotRunner
        runner = BotRunner()
        assert runner.identify_asset_class("NEAR-USD") == AssetClass.CRYPTO
        assert runner.identify_asset_class("SOL/USD") == AssetClass.CRYPTO
        assert runner.identify_asset_class("BTC-USD") == AssetClass.CRYPTO
        assert runner.identify_asset_class("AAPL") == AssetClass.EQUITY
        assert runner.identify_asset_class("NVDA") == AssetClass.EQUITY


class TestIntelligentTimeStopRotation:
    """Tests for Opportunity Cost Time-Stop & Capital Rotation Guard."""

    def test_crypto_stagnant_position_triggers_time_stop(self):
        import time
        config = BotRiskConfig(
            enable_time_stop_rotation=True,
            max_crypto_stagnant_hours=4.0,
            stagnant_range_pct=1.2,
        )
        engine = RiskEngine(config=config)
        now = time.time()
        entry_time = now - (4.5 * 3600.0)  # 4.5 hours ago

        engine.register_position_entry("DOGE-USD", entry_price=0.0800, timestamp=entry_time)

        # Current price: 0.07936 (-0.8% PnL, trapped in stagnation band)
        should_exit, reason = engine.evaluate_time_stop_exit(
            symbol="DOGE-USD",
            current_price=0.07936,
            entry_price=0.0800,
            current_qty=5000.0,
        )
        assert should_exit is True
        assert "Capital Rotation Time-Stop" in reason
        assert "DOGE-USD" in reason

    def test_crypto_fresh_position_does_not_trigger_time_stop(self):
        import time
        config = BotRiskConfig(
            enable_time_stop_rotation=True,
            max_crypto_stagnant_hours=4.0,
            stagnant_range_pct=1.2,
        )
        engine = RiskEngine(config=config)
        now = time.time()
        entry_time = now - (1.5 * 3600.0)  # Only 1.5 hours ago

        engine.register_position_entry("SOL-USD", entry_price=105.0, timestamp=entry_time)

        should_exit, reason = engine.evaluate_time_stop_exit(
            symbol="SOL-USD",
            current_price=104.2,  # -0.76% PnL
            entry_price=105.0,
            current_qty=10.0,
        )
        assert should_exit is False
        assert reason == ""

    def test_crypto_profitable_runner_does_not_trigger_time_stop(self):
        import time
        config = BotRiskConfig(
            enable_time_stop_rotation=True,
            max_crypto_stagnant_hours=4.0,
            stagnant_range_pct=1.2,
        )
        engine = RiskEngine(config=config)
        now = time.time()
        entry_time = now - (5.0 * 3600.0)  # 5.0 hours ago

        engine.register_position_entry("ETH-USD", entry_price=2500.0, timestamp=entry_time)

        # Current price: 2580.0 (+3.2% PnL, breakout above stagnation band)
        should_exit, reason = engine.evaluate_time_stop_exit(
            symbol="ETH-USD",
            current_price=2580.0,
            entry_price=2500.0,
            current_qty=2.0,
        )
        assert should_exit is False

    def test_equity_position_immune_to_crypto_time_stop(self):
        import time
        config = BotRiskConfig(
            enable_time_stop_rotation=True,
            max_crypto_stagnant_hours=4.0,
            stagnant_range_pct=1.2,
        )
        engine = RiskEngine(config=config)
        now = time.time()
        entry_time = now - (8.0 * 3600.0)  # 8.0 hours ago

        engine.register_position_entry("NVDA", entry_price=216.84, timestamp=entry_time)

        # NVDA is an equity; closed overnight market hours must NOT trigger stagnation exit
        should_exit, reason = engine.evaluate_time_stop_exit(
            symbol="NVDA",
            current_price=216.80,
            entry_price=216.84,
            current_qty=50.0,
        )
        assert should_exit is False

    def test_time_stop_exit_does_not_trigger_stoploss_guard_circuit_breaker(self):
        config = BotRiskConfig(cooldown_period_seconds=1800)
        engine = RiskEngine(config=config)

        # Register exit as time stop rotation (is_loss=True, is_stoploss_event=False)
        engine.register_trade_exit("DOGE-USD", is_loss=True, is_stoploss_event=False)

        # Specific symbol is in cooldown to prevent immediate churn
        in_cd, _ = engine.is_in_cooldown("DOGE-USD")
        assert in_cd is True

        # But recent stoploss events list is EMPTY so circuit breaker is not triggered!
        assert len(engine._recent_stoploss_events) == 0


class TestPrecisionAgainstRandomness:
    """Tests for the 4 Precision Pillars Against Market Randomness."""

    def test_hurst_exponent_fractal_noise_detection(self):
        import numpy as np
        from bot.risk.quant_screener import calculate_hurst_exponent

        # 1. Random Walk / Brownian noise (H approx 0.50)
        np.random.seed(42)
        rw = np.cumsum(np.random.randn(200)) + 100
        h_rw = calculate_hurst_exponent(rw)
        assert 0.40 <= h_rw <= 0.55

        # 2. Trending persistent series (H > 0.52)
        trend = np.linspace(100, 300, 200) + np.random.randn(200) * 0.5
        h_trend = calculate_hurst_exponent(trend)
        assert h_trend >= 0.52

    def test_volume_delta_flow_trap_divergence_detection(self):
        from bot.risk.quant_screener import calculate_volume_delta_flow

        # Case 1: Healthy volume expansion in uptrend
        closes = [100.0 + i for i in range(20)]
        volumes = [1000.0] * 20
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        flow, is_trap = calculate_volume_delta_flow(closes, volumes, highs, lows)
        assert is_trap is False

        # Case 2: Bull trap exhaustion (price reaches new high, but closes at bottom of bars)
        trap_closes = [100.0 + i for i in range(20)]
        trap_highs = [c + 3.0 for c in trap_closes]
        trap_lows = [c - 0.1 for c in trap_closes]  # Closed right at the bottom
        trap_flow, is_trap = calculate_volume_delta_flow(trap_closes, volumes, trap_highs, trap_lows)
        assert is_trap is True
        assert trap_flow < -0.15

    def test_meta_labeling_quality_gate(self):
        config = BotRiskConfig(
            enable_meta_labeling=True,
            min_meta_probability=0.62,
        )
        engine = RiskEngine(config=config)

        # High-conviction setup (Persistent trend H=0.58, L2=1.30, Bullish CVD)
        good_signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="BUY",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=100.0,
            metrics={
                "hurst_exponent": 0.58,
                "volume_delta_flow": 0.35,
                "l2_imbalance_ratio": 1.30,
            },
        )
        is_ok, p_success, reason = engine.evaluate_meta_labeling(good_signal, market_regime="TRENDING_BULL_STABLE")
        assert is_ok is True
        assert p_success >= 0.62

        # Low-probability setup trapped in random walk noise (H=0.46, seller overhang L2=0.55, negative CVD)
        noisy_signal = TradeSignal(
            symbol="DOGE-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="BUY",
            action=OrderSide.BUY,
            confidence=0.55,
            current_price=0.08,
            metrics={
                "hurst_exponent": 0.46,
                "volume_delta_flow": -0.25,
                "is_cvd_divergent": 1.0,
                "l2_imbalance_ratio": 0.55,
            },
        )
        is_ok, p_success, reason = engine.evaluate_meta_labeling(noisy_signal, market_regime="CHOPPY_RANGE_BOUND")
        assert is_ok is False
        assert p_success < 0.62
        assert "Meta-Labeling" in reason

    def test_extended_quarantine_on_consecutive_losses(self):
        config = BotRiskConfig(
            cooldown_period_seconds=1800,  # 30m standard
            enable_extended_symbol_quarantine=True,
            symbol_consecutive_losses_threshold=2,
            extended_quarantine_seconds=14400,  # 4h extended
        )
        engine = RiskEngine(config=config)

        # Trade 1: Loss 1 on SOL-USD
        engine.register_trade_exit("SOL-USD", is_loss=True)
        in_cd, remaining = engine.is_in_cooldown("SOL-USD")
        assert in_cd is True
        assert remaining <= 1800.0  # Standard 30m cooldown

        # Trade 2: Loss 2 on SOL-USD -> triggers 4-hour extended quarantine
        engine.register_trade_exit("SOL-USD", is_loss=True)
        in_cd, remaining = engine.is_in_cooldown("SOL-USD")
        assert in_cd is True
        assert remaining > 1800.0  # Extended beyond 30m!
        assert remaining <= 14400.0

        # Trade 3: Other assets (BTC-USD) are NOT in quarantine
        in_cd_btc, _ = engine.is_in_cooldown("BTC-USD")
        assert in_cd_btc is False

        # Reset streak on winning trade
        engine.register_trade_exit("SOL-USD", is_loss=False)
        assert "SOLUSD" not in engine._symbol_loss_streak

