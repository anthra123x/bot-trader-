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

    def test_empirical_meta_labeling_threshold_calibration(self):
        from bot.learning.pattern_memory import PatternMemory
        from bot.models import TradeOutcome
        import uuid

        # Create in-memory pattern memory
        memory = PatternMemory(db_path=":memory:")
        config = BotRiskConfig(
            enable_meta_labeling=True,
            min_meta_probability=0.62,
        )
        engine = RiskEngine(config=config, pattern_memory=memory)

        # 1. Base threshold initially (< 3 trades)
        base_t, reason = engine.get_calibrated_meta_threshold()
        assert base_t == 0.62

        # 2. Simulate 3 consecutive losses -> triggers DEFENSIVE_QUARANTINE
        for sym in ["SOL-USD", "DOGE-USD", "LTC-USD"]:
            outcome = TradeOutcome(
                trade_id=str(uuid.uuid4())[:8],
                symbol=sym,
                asset_class=AssetClass.CRYPTO,
                side=OrderSide.BUY,
                entry_price=100.0,
                exit_price=97.0,
                quantity=1.0,
                pnl_dollars=-3.0,
                pnl_pct=-3.0,
                outcome="LOSS",
                market_regime="CHOPPY_RANGE_BOUND",
            )
            memory.record_trade_outcome(outcome)

        # Clear cache to force real-time recalibration
        engine._meta_calib_cache = (0.0, 0.0, "")
        tightened_t, reason = engine.get_calibrated_meta_threshold()
        # Should ratchet UP to ~0.69 (base 0.62 + 0.07)
        assert tightened_t > 0.62
        assert tightened_t == pytest.approx(0.69, rel=1e-2)
        assert "mayor exigencia" in reason


class TestSystemicEnhancementsAndSanitization:
    """Tests for the 4 critical production hardening improvements."""

    def test_alpaca_order_quantity_normalization(self):
        from bot.brokers.alpaca import AlpacaBroker
        broker = AlpacaBroker()

        # BTC: 4 decimals (step 0.0001)
        assert broker.normalize_order_quantity("BTC-USD", 0.0123987, is_crypto=True) == 0.0123
        # ETH: 3 decimals (step 0.001)
        assert broker.normalize_order_quantity("ETH/USD", 1.4569, is_crypto=True) == 1.456
        # SOL: 2 decimals (step 0.01)
        assert broker.normalize_order_quantity("SOL-USD", 15.6789, is_crypto=True) == 15.67
        # DOGE: 0 decimals (step 1.0)
        assert broker.normalize_order_quantity("DOGE-USD", 250.85, is_crypto=True) == 250.0
        # Equities: integer shares
        assert broker.normalize_order_quantity("NVDA", 12.8, is_crypto=False) == 12.0
        assert broker.normalize_order_quantity("AAPL", -5.0, is_crypto=False) == 0.0

    @patch("bot.brokers.alpaca.requests.post")
    def test_alpaca_crypto_order_sanitization_removes_brackets(self, mock_post):
        """Alpaca crypto orders must NOT contain bracket/stop_loss in payload (HTTP 422 defense)."""
        from bot.brokers.alpaca import AlpacaBroker
        from bot.models import Order, OrderType, OrderStatus

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "order-12345"}
        mock_post.return_value = mock_resp

        broker = AlpacaBroker(api_key="mock_key", secret_key="mock_sec")

        crypto_order = Order(
            order_id="test1",
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.012345,
            stop_loss_price=60000.0,
            take_profit_price=70000.0,
            status=OrderStatus.PENDING,
        )

        broker.submit_order(crypto_order)
        assert mock_post.called
        sent_payload = mock_post.call_args[1]["json"]
        assert "stop_loss" not in sent_payload
        assert "take_profit" not in sent_payload
        assert sent_payload["time_in_force"] == "gtc"
        assert sent_payload["qty"] == "0.0123"

    @patch("bot.brokers.alpaca.requests.post")
    def test_alpaca_equity_order_preserves_brackets(self, mock_post):
        """Equities should include bracket payload when stops and targets are defined."""
        from bot.brokers.alpaca import AlpacaBroker
        from bot.models import Order, OrderType, OrderStatus

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "equity-order-1"}
        mock_post.return_value = mock_resp

        broker = AlpacaBroker(api_key="mock_key", secret_key="mock_sec")

        stock_order = Order(
            order_id="test2",
            symbol="NVDA",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.5,
            stop_loss_price=135.0,
            take_profit_price=150.0,
            status=OrderStatus.PENDING,
        )

        broker.submit_order(stock_order)
        sent_payload = mock_post.call_args[1]["json"]
        assert sent_payload["order_class"] == "bracket"
        assert "stop_loss" in sent_payload
        assert sent_payload["stop_loss"]["stop_price"] == "135.0"
        assert "take_profit" in sent_payload
        assert sent_payload["take_profit"]["limit_price"] == "150.0"
        assert sent_payload["qty"] == "10.0"

    def test_spread_guard_blocks_excessive_spread(self):
        """Spread Guard must reject entries when spread exceeds max_allowed_spread_pct."""
        config = BotRiskConfig(
            enable_spread_guard=True,
            max_allowed_spread_pct=0.45,
        )
        engine = RiskEngine(config=config)
        bal = AccountBalance(cash=50000.0, equity=50000.0, buying_power=50000.0, initial_day_equity=50000.0)

        # 1. Spread = 0.85% > 0.45% -> Rejected by Spread Guard
        wide_spread_signal = TradeSignal(
            symbol="DOGE-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-19",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=0.15,
            stop_loss=0.145,
            metrics={"spread_pct": 0.85, "rsi": 50.0, "ema_ratio": 1.01, "atr": 0.005},
        )
        res = engine.evaluate(wide_spread_signal, bal, positions=[])
        assert res.is_approved is False
        assert any("[SPREAD_GUARD]" in r for r in res.rejection_reasons)

        # 2. Spread = 0.12% <= 0.45% -> Approved
        tight_spread_signal = TradeSignal(
            symbol="DOGE-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-19",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=0.15,
            stop_loss=0.145,
            metrics={"spread_pct": 0.12, "rsi": 50.0, "ema_ratio": 1.01, "atr": 0.005},
        )
        res_ok = engine.evaluate(tight_spread_signal, bal, positions=[])
        assert res_ok.is_approved is True
        assert any("[SPREAD_GUARD]" in w for w in res_ok.risk_warnings)

    def test_max_concurrent_crypto_cap(self):
        """Risk Engine must reject 3rd crypto entry when max_concurrent_crypto_positions is 2."""
        from bot.models import Position
        config = BotRiskConfig(max_concurrent_crypto_positions=2)
        engine = RiskEngine(config=config)
        bal = AccountBalance(cash=50000.0, equity=50000.0, buying_power=50000.0, initial_day_equity=50000.0)

        # Already holding 2 crypto assets
        existing = [
            Position(symbol="BTCUSD", asset_class=AssetClass.CRYPTO, quantity=0.1, entry_price=65000.0, current_price=65000.0),
            Position(symbol="ETHUSD", asset_class=AssetClass.CRYPTO, quantity=2.0, entry_price=2600.0, current_price=2600.0),
        ]

        sol_signal = TradeSignal(
            symbol="SOL-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-19",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.88,
            current_price=150.0,
            stop_loss=145.0,
            metrics={"rsi": 52.0, "ema_ratio": 1.01, "atr": 3.0},
        )
        res = engine.evaluate(sol_signal, bal, positions=existing)
        assert res.is_approved is False
        assert any("Límite de correlación cripto alcanzado" in r for r in res.rejection_reasons)

    def test_position_state_save_and_hydration(self, tmp_path):
        """Active position state (peak prices, TP stages, targets) must survive and hydrate."""
        config = BotRiskConfig(enable_position_state_persistence=True)
        engine = RiskEngine(config=config)

        # Register position
        engine.register_position_entry("NVDA", entry_price=100.0, timestamp=1729000000.0)
        engine.register_position_targets("NVDA", entry_price=100.0, stop_loss=96.0, take_profit=112.0)
        engine.register_partial_tp("NVDA")
        engine.register_entry_context("NVDA", {"rsi": 53.5, "atr": 2.5})
        engine._peak_prices["NVDA"] = 108.50

        # Simulate manual dump and reload
        state = {
            "NVDA": {
                "peak_price": 108.50,
                "tp_stage": 1,
                "partial_tp_executed": True,
                "runner_extended": False,
                "stop_loss": 96.0,
                "take_profit": 112.0,
                "entry_price": 100.0,
                "entry_context": {"rsi": 53.5, "atr": 2.5},
                "entry_time": 1729000000.0,
            }
        }
        import json
        json_str = json.dumps(state)

        # Fresh engine (simulating container restart)
        rebooted_engine = RiskEngine(config=config)
        assert "NVDA" not in rebooted_engine._peak_prices

        # Force hydration with JSON data
        with patch.object(rebooted_engine, "hydrate_active_positions_state", wraps=rebooted_engine.hydrate_active_positions_state):
            with patch("bot.database.neon_repo.get_neon_repo") as mock_neon:
                mock_repo = MagicMock()
                mock_repo.is_connected = True
                mock_repo.get_system_config.return_value = json_str
                mock_neon.return_value = mock_repo
                rebooted_engine._force_test_hydration = True

                count = rebooted_engine.hydrate_active_positions_state(["NVDA"])
                assert count == 1
                assert rebooted_engine._peak_prices["NVDA"] == 108.50
                assert rebooted_engine.get_tp_stage("NVDA") == 1
                assert rebooted_engine._position_targets["NVDA"]["take_profit"] == 112.0
                assert rebooted_engine.get_entry_context("NVDA")["rsi"] == 53.5


