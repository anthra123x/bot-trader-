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

