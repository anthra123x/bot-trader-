"""Unit tests for the trading bot engine, risk management, and paper broker."""

import pytest
from pathlib import Path
from bot.models import (
    AccountBalance,
    AssetClass,
    Order,
    OrderSide,
    OrderStatus,
    Position,
    TradeSignal,
)
from bot.risk.risk_engine import RiskEngine
from bot.config import BotConfig, BotRiskConfig
from bot.brokers.paper import PaperBroker
from bot.learning.pattern_memory import PatternMemory
from bot.runner import BotRunner


class TestRiskEngine:
    def setup_method(self):
        self.config = BotRiskConfig(
            max_risk_per_trade_pct=1.5,
            max_position_size_pct=10.0,
            max_daily_drawdown_pct=3.0,
            default_stop_loss_pct=4.0,
            max_stop_loss_pct=7.0,
        )
        self.engine = RiskEngine(self.config, pattern_memory=PatternMemory(db_path=":memory:"))

    def test_buy_approval_and_position_sizing(self):
        balance = AccountBalance(
            cash=100_000.0,
            equity=100_000.0,
            buying_power=100_000.0,
            initial_day_equity=100_000.0,
        )
        signal = TradeSignal(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=1.0,  # Max conviction (100% Kelly allocation = 50 shares)
            current_price=200.0,
            stop_loss=192.0,  # 4% risk ($8/share)
            target_price=220.0,
        )
        res = self.engine.evaluate(signal, balance, positions=[])
        assert res.is_approved is True
        # Max risk = 1.5% of $100k = $1500. $1500 / $8 = 187.5 -> 187 shares
        # Max alloc = 10% of $100k = $10,000. $10,000 / $200 = 50 shares
        # Min(187, 50) = 50 shares
        assert res.adjusted_quantity == 50
        assert res.allocated_capital == 10_000.0
        assert res.stop_loss_price == 192.0

    def test_kill_switch_triggers_on_drawdown(self):
        balance = AccountBalance(
            cash=96_000.0,
            equity=96_000.0,
            buying_power=96_000.0,
            initial_day_equity=100_000.0,  # 4.0% loss today > 3.0% limit
        )
        signal = TradeSignal(
            symbol="NVDA",
            asset_class=AssetClass.EQUITY,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            current_price=100.0,
            stop_loss=96.0,
        )
        res = self.engine.evaluate(signal, balance, positions=[])
        assert res.is_approved is False
        assert any("Kill-switch active" in r for r in res.rejection_reasons)

    def test_stop_loss_clamped_when_too_wide(self):
        balance = AccountBalance(
            cash=50_000.0,
            equity=50_000.0,
            buying_power=50_000.0,
            initial_day_equity=50_000.0,
        )
        signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            current_price=60_000.0,
            stop_loss=50_000.0,  # 16.6% risk > max 7%
        )
        res = self.engine.evaluate(signal, balance, positions=[])
        assert res.is_approved is True
        # Max 7% stop = 60,000 * 0.93 = 55,800
        assert res.stop_loss_price == 55800.0
        assert any("Clamped to max allowable" in w for w in res.risk_warnings)

    def test_sell_rejected_without_open_position(self):
        balance = AccountBalance(
            cash=100_000.0,
            equity=100_000.0,
            buying_power=100_000.0,
            initial_day_equity=100_000.0,
        )
        signal = TradeSignal(
            symbol="TSLA",
            asset_class=AssetClass.EQUITY,
            trade_date="2026-10-18",
            rating="Sell",
            action=OrderSide.SELL,
            current_price=250.0,
        )
        res = self.engine.evaluate(signal, balance, positions=[])
        assert res.is_approved is False
        assert any("no open position held" in r for r in res.rejection_reasons)

    def test_market_research_prudence_veto(self):
        balance = AccountBalance(
            cash=100_000.0,
            equity=100_000.0,
            buying_power=100_000.0,
            initial_day_equity=100_000.0,
        )
        signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            current_price=60_000.0,
            stop_loss=58_000.0,
            metrics={
                "market_prudence": "IMPRUDENT_ADVERSE_VETO",
                "prudence_directive": "Flujo de mercado bajista con alta presión vendedora.",
            },
        )
        res = self.engine.evaluate(signal, balance, positions=[])
        assert res.is_approved is False
        assert any("VETO DE INVESTIGACIÓN DE MERCADO" in r for r in res.rejection_reasons)

    def test_market_research_caution_sizing(self):
        balance = AccountBalance(
            cash=100_000.0,
            equity=100_000.0,
            buying_power=100_000.0,
            initial_day_equity=100_000.0,
        )
        # Normal signal without caution
        signal_normal = TradeSignal(
            symbol="ETH-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=2000.0,
            stop_loss=1940.0,
            metrics={"rsi": 52.0, "ema_ratio": 1.02, "atr": 40.0},
        )
        res_normal = self.engine.evaluate(signal_normal, balance, positions=[])
        assert res_normal.is_approved is True

        # Signal with caution (50% sizing)
        signal_caution = TradeSignal(
            symbol="ETH-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=2000.0,
            stop_loss=1940.0,
            metrics={
                "rsi": 52.0,
                "ema_ratio": 1.02,
                "atr": 40.0,
                "market_prudence": "CAUTION_HEADWINDS",
                "prudence_multiplier": 0.5,
                "prudence_directive": "Turbulencia elevada en la red.",
            },
        )
        res_caution = self.engine.evaluate(signal_caution, balance, positions=[])
        assert res_caution.is_approved is True
        assert res_caution.adjusted_quantity == round(res_normal.adjusted_quantity * 0.5, 6)
        assert any("ADVERTENCIA DE MERCADO" in w for w in res_caution.risk_warnings)


class TestPaperBroker:
    def test_paper_broker_ledger_flow(self, tmp_path: Path, monkeypatch):
        db_file = tmp_path / "test_paper.db"
        broker = PaperBroker(db_path=db_file, initial_cash=10_000.0)

        # Mock current price to avoid external network call
        monkeypatch.setattr(broker, "get_current_price", lambda sym: 100.0)

        init_bal = broker.get_account_balance()
        assert init_bal.cash == 10_000.0
        assert init_bal.equity == 10_000.0

        # Execute Buy
        order_buy = Order(
            order_id="ord-1",
            symbol="MSFT",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.BUY,
            quantity=20.0,
            limit_price=100.0,
        )
        res_buy = broker.submit_order(order_buy)
        assert res_buy.status == OrderStatus.FILLED
        assert res_buy.filled_price == 100.0

        bal_after_buy = broker.get_account_balance()
        assert bal_after_buy.cash == 8_000.0
        assert bal_after_buy.equity == 10_000.0

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "MSFT"
        assert positions[0].quantity == 20.0
        assert positions[0].market_value == 2000.0

        # Execute Sell
        order_sell = Order(
            order_id="ord-2",
            symbol="MSFT",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.SELL,
            quantity=20.0,
            limit_price=110.0,  # Sell at profit
        )
        res_sell = broker.submit_order(order_sell)
        assert res_sell.status == OrderStatus.FILLED

        bal_after_sell = broker.get_account_balance()
        # Initial 10k - 2k + (20 * 110 = 2200) = 10,200
        assert bal_after_sell.cash == 10_200.0
        assert len(broker.get_positions()) == 0


class TestAlpacaBroker:
    def test_alpaca_broker_cli_fallback(self, monkeypatch):
        from bot.brokers.alpaca import AlpacaBroker

        monkeypatch.setenv("ALPACA_API_KEY", "")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "")
        monkeypatch.setenv("ALPACA_ACCESS_TOKEN", "")
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        monkeypatch.setattr("bot.database.neon_repo.get_neon_repo", lambda: None)

        broker = AlpacaBroker(api_key="", secret_key="")
        assert broker._has_direct_keys() is False

        # Mock _run_cli
        def mock_run_cli(args):
            if args == ["account", "get"]:
                return {"cash": "25000.0", "equity": "30000.0", "buying_power": "50000.0", "last_equity": "29000.0"}
            if args == ["position", "list"]:
                return [{"symbol": "NVDA", "qty": "10", "avg_entry_price": "120.0", "current_price": "130.0"}]
            if args[:2] == ["order", "submit"]:
                return {"id": "alpaca-ord-123"}
            return {}

        monkeypatch.setattr(broker, "_run_cli", mock_run_cli)

        bal = broker.get_account_balance()
        assert bal.cash == 25000.0
        assert bal.equity == 30000.0

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "NVDA"
        assert positions[0].quantity == 10.0

        order = Order(
            order_id="ord-cli-1",
            symbol="NVDA",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.BUY,
            quantity=5.0,
        )
        res_order = broker.submit_order(order)
        assert res_order.order_id == "alpaca-ord-123"
        assert res_order.status == OrderStatus.PENDING


class TestLeanAndFreqtradeProtections:
    def test_trailing_stop_positive_offset_and_ratchet(self):
        config = BotRiskConfig(
            enable_trailing_stop=True,
            trailing_stop_activation_pct=1.5,
            trailing_stop_distance_pct=1.0,
        )
        engine = RiskEngine(config)

        # 1. Entry price = 100.0, Initial Stop Loss = 96.0 (-4%)
        # Price moves to 100.5 (+0.5%, below 0.8% Break-Even and 1.5% activation)
        sl, updated = engine.calculate_trailing_stop("AAPL", current_price=100.5, entry_price=100.0, current_stop_loss=96.0)
        assert updated is False
        assert sl == 96.0

        # Price moves to 101.0 (+1.0%, triggers Stage 1 Dynamic Break-Even at +0.8%)
        sl, updated = engine.calculate_trailing_stop("AAPL", current_price=101.0, entry_price=100.0, current_stop_loss=96.0)
        assert updated is True
        assert sl == 100.15  # 100.0 * 1.0015 (Break-Even locking fees)

        # 2. Price climbs to 103.0 (+3.0%, above 1.5% activation threshold)
        # Peak = 103.0, Trailing Stop = 103.0 * (1 - 0.01) = 101.97
        sl, updated = engine.calculate_trailing_stop("AAPL", current_price=103.0, entry_price=100.0, current_stop_loss=96.0)
        assert updated is True
        assert sl == 101.97

        # 3. Price climbs further to 105.0 (+5.0%)
        # Peak = 105.0, Trailing Stop = 105.0 * (1 - 0.01) = 103.95
        sl, updated = engine.calculate_trailing_stop("AAPL", current_price=105.0, entry_price=100.0, current_stop_loss=101.97)
        assert updated is True
        assert sl == 103.95

        # 4. Price pulls back to 104.2 (Stop must NEVER ratchet downward)
        sl, updated = engine.calculate_trailing_stop("AAPL", current_price=104.2, entry_price=100.0, current_stop_loss=103.95)
        assert updated is False
        assert sl == 103.95

    def test_freqtrade_cooldown_period(self):
        config = BotRiskConfig(cooldown_period_seconds=1800)
        engine = RiskEngine(config)

        # Simulate stop-loss exit on TSLA
        engine.register_trade_exit("TSLA", is_loss=True)

        balance = AccountBalance(cash=50_000.0, equity=50_000.0, buying_power=50_000.0, initial_day_equity=50_000.0)
        signal = TradeSignal(
            symbol="TSLA",
            asset_class=AssetClass.EQUITY,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            current_price=350.0,
            confidence=0.85,
        )

        res = engine.evaluate(signal, balance, positions=[])
        assert res.is_approved is False
        assert any("CooldownPeriod activo" in r for r in res.rejection_reasons)

    def test_freqtrade_stoploss_guard(self):
        config = BotRiskConfig(max_consecutive_stoplosses=3)
        engine = RiskEngine(config)

        # 3 consecutive stoploss events
        for sym in ["NVDA", "AAPL", "MSFT"]:
            engine.register_trade_exit(sym, is_loss=True)

        balance = AccountBalance(cash=50_000.0, equity=50_000.0, buying_power=50_000.0, initial_day_equity=50_000.0)
        signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            current_price=80000.0,
            confidence=0.90,
        )

        res = engine.evaluate(signal, balance, positions=[])
        assert res.is_approved is False
        assert any("StoplossGuard activo" in r for r in res.rejection_reasons)

    def test_tiered_partial_take_profit(self):
        config = BotRiskConfig(
            enable_partial_take_profit=True,
            partial_take_profit_threshold_pct=1.0,
            partial_take_profit_ratio=0.4,
        )
        engine = RiskEngine(config)

        # 1. Price at +0.5% (below threshold) -> No partial TP
        should_tp, qty = engine.evaluate_partial_take_profit("ETH-USD", current_price=100.5, entry_price=100.0, current_qty=10.0)
        assert should_tp is False
        assert qty == 0.0

        # 2. Price at +1.2% (above threshold) -> Triggers 40% scale-out (4.0 u)
        should_tp, qty = engine.evaluate_partial_take_profit("ETH-USD", current_price=101.2, entry_price=100.0, current_qty=10.0)
        assert should_tp is True
        assert qty == 4.0

        # 3. Once registered, subsequent ticks should not re-trigger partial TP
        engine.register_partial_tp("ETH-USD")
        should_tp, qty = engine.evaluate_partial_take_profit("ETH-USD", current_price=101.5, entry_price=100.0, current_qty=6.0)
        assert should_tp is False
        assert qty == 0.0

        # 4. After full trade exit, tracker resets
        engine.register_trade_exit("ETH-USD", is_loss=False)
        should_tp, qty = engine.evaluate_partial_take_profit("ETH-USD", current_price=101.2, entry_price=100.0, current_qty=10.0)
        assert should_tp is True
        assert qty == 4.0

    def test_portfolio_dynamic_rebalancing(self):
        config = BotRiskConfig(
            enable_dynamic_rebalancing=True,
            max_concentration_threshold_pct=35.0,
            target_concentration_pct=25.0,
        )
        engine = RiskEngine(config)

        # 1. Overweight position in profit (52.5% concentration, +5.0% profit)
        pos = Position(
            symbol="ETH-USD",
            asset_class=AssetClass.CRYPTO,
            quantity=25.0,
            entry_price=2000.0,
            current_price=2100.0,
        )
        should_rebal, trim_qty, reason = engine.evaluate_portfolio_rebalancing(pos, current_price=2100.0, total_equity=100_000.0)
        assert should_rebal is True
        assert trim_qty > 0
        assert "Concentración excesiva" in reason

        # 2. Balanced position (< 35% concentration) -> No rebalance
        pos_balanced = Position(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            quantity=0.2,
            entry_price=80000.0,
            current_price=82000.0,
        )
        should_rebal, trim_qty, reason = engine.evaluate_portfolio_rebalancing(pos_balanced, current_price=82000.0, total_equity=100_000.0)
        assert should_rebal is False
        assert trim_qty == 0.0

        # 3. Overweight position at a LOSS -> Do not rebalance at loss
        pos_loss = Position(
            symbol="SOL-USD",
            asset_class=AssetClass.CRYPTO,
            quantity=400.0,
            entry_price=120.0,
            current_price=100.0,
        )
        should_rebal, trim_qty, reason = engine.evaluate_portfolio_rebalancing(pos_loss, current_price=100.0, total_equity=100_000.0)
        assert should_rebal is False
        assert trim_qty == 0.0

    def test_multi_stage_take_profit_flow(self):
        config = BotRiskConfig(
            enable_partial_take_profit=True,
            partial_take_profit_threshold_pct=1.2,
            partial_take_profit_ratio=0.35,
            enable_multi_stage_tp=True,
            stage2_take_profit_threshold_pct=3.0,
            stage2_take_profit_ratio=0.45,
        )
        engine = RiskEngine(config)

        # 1. At +0.8%, below stage 1 -> False
        should_tp, qty = engine.evaluate_partial_take_profit("SOL-USD", current_price=100.8, entry_price=100.0, current_qty=100.0)
        assert should_tp is False

        # 2. At +1.2%, stage 1 triggered -> 35% sold
        should_tp, qty = engine.evaluate_partial_take_profit("SOL-USD", current_price=101.2, entry_price=100.0, current_qty=100.0)
        assert should_tp is True
        assert qty == 35.0
        engine.register_partial_tp("SOL-USD")
        assert engine.get_tp_stage("SOL-USD") == 1

        # 3. At +1.5%, between stage 1 and stage 2 -> False
        should_tp, qty = engine.evaluate_partial_take_profit("SOL-USD", current_price=101.5, entry_price=100.0, current_qty=65.0)
        assert should_tp is False

        # 4. At +3.2%, stage 2 triggered -> 45% of remaining 65.0 sold (29.25 u)
        should_tp, qty = engine.evaluate_partial_take_profit("SOL-USD", current_price=103.2, entry_price=100.0, current_qty=65.0)
        assert should_tp is True
        assert qty == 29.25
        engine.register_partial_tp("SOL-USD")
        assert engine.get_tp_stage("SOL-USD") == 2

        # 5. Subsequent ticks -> False (completed both stages)
        should_tp, qty = engine.evaluate_partial_take_profit("SOL-USD", current_price=104.0, entry_price=100.0, current_qty=35.75)
        assert should_tp is False

    def test_quality_gate_and_crypto_correlation(self):
        config = BotRiskConfig(
            min_signal_confidence=0.70,
            max_concurrent_crypto_positions=2,
        )
        engine = RiskEngine(config)
        bal = AccountBalance(cash=100_000.0, equity=100_000.0, buying_power=100_000.0)

        # Test Quality Gate rejection when confidence < 0.70
        low_conf_signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.55,
            current_price=80_000.0,
            stop_loss=78_000.0,
        )
        res = engine.evaluate(low_conf_signal, bal, positions=[])
        assert res.is_approved is False
        assert any("Quality Gate" in r for r in res.rejection_reasons)

        # Test Crypto Correlation Gate when 2 crypto positions are already open
        high_conf_signal = TradeSignal(
            symbol="DOGE-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=0.10,
            stop_loss=0.096,
        )
        open_crypto = [
            Position(symbol="BTC-USD", asset_class=AssetClass.CRYPTO, quantity=1.0, entry_price=80_000.0, current_price=80_000.0),
            Position(symbol="ETH-USD", asset_class=AssetClass.CRYPTO, quantity=10.0, entry_price=2400.0, current_price=2400.0),
        ]
        res_corr = engine.evaluate(high_conf_signal, bal, positions=open_crypto)
        assert res_corr.is_approved is False
        assert any("Límite de correlación cripto alcanzado" in r for r in res_corr.rejection_reasons)

    def test_quant_screener_divergence_algorithm(self):
        from bot.risk.quant_screener import QuantScreener

        # Synthetic prices: Lower Low in price with Higher Low in RSI (Bullish Divergence)
        closes = [100.0] * 10 + [90.0, 92.0, 91.0, 93.0, 88.0]  # past low 90, current 88 (Lower Low)
        rsis = [50.0] * 10 + [25.0, 32.0, 30.0, 35.0, 38.0]     # past rsi 25, current 38 (Higher Low)
        div_type, desc = QuantScreener._detect_divergence(closes, rsis, lookback=5)
        assert div_type == "BULLISH"
        assert "Divergencia Alcista RSI" in desc

    def test_four_master_levers(self):
        """Validates Palancas A, B, C, D:
        A: Defensive Regime Veto
        B: Macro SMA-200 Filter
        C: Anti-Bull Trap Ceiling (RSI >= 64.0)
        D: Min Risk/Reward >= 2.5:1
        """
        config = BotRiskConfig(
            min_risk_reward_ratio=2.5,
            max_overbought_rsi_buy=64.0,
            enable_macro_sma200_filter=True,
            enable_defensive_regime_veto=True,
        )
        engine = RiskEngine(config)
        bal = AccountBalance(cash=100_000.0, equity=100_000.0, buying_power=100_000.0)

        # 1. Palanca A: Defensive Regime Veto
        defensive_signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            current_price=80_000.0,
            stop_loss=78_000.0,
            take_profit=86_000.0,
            metrics={"regime": "DEFENSIVE_BEAR_BREAKDOWN", "allocation_factor": 0.0, "rsi": 45.0, "sma200": 70_000.0},
        )
        res_a = engine.evaluate(defensive_signal, bal, positions=[])
        assert res_a.is_approved is False
        assert any("Veto por Régimen Defensivo" in r for r in res_a.rejection_reasons)

        # 2. Palanca C: Anti-Bull Trap Ceiling (RSI >= 64.0)
        overbought_signal = TradeSignal(
            symbol="SOL-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            current_price=100.0,
            stop_loss=96.0,
            take_profit=112.0,
            metrics={"regime": "BULL_TREND", "allocation_factor": 1.0, "rsi": 68.5, "sma200": 90.0},
        )
        res_c = engine.evaluate(overbought_signal, bal, positions=[])
        assert res_c.is_approved is False
        assert any("Veto Anti-Trampa" in r for r in res_c.rejection_reasons)

        # 3. Palanca B: Macro SMA-200 Filter (Below 200 SMA without bullish divergence)
        below_sma200_signal = TradeSignal(
            symbol="ETH-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            current_price=2400.0,
            stop_loss=2300.0,
            take_profit=2700.0,
            metrics={"regime": "CHOPPY_RANGE_BREAKDOWN", "allocation_factor": 0.6, "rsi": 42.0, "sma200": 2600.0, "divergence": "NONE"},
        )
        res_b = engine.evaluate(below_sma200_signal, bal, positions=[])
        assert res_b.is_approved is False
        assert any("Filtro Macro SMA-200" in r for r in res_b.rejection_reasons)

        # 4. Palanca D: Min Risk/Reward Ratio >= 2.5:1
        # Risk: $1.00 (40.0 - 39.0), Initial Reward: $1.50 -> R:R = 1.5:1 (fails >= 2.5:1)
        suboptimal_rr_signal = TradeSignal(
            symbol="AVAX-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            current_price=40.0,
            stop_loss=39.0,
            target_price=41.5,
            metrics={"regime": "BULL_TREND", "allocation_factor": 1.0, "rsi": 54.0, "sma200": 30.0, "ema_ratio": 1.06, "atr": 0.8},
        )
        res_d = engine.evaluate(suboptimal_rr_signal, bal, positions=[])
        assert res_d.is_approved is True
        # Verify take profit is automatically elevated to guarantee strictly >= 2.5:1
        expected_reward = (40.0 - 39.0) * 2.5
        assert res_d.take_profit_price >= round(40.0 + expected_reward, 2)
        assert any("elevado asimétricamente a 2.5:1" in w for w in res_d.risk_warnings)

        # 5. Golden Trade: Meets all 4 Master Levers
        # Risk: $1.50 (40.0 - 38.5), Reward: $5.00 (45.0 - 40.0) -> R:R = 3.33:1, RSI = 54.0, Above SMA200, alloc_factor = 1.0
        good_signal = TradeSignal(
            symbol="AVAX-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            current_price=40.0,
            stop_loss=38.5,
            take_profit=45.0,
            metrics={"regime": "BULL_TREND", "allocation_factor": 1.0, "rsi": 54.0, "sma200": 30.0, "ema_ratio": 1.06, "atr": 0.8},
        )
        res_good = engine.evaluate(good_signal, bal, positions=[])
        assert res_good.is_approved is True

    def test_persistent_position_targets_and_cap_protection(self):
        """Verifies:
        1. Automatic registration and retrieval of persistent target TP & SL.
        2. Monotonic ratcheting of stop loss via update_position_stop.
        3. Graceful rejection when attempting to micro-add to an already full position.
        """
        config = BotRiskConfig(
            max_position_size_pct=12.0,
            default_stop_loss_pct=2.5,
            min_risk_reward_ratio=2.5,
        )
        engine = RiskEngine(config)
        bal = AccountBalance(cash=100_000.0, equity=100_000.0, buying_power=100_000.0)

        # 1. New trade registration
        targets = engine.get_position_targets("SOL-USD", entry_price=100.0)
        assert targets["stop_loss"] == 97.5  # 100 - 2.5%
        assert targets["take_profit"] == 106.25  # 100 + (2.5 * 2.5)% = +6.25%

        # 2. Monotonic stop ratchet
        engine.update_position_stop("SOL-USD", 100.15)  # Break even
        assert engine.get_position_targets("SOL-USD", entry_price=100.0)["stop_loss"] == 100.15
        engine.update_position_stop("SOL-USD", 99.0)   # Lower stop should be ignored
        assert engine.get_position_targets("SOL-USD", entry_price=100.0)["stop_loss"] == 100.15

        # 3. Clean rejection when position allocation cap is already filled
        # Max alloc = $100k * 12% * alloc_scale (~$9,120)
        existing_pos = Position(
            symbol="SOL-USD",
            asset_class=AssetClass.CRYPTO,
            quantity=90.0,
            entry_price=100.0,
            current_price=101.0,  # market value $9,090 (within $50 of cap)
        )
        add_signal = TradeSignal(
            symbol="SOL-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.8,
            current_price=101.0,
            stop_loss=98.5,
            take_profit=108.0,
            metrics={"regime": "BULL_TREND", "allocation_factor": 1.0, "rsi": 54.0, "sma200": 80.0, "ema_ratio": 1.06, "atr": 1.2},
        )
        res_cap = engine.evaluate(add_signal, bal, positions=[existing_pos])
        assert res_cap.is_approved is False
        assert any("ya completada" in r or "allocation limit reached" in r for r in res_cap.rejection_reasons)

        # 4. Cleanup on trade exit
        engine.register_trade_exit("SOL-USD", is_loss=False)
        assert "SOLUSD" not in engine._position_targets

    def test_minimum_cash_reserve_floor(self):
        """Ensure RiskEngine never spends cash below the institutional minimum reserve floor."""
        config = BotRiskConfig(min_cash_reserve_pct=25.0, max_position_size_pct=10.0)
        engine = RiskEngine(config, pattern_memory=PatternMemory(db_path=":memory:"))

        # Equity = $100k, min cash reserve = $25k. Cash = $24k -> strictly rejected
        bal_low_cash = AccountBalance(cash=24_000.0, equity=100_000.0, buying_power=24_000.0, initial_day_equity=100_000.0)
        signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=60_000.0,
            stop_loss=58_000.0,
            metrics={"rsi": 50.0, "ema_ratio": 1.01, "atr": 1000.0},
        )
        res_reject = engine.evaluate(signal, bal_low_cash, positions=[])
        assert res_reject.is_approved is False
        assert any("Piso de Liquidez / Reserva de Efectivo activa" in r for r in res_reject.rejection_reasons)

        # Cash = $30k -> Available cash above floor is $5k. Sizing is capped to $5k, preserving the $25k floor!
        bal_ok_cash = AccountBalance(cash=30_000.0, equity=100_000.0, buying_power=30_000.0, initial_day_equity=100_000.0)
        res_ok = engine.evaluate(signal, bal_ok_cash, positions=[])
        assert res_ok.is_approved is True
        assert res_ok.allocated_capital <= 5_000.0

    def test_aggregate_crypto_exposure_cap(self):
        """Ensure total crypto holdings cannot exceed max_total_crypto_allocation_pct."""
        config = BotRiskConfig(max_total_crypto_allocation_pct=30.0, max_position_size_pct=12.0)
        engine = RiskEngine(config, pattern_memory=PatternMemory(db_path=":memory:"))
        bal = AccountBalance(cash=80_000.0, equity=100_000.0, buying_power=80_000.0, initial_day_equity=100_000.0)

        # Existing crypto positions: BTC ($15k) + ETH ($15k) = $30k (already at 30% cap)
        existing_crypto = [
            Position(symbol="BTCUSD", asset_class=AssetClass.CRYPTO, quantity=0.25, entry_price=60_000.0, current_price=60_000.0),
            Position(symbol="ETHUSD", asset_class=AssetClass.CRYPTO, quantity=6.0, entry_price=2500.0, current_price=2500.0),
        ]
        sol_signal = TradeSignal(
            symbol="SOL-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=100.0,
            stop_loss=97.0,
            metrics={"rsi": 50.0, "ema_ratio": 1.01, "atr": 2.0},
        )
        res = engine.evaluate(sol_signal, bal, positions=existing_crypto)
        assert res.is_approved is False
        assert any("Techo de Exposición Agregada Cripto alcanzado" in r for r in res.rejection_reasons)

    def test_portfolio_trailing_drawdown_guard(self):
        """Ensure gains from an intraday peak equity high-water mark are locked against deep drawdowns."""
        config = BotRiskConfig(enable_portfolio_trailing_drawdown_guard=True, max_trailing_drawdown_from_peak_pct=2.5)
        engine = RiskEngine(config, pattern_memory=PatternMemory(db_path=":memory:"))

        # 1. Account sets peak at $105,000
        bal_peak = AccountBalance(cash=50_000.0, equity=105_000.0, buying_power=50_000.0, initial_day_equity=100_000.0)
        signal = TradeSignal(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=200.0,
            stop_loss=194.0,
        )
        res_peak = engine.evaluate(signal, bal_peak, positions=[])
        assert res_peak.is_approved is True
        assert engine._peak_equity == 105_000.0

        # 2. Equity drops from $105,000 to $102,000 (drawdown of 2.85% from peak > 2.5% max)
        bal_dd = AccountBalance(cash=50_000.0, equity=102_000.0, buying_power=50_000.0, initial_day_equity=100_000.0)
        res_dd = engine.evaluate(signal, bal_dd, positions=[])
        assert res_dd.is_approved is False
        assert any("Portfolio Peak Drawdown Guard activo" in r for r in res_dd.rejection_reasons)

    def test_in_flight_orders_prevent_duplicate_pyramiding(self):
        """Ensure back-to-back rapid signals do not exceed position allocation limit while orders are in-flight."""
        config = BotRiskConfig(max_position_size_pct=10.0)
        engine = RiskEngine(config, pattern_memory=PatternMemory(db_path=":memory:"))
        bal = AccountBalance(cash=100_000.0, equity=100_000.0, buying_power=100_000.0, initial_day_equity=100_000.0)

        # Register in-flight buy order of $9,800 for ETH-USD
        engine.register_order_in_flight("ETH-USD", quantity=4.083, price=2400.0, asset_class=AssetClass.CRYPTO)
        assert engine.get_in_flight_capital("ETH-USD") > 9000.0

        # Attempt immediate second buy on ETH-USD before broker positions reflect the fill
        signal = TradeSignal(
            symbol="ETH-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=2400.0,
            stop_loss=2340.0,
            metrics={"rsi": 50.0, "ema_ratio": 1.01, "atr": 40.0},
        )
        # Positions is empty because broker fill has not yet indexed
        res_dup = engine.evaluate(signal, bal, positions=[])
        assert res_dup.is_approved is False
        assert any("allocation limit reached" in r or "ya completada" in r for r in res_dup.rejection_reasons)

    def test_profit_vault_ratchet_floors(self):
        """Ensure Multi-Tier Profit Vault locks guaranteed profit floors as peak price advances."""
        config = BotRiskConfig(
            enable_trailing_stop=True,
            enable_profit_vault_ratchet=True,
            trailing_stop_activation_pct=1.2,
            trailing_stop_distance_pct=1.0,
            vault_tier1_activation_pct=1.5,
            vault_tier1_floor_pct=0.4,
            vault_tier2_activation_pct=2.5,
            vault_tier2_floor_pct=1.2,
            vault_tier3_activation_pct=4.0,
            vault_tier3_floor_pct=2.4,
            vault_tier4_activation_pct=6.0,
            vault_tier4_floor_pct=4.2,
            vault_tier5_activation_pct=8.5,
            vault_tier5_floor_pct=6.5,
            vault_tier6_activation_pct=12.0,
            vault_tier6_floor_pct=9.5,
        )
        engine = RiskEngine(config)

        # Entry at $100, initial SL at $97.50
        entry = 100.0
        current_sl = 97.50

        # 1. Price at $101.50 (+1.5% peak) -> Locks Tier 1 floor (+0.4% = $100.40)
        new_sl, updated = engine.calculate_trailing_stop("ETH-USD", current_price=101.50, entry_price=entry, current_stop_loss=current_sl)
        assert updated is True
        assert new_sl >= 100.40
        current_sl = new_sl

        # 2. Price at $102.50 (+2.5% peak) -> Locks Tier 2 floor (+1.2% = $101.20)
        new_sl, updated = engine.calculate_trailing_stop("ETH-USD", current_price=102.50, entry_price=entry, current_stop_loss=current_sl)
        assert updated is True
        assert new_sl >= 101.20
        current_sl = new_sl

        # 3. Price at $104.00 (+4.0% peak) -> Locks Tier 3 floor (+2.4% = $102.40)
        new_sl, updated = engine.calculate_trailing_stop("ETH-USD", current_price=104.00, entry_price=entry, current_stop_loss=current_sl)
        assert updated is True
        assert new_sl >= 102.40
        current_sl = new_sl

        # 4. Price at $106.00 (+6.0% peak) -> Locks Tier 4 floor (+4.2% = $104.20)
        new_sl, updated = engine.calculate_trailing_stop("ETH-USD", current_price=106.00, entry_price=entry, current_stop_loss=current_sl)
        assert updated is True
        assert new_sl >= 104.20
        current_sl = new_sl

        # 5. Price pulls back sharply from $106.00 to $104.50 -> Stop loss is PRESERVED, never regresses!
        new_sl, updated = engine.calculate_trailing_stop("ETH-USD", current_price=104.50, entry_price=entry, current_stop_loss=current_sl)
        assert new_sl >= 104.20  # Still locked at Tier 4 guaranteed floor

        # 6. Price shoots to $108.50 (+8.5% peak) -> Locks Tier 5 floor (+6.5% = $106.50)
        new_sl, updated = engine.calculate_trailing_stop("ETH-USD", current_price=108.50, entry_price=entry, current_stop_loss=current_sl)
        assert updated is True
        assert new_sl >= 106.50
        current_sl = new_sl

        # 7. Price shoots to $112.00 (+12.0% peak) -> Locks Tier 6 floor (+9.5% = $109.50)
        new_sl, updated = engine.calculate_trailing_stop("ETH-USD", current_price=112.00, entry_price=entry, current_stop_loss=current_sl)
        assert updated is True
        assert new_sl >= 109.50

    def test_three_stage_partial_take_profit(self):
        """Ensure partial take profit executes smoothly across Stage 1, Stage 2, and Stage 3."""
        config = BotRiskConfig(
            enable_partial_take_profit=True,
            partial_take_profit_threshold_pct=1.2,
            partial_take_profit_ratio=0.35,
            enable_multi_stage_tp=True,
            stage2_take_profit_threshold_pct=3.0,
            stage2_take_profit_ratio=0.45,
            enable_stage3_tp=True,
            stage3_take_profit_threshold_pct=5.5,
            stage3_take_profit_ratio=0.50,
        )
        engine = RiskEngine(config)

        # Initial position: 100 units at $100.00
        qty = 100.0

        # Stage 1: +1.2% -> Harvest 35% (35 units)
        should_tp, harvest_qty = engine.evaluate_partial_take_profit("BTC-USD", current_price=101.20, entry_price=100.0, current_qty=qty)
        assert should_tp is True
        assert harvest_qty == 35.0
        engine.register_partial_tp("BTC-USD")
        qty -= harvest_qty  # 65.0 units remaining
        assert engine.get_tp_stage("BTC-USD") == 1

        # Stage 2: +3.0% -> Harvest 45% of 65.0 = 29.25 units
        should_tp, harvest_qty = engine.evaluate_partial_take_profit("BTC-USD", current_price=103.00, entry_price=100.0, current_qty=qty)
        assert should_tp is True
        assert harvest_qty == 29.25
        engine.register_partial_tp("BTC-USD")
        qty -= harvest_qty  # 35.75 units remaining
        assert engine.get_tp_stage("BTC-USD") == 2

        # In between Stage 2 and Stage 3: +4.0% -> False
        should_tp, _ = engine.evaluate_partial_take_profit("BTC-USD", current_price=104.00, entry_price=100.0, current_qty=qty)
        assert should_tp is False

        # Stage 3: +5.5% -> Harvest 50% of 35.75 = 17.875 units
        should_tp, harvest_qty = engine.evaluate_partial_take_profit("BTC-USD", current_price=105.50, entry_price=100.0, current_qty=qty)
        assert should_tp is True
        assert harvest_qty == 17.875
        engine.register_partial_tp("BTC-USD")
        qty -= harvest_qty
        assert engine.get_tp_stage("BTC-USD") == 3

        # After stage 3: +8.0% -> False (all 3 stages completed)
        should_tp, _ = engine.evaluate_partial_take_profit("BTC-USD", current_price=108.00, entry_price=100.0, current_qty=qty)
        assert should_tp is False

    def test_free_roll_winner_pyramiding(self):
        """Ensure Free-Roll Winner Pyramiding allows scaling into winners with locked-in profit."""
        config = BotRiskConfig(
            max_position_size_pct=10.0,
            enable_winner_pyramiding=True,
            min_profit_pct_for_pyramiding=2.8,
            pyramiding_max_allocation_boost=1.35,
        )
        engine = RiskEngine(config, pattern_memory=PatternMemory(db_path=":memory:"))
        bal = AccountBalance(cash=100_000.0, equity=100_000.0, buying_power=100_000.0, initial_day_equity=100_000.0)

        # 1. Existing position on SOL-USD: Entry $100, Stop Loss ratcheted to $101.50 (guaranteed profit), Current Price $103.50 (+3.5%)
        # Current holding is $9,500 (near 10% base cap)
        existing_pos = Position(
            symbol="SOL-USD",
            asset_class=AssetClass.CRYPTO,
            quantity=95.0,
            entry_price=100.0,
            current_price=103.50,
            stop_loss_price=101.50,
        )

        signal = TradeSignal(
            symbol="SOL-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=103.50,
            stop_loss=101.50,
            metrics={"rsi": 55.0, "ema_ratio": 1.02, "atr": 2.0},
        )

        # Should be APPROVED because it's a free-roll winner with locked-in profit!
        res = engine.evaluate(signal, bal, positions=[existing_pos])
        assert res.is_approved is True
        assert any("PYRAMIDING_ACTIVE" in w for w in res.risk_warnings)
        assert res.adjusted_quantity > 0

        # 2. Compare against a LOSING existing position: Entry $100, Current Price $97.00, Stop Loss $95.00
        existing_loss_pos = Position(
            symbol="AVAX-USD",
            asset_class=AssetClass.CRYPTO,
            quantity=95.0,
            entry_price=100.0,
            current_price=97.0,
            stop_loss_price=95.0,
        )
        loss_signal = TradeSignal(
            symbol="AVAX-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=97.0,
            stop_loss=95.0,
            metrics={"rsi": 45.0, "ema_ratio": 0.98, "atr": 2.0},
        )
        # Should be REJECTED - never pyramid into losers or unhedged positions!
        res_loss = engine.evaluate(loss_signal, bal, positions=[existing_loss_pos])
        assert res_loss.is_approved is False

    def test_runner_target_extension_in_runner(self):
        """Ensure Runner Target Extension harvests 60% and extends 40% runner to higher target."""
        config = BotConfig(broker_type="paper")
        config.risk.enable_runner_target_extension = True
        config.risk.runner_extension_harvest_ratio = 0.60
        config.risk.runner_tp_multiplier = 1.08

        config.risk.enable_dynamic_rebalancing = False
        runner = BotRunner(config=config, broker=PaperBroker())
        # Simulate that earlier partial TP stages were already harvested on the way up
        runner.risk_engine.register_partial_tp("NVDA")
        runner.risk_engine.register_partial_tp("NVDA")
        runner.risk_engine.register_partial_tp("NVDA")
        runner.risk_engine.register_position_targets("NVDA", entry_price=100.0, stop_loss=97.5, take_profit=106.0)

        # Create open position at target TP: 100 shares of NVDA, entry $100, price $106.50 >= TP ($106.00)
        pos = Position(
            symbol="NVDA",
            asset_class=AssetClass.EQUITY,
            quantity=100.0,
            entry_price=100.0,
            current_price=106.50,
            take_profit_price=106.0,
            stop_loss_price=100.0,
        )
        runner.broker.get_positions = lambda: [pos]
        runner.broker.get_current_price = lambda sym: 106.50
        runner.broker.get_account_balance = lambda: AccountBalance(cash=100_000.0, equity=100_000.0, buying_power=100_000.0)
        runner.broker.submit_order = lambda order: order

        closed_orders = runner.manage_open_positions()
        # 1. An order was executed for 60 shares (60% harvest)
        assert len(closed_orders) == 1
        assert closed_orders[0].quantity == 60.0
        assert closed_orders[0].side == OrderSide.SELL

        # 2. Position remains open with remaining 40 shares (runner)
        assert pos.quantity == 40.0

        # 3. Target TP was extended (+8% above 106.50 = 115.02)
        assert pos.take_profit_price > 110.0

        # 4. Stop loss was ratcheted tightly behind current price (within 1.2%)
        assert pos.stop_loss_price >= 105.0

        # 5. Runner extension was registered in RiskEngine
        assert runner.risk_engine.is_runner_extended("NVDA") is True

    def test_profit_quarantine_vault_50_pct(self):
        """Ensure 50% of cumulative net profit is quarantined into an untouchable cash vault floor."""
        config = BotRiskConfig(
            min_cash_reserve_pct=20.0,
            enable_profit_quarantine_vault=True,
            profit_quarantine_ratio=0.50,
        )
        engine = RiskEngine(config, pattern_memory=PatternMemory(db_path=":memory:"))

        # Starting equity was $100,000. Current equity is $106,000 (+ $6,000 net profit).
        # Base cash reserve = 20% of $106,000 = $21,200.
        # Quarantined profit = 50% of $6,000 = $3,000.
        # Total protected cash floor = $21,200 + $3,000 = $24,200.
        bal = AccountBalance(cash=25_000.0, equity=106_000.0, buying_power=25_000.0, initial_day_equity=100_000.0)

        total_prot, base_prot, quar = engine.calculate_protected_cash_reserve(bal)
        assert base_prot == 21_200.0
        assert quar == 3_000.0
        assert total_prot == 24_200.0

        # Available cash above floor is only $25,000 - $24,200 = $800.
        signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=60_000.0,
            stop_loss=58_000.0,
            metrics={"rsi": 50.0, "ema_ratio": 1.01, "atr": 1000.0},
        )
        res = engine.evaluate(signal, bal, positions=[])
        assert res.is_approved is True
        assert any("PROFIT_QUARANTINE" in w for w in res.risk_warnings)
        # Allocation must be strictly bounded by available unquarantined cash ($800)
        assert res.allocated_capital <= 800.0

        # If cash is at or below $24,200 (e.g. $24,000) -> BUY MUST BE REJECTED to protect quarantined profit!
        bal_low = AccountBalance(cash=24_000.0, equity=106_000.0, buying_power=24_000.0, initial_day_equity=100_000.0)
        res_reject = engine.evaluate(signal, bal_low, positions=[])
        assert res_reject.is_approved is False
        assert any("bóveda de ganancias 50%" in r for r in res_reject.rejection_reasons)

    def test_profit_quarantine_vault_default_40_pct_and_scaling(self):
        """Ensure default BotRiskConfig applies 40% vault retention (60% active working) and 25% sizing."""
        config = BotRiskConfig()
        assert config.profit_quarantine_ratio == 0.40
        assert config.max_position_size_pct == 25.0
        assert config.partial_take_profit_threshold_pct == 2.2
        assert config.stage2_take_profit_threshold_pct == 4.5
        assert config.stage3_take_profit_threshold_pct == 8.0
        assert config.runner_tp_multiplier == 1.15
        assert config.max_concentration_threshold_pct == 32.0

        engine = RiskEngine(config, pattern_memory=PatternMemory(db_path=":memory:"))

        # Equity $110,000, initial $100,000 -> $10,000 profit.
        # Base reserve = 20% of 110k = $22,000.
        # Quarantined profit = 40% of 10k = $4,000 (leaving $6,000 working).
        # Total protected = $26,000.
        bal = AccountBalance(cash=50_000.0, equity=110_000.0, buying_power=50_000.0, initial_day_equity=100_000.0)
        total_prot, base_prot, quar = engine.calculate_protected_cash_reserve(bal)
        assert base_prot == 22_000.0
        assert quar == 4_000.0
        assert total_prot == 26_000.0

        # Allocation can scale up to 25% of equity ($27,500), constrained by available unencumbered cash ($24,000).
        signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-18",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.90,
            current_price=60_000.0,
            stop_loss=58_500.0,
            metrics={"rsi": 52.0, "ema_ratio": 1.02, "atr": 500.0},
        )
        res = engine.evaluate(signal, bal, positions=[])
        assert res.is_approved is True
        assert any("40% de ganancia neta" in w for w in res.risk_warnings)
        assert res.allocated_capital <= 24_000.0

    def test_screener_conviction_threshold_55_filter(self):
        """Ensure QuantScreener ignores weak marginal signals (<55 score) to protect agent quota."""
        from bot.risk.quant_screener import QuantScreener
        config = BotRiskConfig()
        assert config.screener_conviction_threshold == 55.0

        screener = QuantScreener(conviction_threshold=config.screener_conviction_threshold)
        assert screener.conviction_threshold == 55.0

    def test_profit_quarantine_milestone_ratchet(self):
        """Ensure vault only harvests at configured milestone steps ($200) and NEVER fluctuates down."""
        config = BotRiskConfig(
            min_cash_reserve_pct=20.0,
            enable_profit_quarantine_vault=True,
            profit_quarantine_ratio=0.40,
            profit_quarantine_milestone_step=200.0,
        )
        engine = RiskEngine(config, pattern_memory=PatternMemory(db_path=":memory:"))

        # Initial capital: $100,000.
        # 1. Net profit $120 (< $200 step) -> 0 milestones completed -> $0 quarantined.
        bal_stage1 = AccountBalance(cash=50_000.0, equity=100_120.0, buying_power=50_000.0, initial_day_equity=100_000.0)
        _, _, quar1 = engine.calculate_protected_cash_reserve(bal_stage1)
        assert quar1 == 0.0

        # 2. Net profit advances to $250 -> 1 milestone completed ($200) -> 40% of $200 = $80 quarantined.
        bal_stage2 = AccountBalance(cash=50_000.0, equity=100_250.0, buying_power=50_000.0, initial_day_equity=100_000.0)
        _, _, quar2 = engine.calculate_protected_cash_reserve(bal_stage2)
        assert quar2 == 80.0

        # 3. Market dips temporarily, net profit drops to $140 -> Vault must NOT fluctuate down! Stays at $80.0.
        bal_stage3 = AccountBalance(cash=50_000.0, equity=100_140.0, buying_power=50_000.0, initial_day_equity=100_000.0)
        _, _, quar3 = engine.calculate_protected_cash_reserve(bal_stage3)
        assert quar3 == 80.0

        # 4. Net profit expands to $480 -> 2 milestones completed ($400) -> 40% of $400 = $160 quarantined.
        bal_stage4 = AccountBalance(cash=50_000.0, equity=100_480.0, buying_power=50_000.0, initial_day_equity=100_000.0)
        _, _, quar4 = engine.calculate_protected_cash_reserve(bal_stage4)
        assert quar4 == 160.0

        # 5. Market dips to $310 profit -> Vault remains locked at $160.0!
        bal_stage5 = AccountBalance(cash=50_000.0, equity=100_310.0, buying_power=50_000.0, initial_day_equity=100_000.0)
        _, _, quar5 = engine.calculate_protected_cash_reserve(bal_stage5)
        assert quar5 == 160.0






