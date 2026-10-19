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
from bot.config import BotRiskConfig
from bot.brokers.paper import PaperBroker


class TestRiskEngine:
    def setup_method(self):
        self.config = BotRiskConfig(
            max_risk_per_trade_pct=1.5,
            max_position_size_pct=10.0,
            max_daily_drawdown_pct=3.0,
            default_stop_loss_pct=4.0,
            max_stop_loss_pct=7.0,
        )
        self.engine = RiskEngine(self.config)

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


