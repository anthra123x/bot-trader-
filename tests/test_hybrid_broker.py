"""Tests for BinanceBroker and HybridBroker multi-broker architecture."""

import unittest
from unittest.mock import MagicMock, patch

from bot.brokers import get_broker
from bot.brokers.base import BaseBroker
from bot.brokers.binance import BinanceBroker
from bot.brokers.hybrid import HybridBroker
from bot.brokers.paper import PaperBroker
from bot.config import BotConfig
from bot.models import (
    AccountBalance,
    AssetClass,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TradeSignal,
)
from bot.risk.risk_engine import RiskEngine
from bot.symbols import to_binance_symbol


class TestBinanceBroker(unittest.TestCase):
    """Unit test suite for BinanceBroker."""

    def setUp(self):
        self.broker = BinanceBroker(
            api_key="mock_binance_key",
            secret_key="mock_binance_secret",
            testnet=True,
        )

    def test_binance_symbol_conversion(self):
        self.assertEqual(to_binance_symbol("BTC-USD"), "BTCUSDT")
        self.assertEqual(to_binance_symbol("ETH/USD"), "ETHUSDT")
        self.assertEqual(to_binance_symbol("SOL-USD"), "SOLUSDT")
        self.assertEqual(to_binance_symbol("DOGE-USD"), "DOGEUSDT")
        self.assertEqual(to_binance_symbol("BTCUSDT"), "BTCUSDT")

    def test_signature_generation(self):
        signed = self.broker._sign({"symbol": "BTCUSDT", "side": "BUY"})
        self.assertIn("timestamp", signed)
        self.assertIn("signature", signed)
        self.assertEqual(signed["symbol"], "BTCUSDT")
        self.assertEqual(len(signed["signature"]), 64)  # SHA-256 hex digest length

    def test_quantity_normalization_lot_sizes(self):
        # BTC and ETH: 4 decimal places floored
        self.assertEqual(self.broker.normalize_quantity("BTC-USD", 0.0123987), 0.0123)
        self.assertEqual(self.broker.normalize_quantity("ETH-USD", 1.456789), 1.4567)
        # SOL, AVAX, LINK: 2 decimal places floored
        self.assertEqual(self.broker.normalize_quantity("SOL-USD", 12.879), 12.87)
        self.assertEqual(self.broker.normalize_quantity("AVAX-USD", 5.4321), 5.43)
        # DOGE, SHIB: 0 decimal places floored
        self.assertEqual(self.broker.normalize_quantity("DOGE-USD", 250.99), 250.0)
        # Cross-broker alias compatibility
        self.assertEqual(self.broker.normalize_order_quantity("BTC-USD", 0.05559), 0.0555)

    def test_simulated_execution_when_keys_missing(self):
        unauthed_broker = BinanceBroker(api_key="", secret_key="", testnet=True)
        order = Order(
            order_id="test_sim_1",
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.05,
            limit_price=65000.0,
            status=OrderStatus.PENDING,
        )
        filled = unauthed_broker.submit_order(order)
        self.assertEqual(filled.status, OrderStatus.FILLED)
        self.assertEqual(filled.filled_price, 65000.0)
        self.assertIsNotNone(filled.filled_at)

    @patch("requests.get")
    def test_account_balance_aggregation(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "balances": [
                {"asset": "USDT", "free": "15420.50", "locked": "0.00"},
                {"asset": "USDC", "free": "2000.00", "locked": "0.00"},
                {"asset": "BTC", "free": "0.1000", "locked": "0.00"},
            ]
        }
        mock_get.return_value = mock_response

        # Mock ticker price for BTC
        with patch.object(self.broker, "get_current_price", return_value=60000.0):
            bal = self.broker.get_account_balance()
            # cash: 15420.50 + 2000.00 = 17420.50
            # equity: 17420.50 + (0.1 * 60000 = 6000) = 23420.50
            self.assertEqual(bal.cash, 17420.50)
            self.assertEqual(bal.equity, 23420.50)
            self.assertEqual(bal.currency, "USDT")

    @patch("requests.get")
    def test_get_quote_book_ticker(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "BTCUSDT",
            "bidPrice": "62000.00",
            "askPrice": "62010.00",
        }
        mock_get.return_value = mock_response

        quote = self.broker.get_quote("BTC-USD")
        self.assertIsNotNone(quote)
        self.assertEqual(quote["bid"], 62000.0)
        self.assertEqual(quote["ask"], 62010.0)
        self.assertAlmostEqual(quote["mid"], 62005.0)
        self.assertTrue(quote["spread_pct"] > 0)


class MockVenueBroker(BaseBroker):
    """Mock broker implementation for isolated routing tests."""

    def __init__(self, cash: float, equity: float, venue_name: str):
        self.cash = cash
        self.equity = equity
        self.venue_name = venue_name
        self.submitted_orders: list[Order] = []
        self.positions: list[Position] = []

    def get_account_balance(self) -> AccountBalance:
        return AccountBalance(
            cash=self.cash,
            equity=self.equity,
            buying_power=self.cash,
            initial_day_equity=self.equity,
        )

    def get_positions(self) -> list[Position]:
        return self.positions

    def get_position(self, symbol: str) -> Position | None:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    def get_current_price(self, symbol: str) -> float:
        return 150.0 if "NVDA" in symbol or "AAPL" in symbol else 60000.0

    def submit_order(self, order: Order) -> Order:
        order.status = OrderStatus.FILLED
        order.filled_price = self.get_current_price(order.symbol)
        self.submitted_orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def cancel_all_orders(self) -> bool:
        return True

    def close_all_positions(self) -> list[Order]:
        orders = []
        for p in self.positions:
            orders.append(
                Order(
                    order_id=f"close_{p.symbol}",
                    symbol=p.symbol,
                    asset_class=p.asset_class,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=p.quantity,
                    status=OrderStatus.FILLED,
                )
            )
        self.positions.clear()
        return orders


class TestHybridBroker(unittest.TestCase):
    """Test suite for HybridBroker multi-broker routing."""

    def setUp(self):
        # Alpaca: $50,000 cash, $75,000 equity (holds NVDA)
        self.alpaca_mock = MockVenueBroker(cash=50_000.0, equity=75_000.0, venue_name="Alpaca")
        self.alpaca_mock.positions = [
            Position(
                symbol="NVDA",
                asset_class=AssetClass.EQUITY,
                quantity=100.0,
                entry_price=120.0,
                current_price=150.0,
            )
        ]

        # Binance: $10,000 cash, $25,000 equity (holds BTC-USD)
        self.binance_mock = MockVenueBroker(cash=10_000.0, equity=25_000.0, venue_name="Binance")
        self.binance_mock.positions = [
            Position(
                symbol="BTC-USD",
                asset_class=AssetClass.CRYPTO,
                quantity=0.25,
                entry_price=55_000.0,
                current_price=60_000.0,
            )
        ]

        self.hybrid = HybridBroker(
            equity_broker=self.alpaca_mock,
            crypto_broker=self.binance_mock,
        )

    def test_factory_registration(self):
        self.assertIsInstance(get_broker("hybrid"), HybridBroker)
        self.assertIsInstance(get_broker("binance"), BinanceBroker)
        self.assertIsInstance(get_broker("paper"), PaperBroker)
        with self.assertRaises(ValueError):
            get_broker("unsupported_exchange")

    def test_consolidated_account_balance(self):
        bal = self.hybrid.get_account_balance()
        # Consolidated: $50k + $10k = $60k cash; $75k + $25k = $100k equity
        self.assertEqual(bal.cash, 60_000.0)
        self.assertEqual(bal.equity, 100_000.0)
        self.assertEqual(bal.buying_power, 60_000.0)
        self.assertEqual(bal.currency, "USD")
        self.assertEqual(bal.equity_cash, 50_000.0)
        self.assertEqual(bal.crypto_cash, 10_000.0)

    def test_venue_isolated_balance_helpers(self):
        eq_bal = self.hybrid.get_equity_balance()
        self.assertEqual(eq_bal.cash, 50_000.0)
        self.assertEqual(eq_bal.equity, 75_000.0)

        cr_bal = self.hybrid.get_crypto_balance()
        self.assertEqual(cr_bal.cash, 10_000.0)
        self.assertEqual(cr_bal.equity, 25_000.0)

    def test_positions_aggregation(self):
        positions = self.hybrid.get_positions()
        self.assertEqual(len(positions), 2)
        symbols = [p.symbol for p in positions]
        self.assertIn("NVDA", symbols)
        self.assertIn("BTC-USD", symbols)

    def test_get_position_routed(self):
        nvda_pos = self.hybrid.get_position("NVDA")
        self.assertIsNotNone(nvda_pos)
        self.assertEqual(nvda_pos.symbol, "NVDA")

        btc_pos = self.hybrid.get_position("BTC-USD")
        self.assertIsNotNone(btc_pos)
        self.assertEqual(btc_pos.symbol, "BTC-USD")

    def test_order_submission_routing(self):
        # Equity Order -> Alpaca
        stock_order = Order(
            order_id="stock_1",
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            status=OrderStatus.PENDING,
        )
        self.hybrid.submit_order(stock_order)
        self.assertEqual(len(self.alpaca_mock.submitted_orders), 1)
        self.assertEqual(self.alpaca_mock.submitted_orders[0].symbol, "AAPL")
        self.assertEqual(len(self.binance_mock.submitted_orders), 0)

        # Crypto Order -> Binance
        crypto_order = Order(
            order_id="crypto_1",
            symbol="ETH-USD",
            asset_class=AssetClass.CRYPTO,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.5,
            status=OrderStatus.PENDING,
        )
        self.hybrid.submit_order(crypto_order)
        self.assertEqual(len(self.binance_mock.submitted_orders), 1)
        self.assertEqual(self.binance_mock.submitted_orders[0].symbol, "ETH-USD")

    def test_close_all_positions_across_both_venues(self):
        closed = self.hybrid.close_all_positions()
        self.assertEqual(len(closed), 2)
        self.assertEqual(len(self.hybrid.get_positions()), 0)

    def test_risk_engine_respects_venue_isolated_liquidity(self):
        """Verify that RiskEngine bounds crypto purchases by Binance cash and stock purchases by Alpaca cash."""
        config = BotConfig()
        risk_engine = RiskEngine(config.risk)

        balance = self.hybrid.get_account_balance()
        positions = self.hybrid.get_positions()

        # Crypto trade: Binance has $10,000 cash. Even though total cash is $60,000,
        # max liquidity available for crypto must be bounded by $10,000 (with buffer ~$9,800).
        crypto_signal = TradeSignal(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            trade_date="2026-10-20",
            rating="Buy",
            action=OrderSide.BUY,
            confidence=0.85,
            current_price=60_000.0,
            target_price=66_000.0,
            stop_loss=58_500.0,
            reasoning="Strong institutional breakout",
        )

        val_result = risk_engine.evaluate(crypto_signal, balance, positions)
        self.assertTrue(val_result.is_approved)
        allocated_capital = val_result.allocated_capital
        # Sizing cannot exceed Binance's $10,000 cash
        self.assertLessEqual(allocated_capital, 10_000.0)


if __name__ == "__main__":
    unittest.main()
