"""Hybrid Multi-Broker routing US Equities to Alpaca and Cryptocurrencies to Binance."""

from __future__ import annotations

import logging
from typing import Any

from bot.brokers.base import BaseBroker
from bot.models import AccountBalance, AssetClass, Order, Position
from bot.symbols import is_crypto

logger = logging.getLogger(__name__)


class HybridBroker(BaseBroker):
    """Hybrid broker routing orders and market data to specialized execution venues:
    - US Equities / Stocks -> AlpacaBroker (0% commission during regular trading sessions)
    - Cryptocurrencies -> BinanceBroker (24/7 continuous liquidity, deep order book, ultra-low spreads)
    """

    def __init__(
        self,
        equity_broker: BaseBroker | None = None,
        crypto_broker: BaseBroker | None = None,
        **kwargs: Any,
    ):
        if equity_broker is not None:
            self.equity_broker = equity_broker
        else:
            from bot.brokers.alpaca import AlpacaBroker
            self.equity_broker = AlpacaBroker(**kwargs)

        if crypto_broker is not None:
            self.crypto_broker = crypto_broker
        else:
            from bot.brokers.binance import BinanceBroker
            self.crypto_broker = BinanceBroker(**kwargs)

        logger.info(
            "[HYBRID_BROKER] Initialized HybridBroker (Equities: %s, Crypto: %s)",
            self.equity_broker.__class__.__name__,
            self.crypto_broker.__class__.__name__,
        )

    def _get_broker_for_symbol(self, symbol: str) -> BaseBroker:
        """Select appropriate venue based on asset classification."""
        return self.crypto_broker if is_crypto(symbol) else self.equity_broker

    def _get_broker_for_order(self, order: Order) -> BaseBroker:
        """Select appropriate venue based on order asset class and ticker."""
        if getattr(order, "asset_class", None) == AssetClass.CRYPTO or is_crypto(order.symbol):
            return self.crypto_broker
        return self.equity_broker

    def get_account_balance(self) -> AccountBalance:
        """Aggregate account balances and buying power across both brokers."""
        eq_bal = self.equity_broker.get_account_balance()
        cr_bal = self.crypto_broker.get_account_balance()

        total_cash = round(eq_bal.cash + cr_bal.cash, 2)
        total_equity = round(eq_bal.equity + cr_bal.equity, 2)
        total_buying_power = round(eq_bal.buying_power + cr_bal.buying_power, 2)
        total_initial_equity = round(
            getattr(eq_bal, "initial_day_equity", 0.0) + getattr(cr_bal, "initial_day_equity", 0.0),
            2,
        )

        return AccountBalance(
            currency="USD",
            cash=total_cash,
            equity=total_equity,
            buying_power=total_buying_power,
            initial_day_equity=total_initial_equity,
            equity_cash=round(eq_bal.cash, 2),
            crypto_cash=round(cr_bal.cash, 2),
        )


    def get_equity_balance(self) -> AccountBalance:
        """Fetch balance specifically isolated to the equity broker (Alpaca)."""
        return self.equity_broker.get_account_balance()

    def get_crypto_balance(self) -> AccountBalance:
        """Fetch balance specifically isolated to the crypto broker (Binance)."""
        return self.crypto_broker.get_account_balance()

    def get_positions(self) -> list[Position]:
        """Fetch and combine all open positions across equity and crypto venues."""
        eq_pos = self.equity_broker.get_positions() or []
        cr_pos = self.crypto_broker.get_positions() or []
        return eq_pos + cr_pos

    def get_position(self, symbol: str) -> Position | None:
        """Fetch open position for a given symbol from its designated broker."""
        target_broker = self._get_broker_for_symbol(symbol)
        pos = target_broker.get_position(symbol)
        if pos is not None:
            return pos
        # Fallback check alternate broker in case symbol classification was ambiguous
        alt_broker = self.equity_broker if target_broker is self.crypto_broker else self.crypto_broker
        return alt_broker.get_position(symbol)

    def get_current_price(self, symbol: str) -> float:
        """Fetch real-time market price from the venue hosting the symbol."""
        broker = self._get_broker_for_symbol(symbol)
        return broker.get_current_price(symbol)

    def get_quote(self, symbol: str) -> dict[str, float] | None:
        """Fetch Best Bid / Best Ask and spread metrics from the designated venue."""
        broker = self._get_broker_for_symbol(symbol)
        if hasattr(broker, "get_quote"):
            return broker.get_quote(symbol)
        return None

    def normalize_order_quantity(
        self, symbol: str, qty: float, is_crypto_asset: bool | None = None
    ) -> float:
        """Normalize quantity according to the target broker's lot size rules."""
        if is_crypto_asset is None:
            is_crypto_asset = is_crypto(symbol)

        broker = self.crypto_broker if is_crypto_asset else self.equity_broker

        if hasattr(broker, "normalize_order_quantity"):
            return broker.normalize_order_quantity(symbol, qty, is_crypto=is_crypto_asset)
        elif hasattr(broker, "normalize_quantity"):
            return broker.normalize_quantity(symbol, qty)
        return qty

    def submit_order(self, order: Order) -> Order:
        """Route order execution to the specialized broker."""
        broker = self._get_broker_for_order(order)
        logger.info(
            "[HYBRID_ROUTER] Routing %s %s %.4f to venue %s",
            order.side.value,
            order.symbol,
            order.quantity,
            broker.__class__.__name__,
        )
        return broker.submit_order(order)

    def cancel_order(self, order_id: str) -> bool:
        """Attempt order cancellation across brokers."""
        eq_res = self.equity_broker.cancel_order(order_id)
        cr_res = self.crypto_broker.cancel_order(order_id)
        return eq_res or cr_res

    def cancel_all_orders(self) -> bool:
        """Cancel all pending orders across both venues."""
        eq_res = self.equity_broker.cancel_all_orders()
        cr_res = self.crypto_broker.cancel_all_orders()
        return eq_res and cr_res

    def close_all_positions(self) -> list[Any]:
        """Liquidate all open positions across equity and crypto venues."""
        closed_orders: list[Any] = []
        try:
            closed_orders.extend(self.equity_broker.close_all_positions())
        except Exception as e:
            logger.error("[HYBRID] Error closing equity positions: %s", e)

        try:
            closed_orders.extend(self.crypto_broker.close_all_positions())
        except Exception as e:
            logger.error("[HYBRID] Error closing crypto positions: %s", e)

        return closed_orders

    def get_orders(self, status: str = "all", limit: int = 50) -> list[Order]:
        """Fetch and merge recent orders from both brokers."""
        all_orders: list[Order] = []
        if hasattr(self.equity_broker, "get_orders"):
            try:
                all_orders.extend(self.equity_broker.get_orders(status=status, limit=limit))
            except Exception as e:
                logger.debug("[HYBRID] Error fetching equity orders: %s", e)

        if hasattr(self.crypto_broker, "get_orders"):
            try:
                all_orders.extend(self.crypto_broker.get_orders(status=status, limit=limit))
            except Exception as e:
                logger.debug("[HYBRID] Error fetching crypto orders: %s", e)

        # Sort descending by creation date if available
        all_orders.sort(
            key=lambda o: getattr(o, "created_at", None) or o.filled_at or 0,
            reverse=True,
        )
        return all_orders[:limit]
