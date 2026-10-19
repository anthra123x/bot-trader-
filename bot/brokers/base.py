"""Broker abstraction layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from bot.models import AccountBalance, Order, Position


class BaseBroker(ABC):
    """Abstract interface defining required broker operations."""

    @abstractmethod
    def get_account_balance(self) -> AccountBalance:
        """Fetch current account balances and buying power."""
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Fetch currently held open positions."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        """Fetch open position for a specific symbol."""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Fetch the latest market price for a symbol."""
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit a new order for execution."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        pass
