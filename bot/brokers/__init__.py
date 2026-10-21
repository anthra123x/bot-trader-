"""Brokers package."""

from bot.brokers.base import BaseBroker
from bot.brokers.paper import PaperBroker
from bot.brokers.alpaca import AlpacaBroker
from bot.brokers.binance import BinanceBroker
from bot.brokers.hybrid import HybridBroker

__all__ = ["BaseBroker", "PaperBroker", "AlpacaBroker", "BinanceBroker", "HybridBroker", "get_broker"]


def get_broker(broker_type: str = "paper", **kwargs) -> BaseBroker:
    """Factory to get the appropriate broker implementation."""
    normalized = broker_type.strip().lower()
    if normalized == "alpaca":
        return AlpacaBroker(**kwargs)
    elif normalized == "binance":
        return BinanceBroker(**kwargs)
    elif normalized == "hybrid":
        return HybridBroker(**kwargs)
    elif normalized == "paper":
        return PaperBroker(**kwargs)
    raise ValueError(f"Unsupported broker type: {broker_type}. Options: 'paper', 'alpaca', 'binance', 'hybrid'")

