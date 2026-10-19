"""Brokers package."""

from bot.brokers.base import BaseBroker
from bot.brokers.paper import PaperBroker
from bot.brokers.alpaca import AlpacaBroker


def get_broker(broker_type: str = "paper", **kwargs) -> BaseBroker:
    """Factory to get the appropriate broker implementation."""
    normalized = broker_type.strip().lower()
    if normalized == "alpaca":
        return AlpacaBroker(**kwargs)
    elif normalized == "paper":
        return PaperBroker(**kwargs)
    raise ValueError(f"Unsupported broker type: {broker_type}. Options: 'paper', 'alpaca'")
