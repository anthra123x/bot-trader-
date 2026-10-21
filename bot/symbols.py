"""Symbol normalization and classification utilities for the trading bot."""

from __future__ import annotations

from tradingagents.dataflows.symbols import _CRYPTO_BASES, normalize_symbol as vendor_normalize_symbol


def is_crypto(symbol: str) -> bool:
    """Return True if symbol corresponds to a cryptocurrency instrument."""
    if not symbol:
        return False
    s = symbol.upper().replace("/", "-").strip()
    if "-USD" in s:
        base = s.split("-")[0]
        return base in _CRYPTO_BASES or len(base) <= 6
    if s.endswith("USD") and len(s) > 3:
        base = s[:-3]
        return base in _CRYPTO_BASES
    return s in _CRYPTO_BASES or s in ("BTC", "ETH", "SOL", "USDT", "USDC", "DOGE", "AVAX")


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to standard hyphenated form (e.g. BTC/USD -> BTC-USD, BTCUSD -> BTC-USD)."""
    if not symbol:
        return symbol
    s = symbol.upper().strip()
    if "/" in s:
        return s.replace("/", "-")
    if s.endswith("USD") and "-" not in s and len(s) > 3:
        return f"{s[:-3]}-USD"
    return s


def to_alpaca_symbol(symbol: str) -> str:
    """Convert standard symbol to Alpaca format (e.g. BTC-USD -> BTC/USD)."""
    s = symbol.upper().strip()
    if "-USD" in s:
        return s.replace("-USD", "/USD")
    if s.endswith("USD") and "/" not in s and len(s) > 3:
        return f"{s[:-3]}/USD"
    return s


def to_binance_symbol(symbol: str) -> str:
    """Convert standard symbol to Binance format (e.g. BTC-USD -> BTCUSDT, BTC/USD -> BTCUSDT)."""
    s = symbol.upper().replace("/", "").replace("-", "").strip()
    if s.endswith("USD") and not s.endswith("USDT") and not s.endswith("USDC"):
        return f"{s[:-3]}USDT"
    if not s.endswith("USDT") and not s.endswith("USDC") and not s.endswith("FDUSD"):
        return f"{s}USDT"
    return s

