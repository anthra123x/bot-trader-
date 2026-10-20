"""Order Book Imbalance (Level 2 Depth) Analyzer.

Palanca D: Inspects bid/ask order book depth via Alpaca Market Data API
to detect institutional absorption walls vs seller overhangs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any
import requests

logger = logging.getLogger(__name__)


@dataclass
class OrderBookAnalysis:
    """Outcome of order book level 2 depth evaluation."""
    symbol: str
    imbalance_ratio: float  # bid_depth_usd / ask_depth_usd
    bid_depth_usd: float
    ask_depth_usd: float
    bias: str  # "BULLISH_ABSORPTION", "SELLER_OVERHANG", "NEUTRAL"
    details: str
    spread_pct: float = 0.0  # Bid/Ask spread percentage
    best_bid: float = 0.0
    best_ask: float = 0.0


class OrderBookAnalyzer:
    """Evaluates Level 2 market depth and order book pressure."""

    def __init__(self, cache_ttl_seconds: float = 8.0):
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, OrderBookAnalysis]] = {}

    def _to_alpaca_symbol(self, symbol: str) -> str:
        s = symbol.upper()
        if "-USD" in s:
            return s.replace("-USD", "/USD")
        if s.endswith("USD") and "/" not in s and len(s) > 3:
            return f"{s[:-3]}/USD"
        return s

    def analyze(self, symbol: str, broker: Any = None) -> OrderBookAnalysis:
        """Fetch and calculate order book depth imbalance for a crypto symbol."""
        now = time.time()
        norm_sym = symbol.upper()
        if norm_sym in self._cache:
            ts, cached = self._cache[norm_sym]
            if now - ts < self.cache_ttl:
                return cached

        # Check if asset is crypto (equity L2 requires special enterprise feed)
        is_crypto = "-USD" in norm_sym or "USD" in norm_sym or "/" in norm_sym
        if not is_crypto:
            res = OrderBookAnalysis(
                symbol=symbol,
                imbalance_ratio=1.0,
                bid_depth_usd=0.0,
                ask_depth_usd=0.0,
                bias="NEUTRAL",
                details="L2 Depth no aplicable a instrumentos tradicionales spot sin suscripción OPRA.",
            )
            self._cache[norm_sym] = (now, res)
            return res

        alpaca_sym = self._to_alpaca_symbol(symbol)
        headers = {}
        if broker and hasattr(broker, "_headers"):
            try:
                headers = broker._headers()
            except Exception:
                headers = {}

        # Fallback to env tokens if broker headers empty
        if not headers:
            import os
            token = os.getenv("ALPACA_ACCESS_TOKEN")
            if token:
                headers = {"Authorization": f"Bearer {token}"}
            else:
                key = os.getenv("ALPACA_API_KEY")
                secret = os.getenv("ALPACA_SECRET_KEY")
                if key and secret:
                    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

        if not headers:
            res = OrderBookAnalysis(
                symbol=symbol,
                imbalance_ratio=1.0,
                bid_depth_usd=0.0,
                ask_depth_usd=0.0,
                bias="NEUTRAL",
                details="Credenciales de Alpaca no configuradas; análisis L2 neutral por defecto.",
            )
            self._cache[norm_sym] = (now, res)
            return res

        try:
            url = f"https://data.alpaca.markets/v1beta3/crypto/us/latest/orderbooks?symbols={requests.utils.quote(alpaca_sym)}"
            resp = requests.get(url, headers=headers, timeout=3.5)
            if resp.status_code != 200:
                logger.debug("[ORDER_BOOK] HTTP %d al consultar libro para %s", resp.status_code, alpaca_sym)
                res = OrderBookAnalysis(
                    symbol=symbol,
                    imbalance_ratio=1.0,
                    bid_depth_usd=0.0,
                    ask_depth_usd=0.0,
                    bias="NEUTRAL",
                    details=f"HTTP {resp.status_code} desde Alpaca orderbook API",
                )
                self._cache[norm_sym] = (now, res)
                return res

            data = resp.json()
            orderbooks = data.get("orderbooks", {})
            book = orderbooks.get(alpaca_sym) or {}
            bids = book.get("b", [])  # [{"p": price, "s": size}, ...]
            asks = book.get("a", [])

            if not bids or not asks:
                res = OrderBookAnalysis(
                    symbol=symbol,
                    imbalance_ratio=1.0,
                    bid_depth_usd=0.0,
                    ask_depth_usd=0.0,
                    bias="NEUTRAL",
                    details="Profundidad de libro vacía o insuficiente",
                )
                self._cache[norm_sym] = (now, res)
                return res

            # Extract best bid, best ask and spread percentage
            best_bid = float(bids[0].get("p", 0.0))
            best_ask = float(asks[0].get("p", 0.0))
            mid_price = (best_bid + best_ask) / 2.0 if (best_bid + best_ask) > 0 else best_bid
            spread_pct = round(((best_ask - best_bid) / mid_price) * 100.0, 4) if mid_price > 0 else 0.0

            # Sum top 10 bids & asks in USD value
            bid_depth_usd = sum(float(b.get("p", 0.0)) * float(b.get("s", 0.0)) for b in bids[:10])
            ask_depth_usd = sum(float(a.get("p", 0.0)) * float(a.get("s", 0.0)) for a in asks[:10])

            if ask_depth_usd > 0:
                imbalance_ratio = round(bid_depth_usd / ask_depth_usd, 3)
            else:
                imbalance_ratio = 2.0 if bid_depth_usd > 0 else 1.0

            if imbalance_ratio >= 1.25:
                bias = "BULLISH_ABSORPTION"
                details = f"Muro comprador institucional confirmado: {imbalance_ratio:.2f}x profundidad vs venta (${bid_depth_usd:,.0f} vs ${ask_depth_usd:,.0f})"
            elif imbalance_ratio <= 0.60:
                bias = "SELLER_OVERHANG"
                details = f"Muro vendedor pesado (Overhang): {imbalance_ratio:.2f}x compras vs venta (${bid_depth_usd:,.0f} vs ${ask_depth_usd:,.0f})"
            else:
                bias = "NEUTRAL"
                details = f"Libro balanceado: ratio {imbalance_ratio:.2f}x (${bid_depth_usd:,.0f} bids vs ${ask_depth_usd:,.0f} asks)"

            res = OrderBookAnalysis(
                symbol=symbol,
                imbalance_ratio=imbalance_ratio,
                bid_depth_usd=round(bid_depth_usd, 2),
                ask_depth_usd=round(ask_depth_usd, 2),
                bias=bias,
                details=details,
                spread_pct=spread_pct,
                best_bid=best_bid,
                best_ask=best_ask,
            )
            self._cache[norm_sym] = (now, res)
            return res

        except Exception as e:
            logger.debug("[ORDER_BOOK] Error analizando profundidad para %s: %s", symbol, e)
            res = OrderBookAnalysis(
                symbol=symbol,
                imbalance_ratio=1.0,
                bid_depth_usd=0.0,
                ask_depth_usd=0.0,
                bias="NEUTRAL",
                details=f"Fallback por error de red: {e}",
            )
            self._cache[norm_sym] = (now, res)
            return res


ORDER_BOOK_ANALYZER = OrderBookAnalyzer()
