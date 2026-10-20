"""Dynamic Hot Momentum Universe Scanner.

Palanca A: Dynamically discovers, ranks, and prioritizes top volume and
momentum movers across Alpaca crypto/equity pairs rather than limiting
the engine to a rigid static list.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any
import requests

logger = logging.getLogger(__name__)

# Liquid candidate universe of crypto assets supported on Alpaca
DEFAULT_CANDIDATE_CRYPTOS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
    "DOGE/USD", "NEAR/USD", "UNI/USD", "AAVE/USD", "LTC/USD",
    "BCH/USD", "SHIB/USD", "DOT/USD", "BAT/USD", "CRV/USD",
    "GRT/USD", "MKR/USD", "SUSHI/USD", "XTZ/USD",
]


class HotMomentumScanner:
    """Discovers and ranks top momentum assets in the live market."""

    def __init__(self, cache_ttl_seconds: float = 300.0):
        self.cache_ttl = cache_ttl_seconds
        self._cached_universe: list[str] = []
        self._last_scan_time: float = 0.0

    @staticmethod
    def _to_standard_symbol(alpaca_sym: str) -> str:
        s = alpaca_sym.upper().replace("/", "-")
        if s.endswith("USD") and "-" not in s and len(s) > 3:
            return f"{s[:-3]}-USD"
        return s

    def discover_hot_universe(
        self,
        broker: Any,
        core_symbols: list[str],
        top_n: int = 12,
        force_refresh: bool = False,
    ) -> list[str]:
        """Scan Alpaca crypto snapshots and return a ranked universe merged with core assets."""
        now = time.time()
        if not force_refresh and self._cached_universe and (now - self._last_scan_time) < self.cache_ttl:
            return self._cached_universe

        headers = {}
        if broker and hasattr(broker, "_headers"):
            try:
                headers = broker._headers()
            except Exception:
                headers = {}

        if not headers:
            token = os.getenv("ALPACA_ACCESS_TOKEN")
            if token:
                headers = {"Authorization": f"Bearer {token}"}
            else:
                key = os.getenv("ALPACA_API_KEY")
                secret = os.getenv("ALPACA_SECRET_KEY")
                if key and secret:
                    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

        if not headers:
            logger.debug("[HOT_SCANNER] Sin credenciales Alpaca; preservando lista estática.")
            return core_symbols

        try:
            symbols_query = ",".join(DEFAULT_CANDIDATE_CRYPTOS)
            url = f"https://data.alpaca.markets/v1beta3/crypto/us/snapshots?symbols={requests.utils.quote(symbols_query)}"
            resp = requests.get(url, headers=headers, timeout=4.5)
            if resp.status_code != 200:
                logger.debug("[HOT_SCANNER] HTTP %d al consultar snapshots en Alpaca", resp.status_code)
                return core_symbols

            data = resp.json()
            snapshots = data.get("snapshots", {})
            if not snapshots:
                return core_symbols

            ranked: list[tuple[str, float, float, float]] = []  # (symbol, score, pct_change, vol_usd)
            for sym, snap in snapshots.items():
                daily_bar = snap.get("dailyBar") or {}
                prev_bar = snap.get("prevDailyBar") or {}
                close = float(daily_bar.get("c") or 0.0)
                prev_close = float(prev_bar.get("c") or 0.0)
                vol = float(daily_bar.get("v") or 0.0)
                vwap = float(daily_bar.get("vw") or close)
                vol_usd = vol * vwap

                if close <= 0 or prev_close <= 0 or vol_usd < 5_000:
                    continue

                pct_change = ((close - prev_close) / prev_close) * 100.0
                # Momentum score: rewards positive momentum combined with logarithmic liquidity depth
                momentum_score = pct_change * math.log10(max(vol_usd, 1000.0))
                ranked.append((sym, momentum_score, pct_change, vol_usd))

            # Sort descending by momentum score
            ranked.sort(key=lambda x: x[1], reverse=True)

            # Convert top movers to standard format (e.g. SOL/USD -> SOL-USD)
            top_movers = [self._to_standard_symbol(item[0]) for item in ranked[:top_n]]

            # Combine core watchlist with top movers without duplicates, preserving order
            combined: list[str] = []
            seen: set[str] = set()
            for s in top_movers + core_symbols:
                std = self._to_standard_symbol(s)
                if std not in seen:
                    seen.add(std)
                    combined.append(std)

            logger.info(
                "[HOT_SCANNER] 🔥 Escaneo dinámico completado: %d activos rankeados. Top movers: %s",
                len(ranked), [f"{r[0]} ({r[2]:+.1f}%)" for r in ranked[:4]]
            )
            self._cached_universe = combined
            self._last_scan_time = now
            return combined

        except Exception as e:
            logger.warning("[HOT_SCANNER] Notice ejecutando escáner dinámico de momentum: %s", e)
            return core_symbols


HOT_MOMENTUM_SCANNER = HotMomentumScanner()
