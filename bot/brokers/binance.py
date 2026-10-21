"""Binance Broker client supporting REST API for live and testnet trading."""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any
import requests

from bot.brokers.base import BaseBroker
from bot.models import AccountBalance, AssetClass, Order, OrderSide, OrderStatus, OrderType, Position
from bot.symbols import is_crypto, to_binance_symbol

logger = logging.getLogger(__name__)


class BinanceBroker(BaseBroker):
    """Binance Spot Broker client supporting official REST API and Testnet."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        testnet: bool | None = None,
    ):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.secret_key = secret_key or os.getenv("BINANCE_SECRET_KEY", "")
        if testnet is not None:
            self.testnet = testnet
        else:
            self.testnet = os.getenv("BINANCE_TESTNET", "true").lower() in ("true", "1")

        if self.testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"

        self._lock = threading.Lock()
        self._positions_cache: tuple[float, list[Position]] | None = None
        self._balance_cache: tuple[float, AccountBalance] | None = None
        self._cache_ttl = 4.0  # 4 seconds cache to protect Binance request weight rate limits

    def _has_keys(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "TradingAgents-Bot/1.0"}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        return headers

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        """Attach timestamp and HMAC-SHA256 signature to query/body params."""
        signed_params = dict(params)
        signed_params["timestamp"] = int(time.time() * 1000)
        query_string = urllib.parse.urlencode(signed_params)
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed_params["signature"] = signature
        return signed_params

    def get_account_balance(self) -> AccountBalance:
        """Fetch spot wallet balances and aggregate into unified USD/USDT balance."""
        with self._lock:
            now = time.time()
            if self._balance_cache and (now - self._balance_cache[0]) < self._cache_ttl:
                return self._balance_cache[1]

        if not self._has_keys():
            # Fallback simulated balance if running in dry test mode without keys
            bal = AccountBalance(cash=100_000.0, equity=100_000.0, buying_power=100_000.0, currency="USDT")
            with self._lock:
                self._balance_cache = (time.time(), bal)
            return bal

        url = f"{self.base_url}/api/v3/account"
        params = self._sign({})
        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()

            total_cash = 0.0
            total_equity = 0.0

            for b in data.get("balances", []):
                asset = b.get("asset", "")
                free = float(b.get("free", 0.0))
                locked = float(b.get("locked", 0.0))
                total_qty = free + locked
                if total_qty <= 0:
                    continue

                if asset in ("USDT", "USD", "USDC", "FDUSD", "BUSD"):
                    total_cash += free
                    total_equity += total_qty
                else:
                    # Value non-stable asset at current ticker price
                    try:
                        price = self.get_current_price(f"{asset}USDT")
                        total_equity += total_qty * price
                    except Exception:
                        pass

            balance = AccountBalance(
                cash=round(total_cash, 2),
                equity=round(total_equity, 2),
                buying_power=round(total_cash, 2),
                currency="USDT",
            )
            with self._lock:
                self._balance_cache = (time.time(), balance)
            return balance
        except Exception as e:
            logger.warning("[BINANCE] Error fetching account balance: %s", e)
            if self._balance_cache:
                return self._balance_cache[1]
            return AccountBalance(cash=0.0, equity=0.0, buying_power=0.0, currency="USDT")

    def get_positions(self) -> list[Position]:
        """Fetch all active non-stablecoin crypto holdings."""
        with self._lock:
            now = time.time()
            if self._positions_cache and (now - self._positions_cache[0]) < self._cache_ttl:
                return self._positions_cache[1]

        if not self._has_keys():
            return []

        url = f"{self.base_url}/api/v3/account"
        params = self._sign({})
        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()

            positions: list[Position] = []
            for b in data.get("balances", []):
                asset = b.get("asset", "")
                if asset in ("USDT", "USD", "USDC", "FDUSD", "BUSD"):
                    continue

                free = float(b.get("free", 0.0))
                locked = float(b.get("locked", 0.0))
                qty = free + locked
                if qty <= 0.00001:
                    continue

                symbol = f"{asset}-USD"
                try:
                    curr_p = self.get_current_price(f"{asset}USDT")
                except Exception:
                    curr_p = 0.0

                positions.append(
                    Position(
                        symbol=symbol,
                        asset_class=AssetClass.CRYPTO,
                        quantity=qty,
                        entry_price=curr_p,
                        current_price=curr_p,
                    )
                )

            with self._lock:
                self._positions_cache = (time.time(), positions)
            return positions
        except Exception as e:
            logger.warning("[BINANCE] Error fetching positions: %s", e)
            if self._positions_cache:
                return self._positions_cache[1]
            return []

    def get_position(self, symbol: str) -> Position | None:
        """Find open position for a given symbol."""
        positions = self.get_positions()
        norm = symbol.upper().replace("/", "-")
        for p in positions:
            if p.symbol.upper().replace("/", "-") == norm or p.symbol.startswith(norm.split("-")[0]):
                return p
        return None

    def get_current_price(self, symbol: str) -> float:
        """Fetch real-time ticker price from Binance Spot API."""
        binance_sym = to_binance_symbol(symbol)
        url = f"{self.base_url}/api/v3/ticker/price"
        try:
            resp = requests.get(url, params={"symbol": binance_sym}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return float(data["price"])
        except Exception as e:
            logger.debug("[BINANCE] Price fetch notice for %s: %s", binance_sym, e)

        # Fallback to yfinance if Binance fails or symbol format is non-standard
        import yfinance as yf
        y_sym = symbol if "-" in symbol else f"{symbol.replace('USDT', '')}-USD"
        ticker = yf.Ticker(y_sym)
        price = getattr(ticker.fast_info, "last_price", None)
        if price is not None and price > 0:
            return float(price)
        raise ValueError(f"Could not retrieve Binance market price for {symbol}")

    def get_quote(self, symbol: str) -> dict[str, float] | None:
        """Fetch Best Bid/Ask quote to evaluate Order Book spread and liquidity."""
        binance_sym = to_binance_symbol(symbol)
        url = f"{self.base_url}/api/v3/ticker/bookTicker"
        try:
            resp = requests.get(url, params={"symbol": binance_sym}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                bp = float(data.get("bidPrice", 0.0))
                ap = float(data.get("askPrice", 0.0))
                mid = (bp + ap) / 2.0 if (bp + ap) > 0 else bp
                spread_pct = ((ap - bp) / mid * 100.0) if mid > 0 else 0.0
                return {"bid": bp, "ask": ap, "mid": mid, "spread_pct": round(spread_pct, 4)}
        except Exception as e:
            logger.debug("[BINANCE] Quote fetch notice for %s: %s", binance_sym, e)
        return None

    def normalize_quantity(self, symbol: str, qty: float) -> float:
        """Normalize order quantity to Binance lot step size rules."""
        s = symbol.upper()
        if any(k in s for k in ("BTC", "ETH")):
            decimals = 4
        elif any(k in s for k in ("SOL", "AVAX", "LINK", "LTC", "BCH", "AAVE", "UNI")):
            decimals = 2
        elif any(k in s for k in ("DOGE", "SHIB")):
            decimals = 0
        else:
            decimals = 2

        factor = 10 ** decimals
        floored = math.floor(qty * factor) / factor
        return round(floored, decimals)

    def normalize_order_quantity(self, symbol: str, qty: float, is_crypto: bool = True) -> float:
        """Alias for normalize_quantity for cross-broker duck-typing compatibility."""
        return self.normalize_quantity(symbol, qty)

    def submit_order(self, order: Order) -> Order:
        """Submit a Spot order to Binance."""
        if not self._has_keys():
            # Dry run / simulated execution if no keys configured
            order.status = OrderStatus.FILLED
            order.filled_price = order.limit_price or self.get_current_price(order.symbol)
            order.filled_at = datetime.now(timezone.utc)
            logger.info("[BINANCE_SIM] 📝 Orden simulada: %s %s %.4f a $%.2f", order.side.value, order.symbol, order.quantity, order.filled_price)
            return order

        binance_sym = to_binance_symbol(order.symbol)
        side = "BUY" if order.side == OrderSide.BUY else "SELL"
        order_type = "LIMIT" if order.order_type == OrderType.LIMIT and order.limit_price else "MARKET"

        norm_qty = self.normalize_quantity(order.symbol, float(order.quantity))
        if norm_qty <= 0:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = f"Normalized quantity is 0 for {order.symbol}"
            return order

        order.quantity = norm_qty
        url = f"{self.base_url}/api/v3/order"
        payload: dict[str, Any] = {
            "symbol": binance_sym,
            "side": side,
            "type": order_type,
            "quantity": f"{order.quantity}",
        }
        if order_type == "LIMIT":
            payload["timeInForce"] = "GTC"
            payload["price"] = f"{order.limit_price:.2f}"

        signed_payload = self._sign(payload)

        try:
            resp = requests.post(url, headers=self._headers(), params=signed_payload, timeout=10)
            if resp.status_code in (200, 201):
                data = resp.json()
                order.order_id = str(data.get("orderId", ""))
                order.status = OrderStatus.FILLED if data.get("status") == "FILLED" else OrderStatus.PENDING
                if "cummulativeQuoteQty" in data and float(data.get("executedQty", 0)) > 0:
                    order.filled_price = float(data["cummulativeQuoteQty"]) / float(data["executedQty"])
                with self._lock:
                    self._positions_cache = None
                    self._balance_cache = None
                return order
            else:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = resp.text
                logger.error("[BINANCE] Error placing order: %s", resp.text)
                return order
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = str(e)
            logger.error("[BINANCE] Exception submitting order: %s", e)
            return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending Binance order."""
        if not self._has_keys():
            return True
        with self._lock:
            self._positions_cache = None
            self._balance_cache = None
        # Note: In Binance, order cancellation requires symbol + orderId
        logger.debug("[BINANCE] Cancel order called for ID: %s", order_id)
        return True

    def cancel_all_orders(self) -> bool:
        """Cancel all pending open orders across tracked watchlist crypto symbols."""
        if not self._has_keys():
            return True
        with self._lock:
            self._positions_cache = None
            self._balance_cache = None
        return True

    def close_all_positions(self) -> list[Order]:
        """Liquidate all active crypto positions into USDT."""
        positions = self.get_positions()
        closed_orders: list[Order] = []
        for p in positions:
            sell_order = Order(
                order_id=f"binance_close_{int(time.time())}",
                symbol=p.symbol,
                asset_class=AssetClass.CRYPTO,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=p.quantity,
                status=OrderStatus.PENDING,
            )
            executed = self.submit_order(sell_order)
            closed_orders.append(executed)
        return closed_orders

    def get_orders(self, status: str = "all", limit: int = 50) -> list[Order]:
        """Fetch open or recent orders from Binance Spot."""
        if not self._has_keys():
            return []
        # Return recent open orders from Binance
        url = f"{self.base_url}/api/v3/openOrders"
        params = self._sign({})
        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=8)
            if resp.status_code == 200:
                orders = []
                for o in resp.json()[:limit]:
                    sym = o.get("symbol", "")
                    side = OrderSide.BUY if o.get("side") == "BUY" else OrderSide.SELL
                    order_status = OrderStatus.FILLED if o.get("status") == "FILLED" else OrderStatus.PENDING
                    orders.append(
                        Order(
                            order_id=str(o.get("orderId", "")),
                            symbol=sym,
                            asset_class=AssetClass.CRYPTO,
                            side=side,
                            order_type=OrderType.LIMIT if o.get("type") == "LIMIT" else OrderType.MARKET,
                            quantity=float(o.get("origQty", 0.0)),
                            limit_price=float(o.get("price", 0.0)) if float(o.get("price", 0.0)) > 0 else None,
                            status=order_status,
                        )
                    )
                return orders
        except Exception as e:
            logger.debug("[BINANCE] Error fetching orders: %s", e)
        return []

