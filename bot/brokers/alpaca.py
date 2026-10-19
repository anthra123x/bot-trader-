"""Alpaca Trading Broker adapter for US Equities and Crypto.

Supports two authentication modes:
1. Direct REST API via ALPACA_API_KEY and ALPACA_SECRET_KEY in .env.
2. Official Alpaca CLI (alpacahq/cli) session via browser OAuth (`alpaca profile login`).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Any
import requests

from bot.brokers.base import BaseBroker
from bot.models import AccountBalance, AssetClass, Order, OrderSide, OrderStatus, OrderType, Position

logger = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """Alpaca Broker client supporting REST API and Alpaca CLI fallback."""

    def __init__(self, api_key: str | None = None, secret_key: str | None = None, paper: bool = True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.access_token = os.getenv("ALPACA_ACCESS_TOKEN", "")
        self.paper = paper or (os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1"))

        # Auto-detect OAuth access token from official Alpaca CLI profile if direct keys are not set
        if not (self.api_key and self.secret_key) and not self.access_token:
            from pathlib import Path
            oauth_file = Path.home() / ".config" / "alpaca" / "profiles" / "paper.yaml"
            if oauth_file.exists():
                try:
                    import yaml
                    content = yaml.safe_load(oauth_file.read_text()) or {}
                    self.access_token = content.get("access_token", "")
                except Exception:
                    pass
        
        self.base_url = (
            "https://paper-api.alpaca.markets"
            if self.paper
            else "https://api.alpaca.markets"
        )
        self.data_url = "https://data.alpaca.markets"
        self.cli_available = shutil.which("alpaca") is not None

        # Short TTL cache (3.0s) and thread-lock to absorb concurrent frontend/stress polling
        self._balance_cache: tuple[float, AccountBalance] | None = None
        self._positions_cache: tuple[float, list[Position]] | None = None
        self._lock = threading.Lock()

    def _has_direct_keys(self) -> bool:
        return bool((self.api_key and self.secret_key) or self.access_token)

    def _headers(self) -> dict[str, str]:
        if self.access_token:
            return {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def _run_cli(self, args: list[str]) -> Any:
        """Run an Alpaca CLI command and return parsed JSON."""
        if not self.cli_available:
            raise RuntimeError("Alpaca CLI ('alpaca') is not installed or not in PATH.")
        cmd = ["alpaca"] + args
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = res.stdout.strip()
            if output:
                return json.loads(output)
            return {}
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() or e.stdout.strip()
            raise RuntimeError(f"Alpaca CLI error ({' '.join(args)}): {err_msg}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Alpaca CLI output as JSON: {res.stdout}") from e

    def get_account_balance(self) -> AccountBalance:
        now = time.time()
        # Fast path check outside lock
        if self._balance_cache and (now - self._balance_cache[0]) < 3.0:
            return self._balance_cache[1]

        with self._lock:
            # Double check inside lock
            now = time.time()
            if self._balance_cache and (now - self._balance_cache[0]) < 3.0:
                return self._balance_cache[1]

            try:
                if self._has_direct_keys():
                    url = f"{self.base_url}/v2/account"
                    resp = requests.get(url, headers=self._headers(), timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                elif self.cli_available:
                    data = self._run_cli(["account", "get"])
                else:
                    logger.warning("[ALPACA] No credentials or CLI detected; operating in simulated baseline mode.")
                    data = {"cash": 100000.0, "equity": 100000.0, "buying_power": 200000.0, "currency": "USD"}

                cash = float(data.get("cash", 0.0))
                equity = float(data.get("equity", 0.0))
                buying_power = float(data.get("buying_power", 0.0))
                initial_day = float(data.get("last_equity", equity))

                bal = AccountBalance(
                    currency=data.get("currency", "USD"),
                    cash=cash,
                    equity=equity,
                    buying_power=buying_power,
                    initial_day_equity=initial_day,
                )
                self._balance_cache = (now, bal)
                return bal
            except Exception as e:
                if self._balance_cache:
                    logger.warning("Alpaca get_account_balance failed (%s); returning cached balance", e)
                    return self._balance_cache[1]
                logger.warning("Alpaca get_account_balance failed (%s); returning fallback paper balance", e)
                return AccountBalance(
                    currency="USD",
                    cash=100000.0,
                    equity=100000.0,
                    buying_power=200000.0,
                    initial_day_equity=100000.0,
                )

    def get_positions(self) -> list[Position]:
        now = time.time()
        # Fast path check outside lock
        if self._positions_cache and (now - self._positions_cache[0]) < 3.0:
            return self._positions_cache[1]

        with self._lock:
            # Double check inside lock
            now = time.time()
            if self._positions_cache and (now - self._positions_cache[0]) < 3.0:
                return self._positions_cache[1]

            try:
                if self._has_direct_keys():
                    url = f"{self.base_url}/v2/positions"
                    resp = requests.get(url, headers=self._headers(), timeout=10)
                    resp.raise_for_status()
                    positions_data = resp.json()
                elif self.cli_available:
                    positions_data = self._run_cli(["position", "list"])
                else:
                    positions_data = []

                positions: list[Position] = []
                for p in positions_data:
                    sym = p["symbol"]
                    qty = float(p.get("qty", p.get("quantity", 0.0)))
                    entry = float(p.get("avg_entry_price", p.get("entry_price", 0.0)))
                    curr = float(p.get("current_price", entry))
                    asset_class = AssetClass.CRYPTO if "/" in sym or "USD" in sym else AssetClass.EQUITY

                    positions.append(
                        Position(
                            symbol=sym,
                            asset_class=asset_class,
                            quantity=qty,
                            entry_price=entry,
                            current_price=curr,
                        )
                    )
                self._positions_cache = (now, positions)
                return positions
            except Exception as e:
                if self._positions_cache:
                    logger.warning("Alpaca get_positions failed (%s); returning cached positions", e)
                    return self._positions_cache[1]
                logger.warning("Alpaca get_positions failed (%s); returning empty positions list", e)
                return []

    def get_position(self, symbol: str) -> Position | None:
        if self._has_direct_keys():
            url = f"{self.base_url}/v2/positions/{symbol}"
            resp = requests.get(url, headers=self._headers(), timeout=10)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            p = resp.json()
        else:
            try:
                p = self._run_cli(["position", "get", "--symbol", symbol])
            except RuntimeError:
                return None

        sym = p["symbol"]
        qty = float(p.get("qty", p.get("quantity", 0.0)))
        entry = float(p.get("avg_entry_price", p.get("entry_price", 0.0)))
        curr = float(p.get("current_price", entry))
        return Position(
            symbol=sym,
            asset_class=AssetClass.CRYPTO if "/" in symbol or "USD" in symbol else AssetClass.EQUITY,
            quantity=qty,
            entry_price=entry,
            current_price=curr,
        )

    def get_current_price(self, symbol: str) -> float:
        alpaca_sym = self._to_alpaca_symbol(symbol)
        is_crypto = "/" in alpaca_sym or "-USD" in symbol.upper()

        if self._has_direct_keys():
            endpoint = f"{self.data_url}/v2/crypto/{alpaca_sym}/trades/latest" if is_crypto else f"{self.data_url}/v2/stocks/{symbol}/trades/latest"
            try:
                resp = requests.get(endpoint, headers=self._headers(), timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    trade = data.get("trade", {})
                    if "p" in trade:
                        return float(trade["p"])
            except Exception:
                pass

        if self.cli_available:
            try:
                if is_crypto:
                    res = self._run_cli(["data", "crypto", "latest-quotes", "--symbols", alpaca_sym])
                    quotes = res.get("quotes", {})
                    q = quotes.get(alpaca_sym) or (next(iter(quotes.values())) if quotes else {})
                    if "ap" in q and "bp" in q:
                        return round((float(q["ap"]) + float(q["bp"])) / 2.0, 2)
                else:
                    res = self._run_cli(["data", "snapshot", "--symbol", symbol])
                    trade = res.get("latestTrade", {})
                    if "p" in trade:
                        return float(trade["p"])
                    bar = res.get("dailyBar", {})
                    if "c" in bar:
                        return float(bar["c"])
            except Exception as e:
                logger.debug("Alpaca CLI quote fetch fallback for %s: %s", symbol, e)

        # Fallback to yfinance
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        price = getattr(ticker.fast_info, "last_price", None)
        if price is not None and price > 0:
            return float(price)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        raise ValueError(f"Could not retrieve current market price for {symbol}")

    def _to_alpaca_symbol(self, symbol: str) -> str:
        """Convert standard symbols to Alpaca format (e.g. BTC-USD -> BTC/USD, BTCUSD -> BTC/USD)."""
        s = symbol.upper()
        if "-USD" in s:
            return s.replace("-USD", "/USD")
        if s.endswith("USD") and "/" not in s and len(s) > 3:
            return f"{s[:-3]}/USD"
        return s

    def submit_order(self, order: Order) -> Order:
        side = "buy" if order.side == OrderSide.BUY else "sell"
        alpaca_sym = self._to_alpaca_symbol(order.symbol)
        is_crypto = "/" in alpaca_sym or "-USD" in order.symbol.upper()
        time_in_force = "gtc" if is_crypto else "day"
        
        if self._has_direct_keys():
            url = f"{self.base_url}/v2/orders"
            order_payload: dict[str, Any] = {
                "symbol": alpaca_sym,
                "qty": str(order.quantity),
                "side": side,
                "type": order.order_type.value.lower(),
                "time_in_force": time_in_force,
            }
            if order.order_type == OrderType.LIMIT and order.limit_price:
                order_payload["limit_price"] = str(order.limit_price)
            if order.stop_loss_price:
                order_payload["stop_loss"] = {"stop_price": str(order.stop_loss_price)}
            if order.take_profit_price:
                order_payload["take_profit"] = {"limit_price": str(order.take_profit_price)}

            resp = requests.post(url, headers=self._headers(), json=order_payload, timeout=10)
            if resp.status_code in (200, 201):
                data = resp.json()
                order.order_id = data["id"]
                order.status = OrderStatus.PENDING
                return order
            else:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = resp.text
                return order
        else:
            # Submit via CLI
            cli_args = [
                "order", "submit",
                "--symbol", alpaca_sym,
                "--qty", str(order.quantity),
                "--side", side,
                "--type", order.order_type.value.lower(),
                "--time-in-force", time_in_force,
            ]
            if order.order_type == OrderType.LIMIT and order.limit_price:
                cli_args.extend(["--limit-price", str(order.limit_price)])
            if order.stop_loss_price:
                cli_args.extend(["--stop-price", str(order.stop_loss_price)])

            try:
                res = self._run_cli(cli_args)
                order.order_id = str(res.get("id", ""))
                order.status = OrderStatus.PENDING
                return order
            except Exception as e:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = str(e)
                return order

    def cancel_order(self, order_id: str) -> bool:
        if self._has_direct_keys():
            url = f"{self.base_url}/v2/orders/{order_id}"
            resp = requests.delete(url, headers=self._headers(), timeout=10)
            return resp.status_code in (200, 204)
        else:
            try:
                self._run_cli(["order", "cancel", "--order-id", order_id])
                return True
            except Exception:
                return False

    def get_orders(self, status: str = "all", limit: int = 50) -> list[Order]:
        """Fetch historical and pending orders from Alpaca."""
        if self._has_direct_keys():
            url = f"{self.base_url}/v2/orders"
            params = {"status": status, "limit": limit}
            try:
                resp = requests.get(url, headers=self._headers(), params=params, timeout=10)
                if resp.status_code == 200:
                    orders = []
                    for o in resp.json():
                        side = OrderSide.BUY if o.get("side") == "buy" else OrderSide.SELL
                        sym = o.get("symbol", "")
                        filled_p = float(o.get("filled_avg_price")) if o.get("filled_avg_price") else None
                        c_at = o.get("created_at") or o.get("submitted_at")
                        raw_status = o.get("status", "").lower()
                        if raw_status in ("filled", "partially_filled"):
                            order_status = OrderStatus.FILLED
                        elif raw_status in ("canceled", "cancelled", "expired"):
                            order_status = OrderStatus.CANCELLED
                        elif raw_status in ("rejected", "suspended"):
                            order_status = OrderStatus.REJECTED
                        else:
                            order_status = OrderStatus.PENDING

                        orders.append(
                            Order(
                                order_id=o.get("id", ""),
                                symbol=sym,
                                asset_class=AssetClass.CRYPTO if "/" in sym or "USD" in sym else AssetClass.EQUITY,
                                side=side,
                                order_type=OrderType.MARKET if o.get("type") == "market" else OrderType.LIMIT,
                                quantity=float(o.get("qty", 0.0) or 0.0),
                                limit_price=float(o.get("limit_price", 0.0) or 0.0) if o.get("limit_price") else None,
                                filled_price=filled_p,
                                status=order_status,
                            )
                        )
                    return orders
            except Exception as e:
                logger.debug("Failed to fetch orders from Alpaca REST: %s", e)
        return []
