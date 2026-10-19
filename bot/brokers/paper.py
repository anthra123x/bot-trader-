"""Paper trading broker with SQLite persistence and real-time market data."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

from bot.brokers.base import BaseBroker
from bot.models import AccountBalance, AssetClass, Order, OrderSide, OrderStatus, Position

logger = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    """Simulated execution engine with persistent ledger in SQLite."""

    def __init__(self, db_path: Path | str = "data/paper_trading.db", initial_cash: float = 100_000.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initial_cash = initial_cash
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    currency TEXT NOT NULL,
                    cash REAL NOT NULL,
                    initial_day_equity REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    asset_class TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    limit_price REAL,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    status TEXT NOT NULL,
                    filled_price REAL,
                    filled_at TEXT,
                    rejection_reason TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            # Initialize account if not present
            cur = conn.execute("SELECT id FROM account WHERE id = 1")
            if not cur.fetchone():
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO account (id, currency, cash, initial_day_equity, updated_at) VALUES (1, 'USD', ?, ?, ?)",
                    (self.initial_cash, self.initial_cash, now),
                )
            conn.commit()

    def reset(self, initial_cash: float = 100_000.0) -> None:
        """Reset the paper trading account, clearing positions and orders."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("UPDATE account SET cash = ?, initial_day_equity = ?, updated_at = ? WHERE id = 1", (initial_cash, initial_cash, now))
            conn.execute("DELETE FROM positions")
            conn.execute("DELETE FROM orders")
            conn.commit()

    def get_current_price(self, symbol: str) -> float:
        """Fetch latest market price using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            # Try fast_info first
            price = getattr(ticker.fast_info, "last_price", None)
            if price is not None and price > 0:
                return float(price)
            # Fallback to 1d history
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.warning("Error fetching price for %s: %s", symbol, e)
        raise ValueError(f"Could not retrieve current market price for {symbol}")

    def get_account_balance(self) -> AccountBalance:
        with self._get_connection() as conn:
            row = conn.execute("SELECT currency, cash, initial_day_equity FROM account WHERE id = 1").fetchone()
            cash = float(row["cash"])
            initial_day = float(row["initial_day_equity"])

        positions = self.get_positions()
        portfolio_market_value = sum(pos.market_value for pos in positions)
        total_equity = cash + portfolio_market_value

        return AccountBalance(
            currency="USD",
            cash=cash,
            equity=total_equity,
            buying_power=cash,
            initial_day_equity=initial_day,
        )

    def get_positions(self, fetch_live_price: bool = False) -> list[Position]:
        positions: list[Position] = []
        with self._get_connection() as conn:
            rows = conn.execute("SELECT symbol, asset_class, quantity, entry_price FROM positions WHERE quantity > 0").fetchall()
            for r in rows:
                sym = r["symbol"]
                qty = float(r["quantity"])
                entry = float(r["entry_price"])
                curr_price = entry
                if fetch_live_price:
                    try:
                        curr_price = self.get_current_price(sym)
                    except Exception:
                        curr_price = entry
                positions.append(
                    Position(
                        symbol=sym,
                        asset_class=AssetClass(r["asset_class"]),
                        quantity=qty,
                        entry_price=entry,
                        current_price=curr_price,
                    )
                )
        return positions

    def get_position(self, symbol: str) -> Position | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT symbol, asset_class, quantity, entry_price FROM positions WHERE symbol = ? AND quantity > 0",
                (symbol,),
            ).fetchone()
            if not row:
                return None
            sym = row["symbol"]
            qty = float(row["quantity"])
            entry = float(row["entry_price"])
            try:
                curr_price = self.get_current_price(sym)
            except Exception:
                curr_price = entry
            return Position(
                symbol=sym,
                asset_class=AssetClass(row["asset_class"]),
                quantity=qty,
                entry_price=entry,
                current_price=curr_price,
            )

    def submit_order(self, order: Order) -> Order:
        now = datetime.now(timezone.utc)
        if order.limit_price and order.limit_price > 0:
            fill_price = order.limit_price
        else:
            try:
                fill_price = self.get_current_price(order.symbol)
            except Exception:
                fill_price = 100.0

        with self._get_connection() as conn:
            row = conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()
            current_cash = float(row["cash"])

            if order.side == OrderSide.BUY:
                required_funds = order.quantity * fill_price
                if required_funds > current_cash:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = f"Insufficient funds: required ${required_funds:,.2f}, available ${current_cash:,.2f}"
                    self._save_order(conn, order)
                    return order

                # Update cash
                new_cash = current_cash - required_funds
                conn.execute("UPDATE account SET cash = ?, updated_at = ? WHERE id = 1", (new_cash, now.isoformat()))

                # Update or insert position
                pos = conn.execute("SELECT quantity, entry_price FROM positions WHERE symbol = ?", (order.symbol,)).fetchone()
                if pos:
                    old_qty = float(pos["quantity"])
                    old_entry = float(pos["entry_price"])
                    new_qty = old_qty + order.quantity
                    avg_entry = ((old_qty * old_entry) + (order.quantity * fill_price)) / new_qty
                    conn.execute(
                        "UPDATE positions SET quantity = ?, entry_price = ?, updated_at = ? WHERE symbol = ?",
                        (new_qty, avg_entry, now.isoformat(), order.symbol),
                    )
                else:
                    conn.execute(
                        "INSERT INTO positions (symbol, asset_class, quantity, entry_price, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (order.symbol, order.asset_class.value, order.quantity, fill_price, now.isoformat()),
                    )

                order.status = OrderStatus.FILLED
                order.filled_price = fill_price
                order.filled_at = now
                self._save_order(conn, order)
                conn.commit()
                return order

            elif order.side == OrderSide.SELL:
                pos = conn.execute("SELECT quantity FROM positions WHERE symbol = ?", (order.symbol,)).fetchone()
                current_qty = float(pos["quantity"]) if pos else 0.0

                if current_qty < order.quantity:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = f"Insufficient position: sell qty {order.quantity} > open qty {current_qty}"
                    self._save_order(conn, order)
                    return order

                proceeds = order.quantity * fill_price
                new_cash = current_cash + proceeds
                new_qty = current_qty - order.quantity

                conn.execute("UPDATE account SET cash = ?, updated_at = ? WHERE id = 1", (new_cash, now.isoformat()))
                if new_qty <= 1e-6:
                    conn.execute("DELETE FROM positions WHERE symbol = ?", (order.symbol,))
                else:
                    conn.execute("UPDATE positions SET quantity = ?, updated_at = ? WHERE symbol = ?", (new_qty, now.isoformat(), order.symbol))

                order.status = OrderStatus.FILLED
                order.filled_price = fill_price
                order.filled_at = now
                self._save_order(conn, order)
                conn.commit()
                return order

            else:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = f"Unsupported order side: {order.side}"
                self._save_order(conn, order)
                return order

    def _save_order(self, conn: sqlite3.Connection, order: Order) -> None:
        conn.execute("""
            INSERT OR REPLACE INTO orders (
                order_id, symbol, asset_class, side, order_type, quantity,
                limit_price, stop_loss_price, take_profit_price, status,
                filled_price, filled_at, rejection_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.order_id, order.symbol, order.asset_class.value, order.side.value,
            order.order_type.value, order.quantity, order.limit_price, order.stop_loss_price,
            order.take_profit_price, order.status.value, order.filled_price,
            order.filled_at.isoformat() if order.filled_at else None,
            order.rejection_reason, order.created_at.isoformat(),
        ))

    def cancel_order(self, order_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("UPDATE orders SET status = 'CANCELLED' WHERE order_id = ? AND status = 'PENDING'", (order_id,))
            conn.commit()
            return cur.rowcount > 0
