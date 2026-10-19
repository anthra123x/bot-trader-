"""Neon PostgreSQL Repository for TradingAgents Bot.

Provides a thread-safe connection pool, automated schema resilience,
and clean data-access methods for Orders, Agent Debates, Equity Snapshots,
and Machine Learning Mistake Patterns.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from bot.models import Order

logger = logging.getLogger(__name__)


class NeonRepository:
    """Thread-safe PostgreSQL repository connected to Neon Cloud Serverless."""

    def __init__(self, database_url: str | None = None):
        self.database_url = (
            database_url
            or os.getenv("DATABASE_URL")
            or os.getenv("NEON_DATABASE_URL")
        )
        self.pool: ThreadedConnectionPool | None = None
        self._is_enabled = bool(self.database_url)

        if self._is_enabled:
            self._init_pool()
        else:
            logger.warning("[NEON] No DATABASE_URL or NEON_DATABASE_URL found in environment. Neon persistence disabled.")

    def _init_pool(self) -> None:
        """Initialize the PostgreSQL connection pool."""
        if not self.database_url:
            return
        try:
            # Min 2, Max 25 connections for background agents, web requests, and burst traffic
            self.pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=25,
                dsn=self.database_url,
                connect_timeout=10,
            )
            logger.info("[NEON] 🟢 Conexión a Neon PostgreSQL establecida exitosamente (Connection Pool activo: 25 max).")
        except Exception as e:
            logger.error("[NEON] ❌ Error al conectar a Neon PostgreSQL: %s", e)
            self.pool = None

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Context manager to safely acquire, yield, and return a connection to the pool."""
        if not self.pool:
            # Attempt lazy reconnection if pool was dropped
            self._init_pool()
            if not self.pool:
                raise RuntimeError("Neon database connection pool is not available.")

        conn = None
        try:
            conn = self.pool.getconn()
            if conn.closed != 0:
                self.pool.putconn(conn, close=True)
                conn = self.pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise e
        finally:
            if conn and self.pool:
                try:
                    self.pool.putconn(conn)
                except Exception:
                    pass

    def ping(self) -> bool:
        """Check if Neon database is reachable and operational."""
        if not self.database_url:
            return False
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
            return True
        except Exception as e:
            logger.debug("[NEON] Ping failed: %s", e)
            return False

    # --------------------------------------------------------------------------
    # ORDERS
    # --------------------------------------------------------------------------

    def save_order(self, order: Order, alpaca_id: str | None = None) -> bool:
        """Persist or update an executed order in Neon PostgreSQL."""
        if not self._is_enabled:
            return False

        query = """
            INSERT INTO orders (
                id, symbol, side, order_type, quantity, limit_price, filled_price, status, alpaca_id, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                filled_price = EXCLUDED.filled_price,
                alpaca_id = COALESCE(EXCLUDED.alpaca_id, orders.alpaca_id);
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    created_at = order.created_at if isinstance(order.created_at, datetime) else datetime.now(timezone.utc)
                    side_val = order.side.value if hasattr(order.side, "value") else str(order.side)
                    type_val = order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type)
                    status_val = order.status.value if hasattr(order.status, "value") else str(order.status)
                    
                    cur.execute(
                        query,
                        (
                            order.order_id,
                            order.symbol,
                            side_val,
                            type_val,
                            float(order.quantity),
                            float(order.limit_price) if order.limit_price is not None else None,
                            float(order.filled_price) if order.filled_price is not None else None,
                            status_val,
                            alpaca_id,
                            created_at,
                        ),
                    )
            logger.info("[NEON] 💾 Orden %s guardada en Neon PostgreSQL (Símbolo: %s)", order.order_id, order.symbol)
            return True
        except Exception as e:
            logger.error("[NEON] Error guardando orden %s en Neon: %s", order.order_id, e)
            return False

    def save_orders_batch(self, orders: list[Order]) -> bool:
        """Persist or update multiple executed orders in a single connection and transaction."""
        if not self._is_enabled or not orders:
            return False

        query = """
            INSERT INTO orders (
                id, symbol, side, order_type, quantity, limit_price, filled_price, status, alpaca_id, created_at
            ) VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                filled_price = EXCLUDED.filled_price,
                alpaca_id = COALESCE(EXCLUDED.alpaca_id, orders.alpaca_id);
        """
        records = []
        for o in orders:
            created_at = o.created_at if isinstance(o.created_at, datetime) else datetime.now(timezone.utc)
            side_val = o.side.value if hasattr(o.side, "value") else str(o.side)
            type_val = o.order_type.value if hasattr(o.order_type, "value") else str(o.order_type)
            status_val = o.status.value if hasattr(o.status, "value") else str(o.status)
            records.append((
                o.order_id,
                o.symbol,
                side_val,
                type_val,
                float(o.quantity),
                float(o.limit_price) if o.limit_price is not None else None,
                float(o.filled_price) if o.filled_price is not None else None,
                status_val,
                None,
                created_at,
            ))

        try:
            from psycopg2.extras import execute_values
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    execute_values(cur, query, records)
            return True
        except Exception as e:
            logger.debug("[NEON] Batch order save notice: %s", e)
            return False

    def get_recent_orders(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve recent executed orders ordered by creation timestamp."""
        if not self._is_enabled:
            return []

        query = """
            SELECT id AS order_id, symbol, side, order_type, quantity, limit_price, filled_price, status, alpaca_id, created_at
            FROM orders
            ORDER BY created_at DESC
            LIMIT %s;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (limit,))
                    rows = cur.fetchall()
                    result = []
                    for r in rows:
                        row_dict = dict(r)
                        if isinstance(row_dict.get("created_at"), datetime):
                            row_dict["created_at"] = row_dict["created_at"].isoformat()
                        result.append(row_dict)
                    return result
        except Exception as e:
            logger.error("[NEON] Error leyendo órdenes recientes: %s", e)
            return []

    # --------------------------------------------------------------------------
    # AGENT DEBATES
    # --------------------------------------------------------------------------

    def save_debate(
        self,
        symbol: str,
        trade_date: str,
        rating: str,
        action: str,
        current_price: float,
        analyst_signals: dict[str, Any] | None = None,
        reasoning: str = "",
    ) -> int | None:
        """Record multi-agent consensus, bull/bear perspectives, and decisions."""
        if not self._is_enabled:
            return None

        query = """
            INSERT INTO agent_debates (
                symbol, trade_date, rating, action, current_price, analyst_signals, reasoning, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING id;
        """
        try:
            signals_json = json.dumps(analyst_signals or {}, default=str)
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            symbol,
                            trade_date,
                            rating,
                            action,
                            float(current_price),
                            signals_json,
                            reasoning[:4000],  # safeguard text limit
                        ),
                    )
                    inserted_id = cur.fetchone()[0]
            logger.info("[NEON] 🧠 Debate multi-agente #%s registrado para %s (Rating: %s)", inserted_id, symbol, rating)
            return inserted_id
        except Exception as e:
            logger.error("[NEON] Error guardando debate para %s: %s", symbol, e)
            return None

    def get_recent_debates(self, symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch latest multi-agent consensus records."""
        if not self._is_enabled:
            return []

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if symbol:
                        query = """
                            SELECT id, symbol, trade_date, rating, action, current_price, analyst_signals, reasoning, created_at
                            FROM agent_debates
                            WHERE symbol = %s
                            ORDER BY created_at DESC
                            LIMIT %s;
                        """
                        cur.execute(query, (symbol.upper().strip(), limit))
                    else:
                        query = """
                            SELECT id, symbol, trade_date, rating, action, current_price, analyst_signals, reasoning, created_at
                            FROM agent_debates
                            ORDER BY created_at DESC
                            LIMIT %s;
                        """
                        cur.execute(query, (limit,))
                    rows = cur.fetchall()
                    result = []
                    for r in rows:
                        row_dict = dict(r)
                        if isinstance(row_dict.get("created_at"), datetime):
                            row_dict["created_at"] = row_dict["created_at"].isoformat()
                        result.append(row_dict)
                    return result
        except Exception as e:
            logger.error("[NEON] Error leyendo debates: %s", e)
            return []

    # --------------------------------------------------------------------------
    # EQUITY SNAPSHOTS
    # --------------------------------------------------------------------------

    def save_equity_snapshot(
        self,
        equity: float,
        cash: float,
        buying_power: float,
        daily_pnl: float = 0.0,
        drawdown_pct: float = 0.0,
    ) -> bool:
        """Record account telemetry snapshot into Neon for persistent equity curves."""
        if not self._is_enabled:
            return False

        query = """
            INSERT INTO equity_snapshots (
                equity, cash, buying_power, daily_pnl, drawdown_pct, recorded_at
            ) VALUES (%s, %s, %s, %s, %s, NOW());
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            float(equity),
                            float(cash),
                            float(buying_power),
                            float(daily_pnl),
                            float(drawdown_pct),
                        ),
                    )
            return True
        except Exception as e:
            logger.error("[NEON] Error guardando snapshot de equity: %s", e)
            return False

    def get_equity_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve historical equity snapshots for chart rendering."""
        if not self._is_enabled:
            return []

        query = """
            SELECT id, equity, cash, buying_power, daily_pnl, drawdown_pct, recorded_at
            FROM equity_snapshots
            ORDER BY recorded_at ASC
            LIMIT %s;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (limit,))
                    rows = cur.fetchall()
                    result = []
                    for r in rows:
                        row_dict = dict(r)
                        if isinstance(row_dict.get("recorded_at"), datetime):
                            row_dict["recorded_at"] = row_dict["recorded_at"].isoformat()
                        result.append(row_dict)
                    return result
        except Exception as e:
            logger.error("[NEON] Error leyendo historial de equity: %s", e)
            return []

    # --------------------------------------------------------------------------
    # LEARNED PATTERNS
    # --------------------------------------------------------------------------

    def save_learned_pattern(
        self,
        regime: str,
        pattern_type: str,
        symbol: str,
        outcome: str,
        mistake_signature: str = "",
        lesson_learned: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> int | None:
        """Store post-mortem feedback pattern into Neon PostgreSQL."""
        if not self._is_enabled:
            return None

        query = """
            INSERT INTO learned_patterns (
                regime, pattern_type, symbol, outcome, mistake_signature, lesson_learned, metrics, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING id;
        """
        try:
            metrics_json = json.dumps(metrics or {}, default=str)
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            regime,
                            pattern_type,
                            symbol,
                            outcome,
                            mistake_signature,
                            lesson_learned,
                            metrics_json,
                        ),
                    )
                    pattern_id = cur.fetchone()[0]
            logger.info("[NEON] 📚 Patrón de aprendizaje #%s registrado para %s (%s)", pattern_id, symbol, mistake_signature or outcome)
            return pattern_id
        except Exception as e:
            logger.error("[NEON] Error guardando patrón en Neon: %s", e)
            return None

    def get_learned_patterns(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve recent learned mistake patterns and positive edges."""
        if not self._is_enabled:
            return []

        query = """
            SELECT id, regime, pattern_type, symbol, outcome, mistake_signature, lesson_learned, metrics, created_at
            FROM learned_patterns
            ORDER BY created_at DESC
            LIMIT %s;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (limit,))
                    rows = cur.fetchall()
                    result = []
                    for r in rows:
                        row_dict = dict(r)
                        if isinstance(row_dict.get("created_at"), datetime):
                            row_dict["created_at"] = row_dict["created_at"].isoformat()
                        result.append(row_dict)
                    return result
        except Exception as e:
            logger.error("[NEON] Error leyendo patrones de aprendizaje: %s", e)
            return []


# Global Singleton Helper
_GLOBAL_NEON_REPO: NeonRepository | None = None


def get_neon_repo() -> NeonRepository:
    """Retrieve or create the shared global NeonRepository instance."""
    global _GLOBAL_NEON_REPO
    if _GLOBAL_NEON_REPO is None:
        _GLOBAL_NEON_REPO = NeonRepository()
    return _GLOBAL_NEON_REPO
