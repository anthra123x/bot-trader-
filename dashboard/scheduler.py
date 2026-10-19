"""Autonomous 24/7 background scheduler for the trading bot."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from bot.models import Order
from bot.runner import BotRunner

logger = logging.getLogger(__name__)


class BotScheduler:
    """Manages recurring automated portfolio scans, position management, and health checks."""

    def __init__(
        self,
        runner: BotRunner,
        order_history: list[dict[str, Any]] | None = None,
        log_buffer: deque[str] | None = None,
        notification_buffer: deque[dict[str, Any]] | None = None,
        scan_interval_seconds: int = 1800,
        position_check_interval_seconds: int = 60,
        web_interval_seconds: int = 900,
    ):
        self.runner = runner
        self.order_history = order_history if order_history is not None else []
        self.log_buffer = log_buffer
        self.notification_buffer = notification_buffer
        self.scan_interval_seconds = scan_interval_seconds
        self.position_check_interval_seconds = position_check_interval_seconds
        self.web_interval_seconds = web_interval_seconds
        self.is_running = False
        self._scan_task: asyncio.Task | None = None
        self._position_task: asyncio.Task | None = None
        self._web_task: asyncio.Task | None = None

    def _log(self, message: str) -> None:
        logger.info(message)
        if self.log_buffer is not None:
            self.log_buffer.append(message)

    def _record_equity_snapshot(self) -> None:
        """Query active broker balance and persist equity telemetry snapshot to Neon PostgreSQL."""
        try:
            bal = self.runner.broker.get_account_balance()
            if hasattr(self.runner, "db") and self.runner.db:
                self.runner.db.save_equity_snapshot(
                    equity=bal.equity,
                    cash=bal.cash,
                    buying_power=bal.buying_power,
                    daily_pnl=bal.daily_pnl,
                    drawdown_pct=bal.daily_drawdown_pct,
                )
        except Exception as e:
            logger.debug("[NEON] Notice recording equity snapshot: %s", e)

    async def start(self) -> None:
        """Start the background autonomous cycles."""
        if self.is_running:
            return
        self.is_running = True
        self._log(
            f"[SCHEDULER] 🚀 Motor autónomo activado. Escaneo de cartera cada {self.scan_interval_seconds // 60}m, "
            f"monitor de Trailing Stop cada {self.position_check_interval_seconds}s, "
            f"investigación web macro cada {self.web_interval_seconds // 60}m."
        )
        # Record initial baseline equity snapshot on startup
        await asyncio.to_thread(self._record_equity_snapshot)
        self._scan_task = asyncio.create_task(self._scan_loop())
        self._position_task = asyncio.create_task(self._position_loop())
        self._web_task = asyncio.create_task(self._web_investigation_loop())

    async def stop(self) -> None:
        """Stop all autonomous scheduler tasks."""
        self.is_running = False
        if self._scan_task:
            self._scan_task.cancel()
        if self._position_task:
            self._position_task.cancel()
        if self._web_task:
            self._web_task.cancel()
        self._log("[SCHEDULER] 🛑 Motor autónomo en pausa.")

    async def _web_investigation_loop(self) -> None:
        """Autonomous 24/7 web investigation loop monitoring macro sentiment, Fear & Greed, and news."""
        # Initial short delay (2s)
        await asyncio.sleep(2)
        while self.is_running:
            try:
                from bot.risk.regime_specialist import REGIME_SPECIALIST
                res = await asyncio.to_thread(REGIME_SPECIALIST.investigate_web_sentiment, True)
                self._log(
                    f"[WEB_RESEARCHER] 🌐 Investigación Web activa: "
                    f"FGI: {res.fear_greed_value}/100 ({res.fear_greed_classification}) | "
                    f"Sentimiento: {res.sentiment_label} ({res.sentiment_score:+.2f}) | "
                    f"{len(res.top_headlines)} noticias analizadas."
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Web investigation loop notice: %s", e)

            await asyncio.sleep(self.web_interval_seconds)

    async def _position_loop(self) -> None:
        """High-frequency position and Trailing Stop protection loop (QuantConnect LEAN)."""
        # Small initial delay to allow startup
        await asyncio.sleep(5)
        ticks = 0
        while self.is_running:
            try:
                # Runs in thread to prevent blocking event loop
                exit_orders: list[Order] = await asyncio.to_thread(self.runner.manage_open_positions)
                ticks += 1
                # Record equity snapshot every 5 position checks (~5 mins)
                if ticks % 5 == 0:
                    await asyncio.to_thread(self._record_equity_snapshot)
                if exit_orders:
                    for order in exit_orders:
                        self.order_history.insert(0, {
                            "order_id": order.order_id,
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "quantity": order.quantity,
                            "status": order.status.value,
                            "filled_price": order.filled_price,
                            "limit_price": order.limit_price,
                            "created_at": order.created_at.isoformat() if hasattr(order.created_at, "isoformat") else str(order.created_at),
                        })
                        if self.notification_buffer is not None:
                            self.notification_buffer.appendleft({
                                "id": str(uuid.uuid4())[:8],
                                "type": "ORDER_EXIT",
                                "title": f"🛡️ Salida: {order.symbol}",
                                "message": f"{order.side.value} {order.quantity:.4f} u a ${order.filled_price or order.limit_price or 0.0:,.2f} ({order.status.value})",
                                "icon": "💰" if order.side.value == "SELL" else "🛡️",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "meta": {"symbol": order.symbol, "quantity": order.quantity, "status": order.status.value},
                            })
                        self._log(f"[EXIT_TRIGGERED] 🛡️ Salida protectora ejecutada para {order.symbol}: {order.quantity} u")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Position management cycle notice: %s", e)

            await asyncio.sleep(self.position_check_interval_seconds)

    async def _scan_loop(self) -> None:
        """Periodic multi-agent analysis loop across the full asset universe."""
        # Initial delay (15s) to ensure web server boot and healthchecks pass before first scan
        await asyncio.sleep(15)
        while self.is_running:
            try:
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                self._log(f"[AUTONOMOUS] ⚡ Iniciando escaneo autónomo de cartera en Alpaca ({now_str})...")

                orders: list[Order] = await asyncio.to_thread(self.runner.run_portfolio_cycle)

                if orders:
                    for order in orders:
                        self.order_history.insert(0, {
                            "order_id": order.order_id,
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "quantity": order.quantity,
                            "status": order.status.value,
                            "filled_price": order.filled_price,
                            "limit_price": order.limit_price,
                            "created_at": order.created_at.isoformat() if hasattr(order.created_at, "isoformat") else str(order.created_at),
                        })
                        if self.notification_buffer is not None:
                            self.notification_buffer.appendleft({
                                "id": str(uuid.uuid4())[:8],
                                "type": "ORDER_BUY",
                                "title": f"⚡ Nueva Orden: {order.symbol}",
                                "message": f"{order.side.value} {order.quantity:.4f} u a ${order.filled_price or order.limit_price or 0.0:,.2f} ({order.status.value})",
                                "icon": "⚡",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "meta": {"symbol": order.symbol, "quantity": order.quantity, "status": order.status.value},
                            })
                    self._log(f"[AUTONOMOUS] ✅ Escaneo finalizado. {len(orders)} órdenes procesadas en Alpaca.")
                else:
                    self._log("[AUTONOMOUS] ✅ Escaneo finalizado. Todos los consensos evaluados sin órdenes requeridas (capital preservado).")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log(f"[ERROR] Error durante ciclo autónomo: {e}")

            # Sleep until next scan
            await asyncio.sleep(self.scan_interval_seconds)
