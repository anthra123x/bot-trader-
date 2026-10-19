"""Notification and telemetry service for the bot."""

from __future__ import annotations

import logging
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bot.models import Order, RiskValidationResult, TradeSignal

logger = logging.getLogger(__name__)
console = Console()


class Notifier:
    """Dispatches trade decisions and order updates to console and webhooks."""

    def __init__(self, telegram_token: str | None = None, telegram_chat_id: str | None = None, discord_webhook: str | None = None):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook = discord_webhook

    def notify_signal_and_risk(self, signal: TradeSignal, risk: RiskValidationResult) -> None:
        """Render signal analysis and risk check to the terminal."""
        table = Table(title=f"🤖 Decision Multi-Agente: {signal.symbol} ({signal.asset_class.value})")
        table.add_column("Métrica / Parámetro", style="cyan")
        table.add_column("Valor", style="bold green" if risk.is_approved else "bold red")

        table.add_row("Rating", signal.rating)
        table.add_row("Acción Propuesta", signal.action.value)
        table.add_row("Precio Actual", f"${signal.current_price:,.2f}")
        table.add_row("Confianza", f"{signal.confidence * 100:.1f}%")
        table.add_row("Aprobado por Riesgo", "✅ SÍ" if risk.is_approved else "❌ NO")

        if risk.is_approved:
            table.add_row("Cantidad Asignada", f"{risk.adjusted_quantity:.4f}")
            table.add_row("Capital Comprometido", f"${risk.allocated_capital:,.2f}")
            if risk.stop_loss_price:
                table.add_row("Stop Loss", f"${risk.stop_loss_price:,.2f}")
            if risk.take_profit_price:
                table.add_row("Take Profit", f"${risk.take_profit_price:,.2f}")
        else:
            table.add_row("Motivo de Rechazo", "\n".join(risk.rejection_reasons))

        if risk.risk_warnings:
            table.add_row("Advertencias", "\n".join(risk.risk_warnings))

        console.print(table)

    def notify_order_execution(self, order: Order) -> None:
        """Render order fill or rejection to terminal and webhooks."""
        status_color = "green" if order.status.value == "FILLED" else "red"
        title = f"Orden {order.status.value}: {order.side.value} {order.quantity} {order.symbol}"
        content = (
            f"**ID:** `{order.order_id}`\n"
            f"**Símbolo:** {order.symbol}\n"
            f"**Tipo:** {order.order_type.value}\n"
            f"**Precio Ejecución:** ${order.filled_price:,.2f}" if order.filled_price else ""
        )
        if order.rejection_reason:
            content += f"\n**Motivo:** {order.rejection_reason}"

        console.print(Panel(content, title=title, border_style=status_color))

        # Webhook dispatch
        self._send_discord(title, content)
        self._send_telegram(f"{title}\n{content}")

    def notify_alert(self, title: str, message: str, alert_type: str = "INFO") -> None:
        """Send generic system notification (e.g. Partial TP, Rebalance, Trailing stop, Regime alert)."""
        color = "green" if alert_type == "SUCCESS" else ("yellow" if alert_type == "WARNING" else "cyan")
        console.print(Panel(message, title=title, border_style=color))
        self._send_discord(title, message)
        self._send_telegram(f"🔔 *{title}*\n{message}")

    def _send_discord(self, title: str, description: str) -> None:
        if not self.discord_webhook:
            return
        try:
            payload = {
                "embeds": [{
                    "title": title,
                    "description": description,
                    "color": 3066993 if "FILLED" in title else 15158332,
                }]
            }
            requests.post(self.discord_webhook, json=payload, timeout=5)
        except Exception as e:
            logger.warning("Failed to send Discord webhook: %s", e)

    def _send_telegram(self, message: str) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.warning("Failed to send Telegram message: %s", e)
