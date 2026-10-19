"""Bot Main CLI Entrypoint."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from bot.config import BOT_CONFIG
from bot.brokers import get_broker
from bot.runner import BotRunner

app = typer.Typer(help="TradingAgents Automated Trading Bot")
console = Console()


@app.command()
def trade(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Ticker or pair symbol (e.g. NVDA, AAPL, BTC-USD)"),
    date: str | None = typer.Option(None, "--date", "-d", help="Analysis date (YYYY-MM-DD), default is current date"),
    broker: str = typer.Option(BOT_CONFIG.broker_type, "--broker", "-b", help="Broker to use: 'paper' or 'alpaca'"),
):
    """Run full analysis, risk evaluation, and execute trade for a single symbol."""
    console.print(f"[bold green]Starting trade cycle for {symbol} using {broker} broker...[/bold green]")
    runner = BotRunner(broker=get_broker(broker))
    order = runner.run_symbol(symbol, trade_date=date)
    if order:
        console.print(f"[bold green]Order completed: {order.side} {order.quantity} {order.symbol} at status {order.status}[/bold green]")
    else:
        console.print(f"[yellow]No order executed for {symbol} (rejected by risk, hold, or error).[/yellow]")


@app.command()
def scan(
    date: str | None = typer.Option(None, "--date", "-d", help="Analysis date (YYYY-MM-DD)"),
    broker: str = typer.Option(BOT_CONFIG.broker_type, "--broker", "-b", help="Broker to use: 'paper' or 'alpaca'"),
):
    """Scan and execute trades across the entire watchlist (Equities + Crypto)."""
    console.print(f"[bold cyan]Scanning portfolio watchlist using {broker} broker...[/bold cyan]")
    runner = BotRunner(broker=get_broker(broker))
    orders = runner.run_portfolio_cycle(trade_date=date)
    console.print(f"[bold green]Scan finished. Total executed orders: {len(orders)}[/bold green]")


@app.command()
def balance(
    broker: str = typer.Option(BOT_CONFIG.broker_type, "--broker", "-b", help="Broker to inspect: 'paper' or 'alpaca'"),
):
    """Display account balance and currently open positions."""
    b = get_broker(broker)
    acc = b.get_account_balance()
    positions = b.get_positions()

    table = Table(title=f"💼 Estado de Cuenta ({broker.upper()})")
    table.add_column("Concepto", style="cyan")
    table.add_column("Monto / Valor", style="bold green")

    table.add_row("Cash Disponible", f"${acc.cash:,.2f} {acc.currency}")
    table.add_row("Equity Total", f"${acc.equity:,.2f} {acc.currency}")
    table.add_row("Buying Power", f"${acc.buying_power:,.2f} {acc.currency}")
    table.add_row("Drawdown Diario", f"{acc.daily_drawdown_pct:.2f}%")
    console.print(table)

    if positions:
        pos_table = Table(title="📊 Posiciones Abiertas")
        pos_table.add_column("Símbolo", style="cyan")
        pos_table.add_column("Tipo", style="magenta")
        pos_table.add_column("Cantidad", style="yellow")
        pos_table.add_column("Precio Entrada", style="white")
        pos_table.add_column("Precio Actual", style="white")
        pos_table.add_column("PnL No Realizado", style="bold green")

        for p in positions:
            pnl_color = "green" if p.unrealized_pnl >= 0 else "red"
            pos_table.add_row(
                p.symbol,
                p.asset_class.value,
                f"{p.quantity:.4f}",
                f"${p.entry_price:,.2f}",
                f"${p.current_price:,.2f}",
                f"[{pnl_color}]${p.unrealized_pnl:,.2f} ({p.unrealized_pnl_pct:.2f}%)[/{pnl_color}]",
            )
        console.print(pos_table)
    else:
        console.print("[dim]No hay posiciones abiertas actualmente.[/dim]")


if __name__ == "__main__":
    app()
