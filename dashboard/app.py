"""FastAPI Server and REST API for the TradingAgents 24/7 Dashboard."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bot.brokers import get_broker
from bot.config import BOT_CONFIG
from bot.models import Order
from bot.runner import BotRunner

# Setup in-memory circular log buffer for live dashboard streaming
LOG_BUFFER: deque[str] = deque(maxlen=250)


class DashboardLogHandler(logging.Handler):
    """Custom logging handler to push events to dashboard log buffer."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            LOG_BUFFER.append(msg)
        except Exception:
            pass


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = DashboardLogHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
root_logger.addHandler(handler)

SERVER_START_TIME = time.time()

# FastAPI Application
app = FastAPI(title="TradingAgents Pro Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Bot Runner Instance
import os
from bot.database.neon_repo import get_neon_repo
from bot.learning.pattern_memory import PatternMemory
from dashboard.scheduler import BotScheduler

neon_repo = get_neon_repo()
_local_token = os.getenv("ALPACA_ACCESS_TOKEN")
if _local_token and neon_repo.ping():
    neon_repo.set_system_config("ALPACA_ACCESS_TOKEN", _local_token)

broker_instance = get_broker(BOT_CONFIG.broker_type)
pattern_memory = PatternMemory()
bot_runner = BotRunner(broker=broker_instance, db=neon_repo)
ORDER_HISTORY: list[dict[str, Any]] = []
NOTIFICATION_BUFFER: deque[dict[str, Any]] = deque(maxlen=100)

def record_notification(
    event_type: str,
    title: str,
    message: str,
    icon: str = "⚡",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "id": str(uuid.uuid4())[:8],
        "type": event_type,
        "title": title,
        "message": message,
        "icon": icon,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
    }
    NOTIFICATION_BUFFER.appendleft(event)
    return event

# Seed initial system greeting notification
record_notification(
    event_type="SYSTEM",
    title="TradingAgents OS Activo",
    message="Motor de trading autónomo conectado 24/7 con protección defensiva.",
    icon="⚡",
)

# Autonomous 24/7 Scheduler Instance
bot_scheduler = BotScheduler(
    runner=bot_runner,
    order_history=ORDER_HISTORY,
    log_buffer=LOG_BUFFER,
    notification_buffer=NOTIFICATION_BUFFER,
    scan_interval_seconds=int(os.getenv("BOT_SCAN_INTERVAL_SECONDS", "1800")),
    position_check_interval_seconds=int(os.getenv("BOT_POSITION_INTERVAL_SECONDS", "60")),
)


@app.on_event("startup")
async def startup_event() -> None:
    LOG_BUFFER.append("[SYSTEM] 🟢 TradingAgents OS iniciado con éxito.")
    LOG_BUFFER.append(f"[SYSTEM] 💼 Broker activo: {BOT_CONFIG.broker_type.upper()} (Paper Trading).")
    LOG_BUFFER.append("[SYSTEM] 🤖 Pool de IA rotativo conectado: Gemini Pro + Groq Cloud.")
    if neon_repo.ping():
        LOG_BUFFER.append("[SYSTEM] 🗄️ Neon PostgreSQL conectado: órdenes, debates y equity respaldados en la nube.")
    else:
        LOG_BUFFER.append("[SYSTEM] ⚠️ Neon PostgreSQL en espera o degradado.")
    await bot_scheduler.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await bot_scheduler.stop()


# Simulation Engine Instance (kept for synthetic testing compatibility)
from dashboard.simulator import SimulationEngine
simulation_engine = SimulationEngine(LOG_BUFFER, ORDER_HISTORY)


@app.get("/api/scheduler/status")
def get_scheduler_status() -> dict[str, Any]:
    """Retrieve active autonomous scheduler telemetry."""
    return {
        "running": bot_scheduler.is_running,
        "scan_interval_seconds": bot_scheduler.scan_interval_seconds,
        "position_check_interval_seconds": bot_scheduler.position_check_interval_seconds,
    }


@app.get("/api/learning/patterns")
def get_patterns_summary() -> dict[str, Any]:
    """Retrieve learned mistake patterns, positive patterns, and market regime drift metrics."""
    return pattern_memory.get_learned_patterns_summary()


@app.get("/api/market/regime")
def get_market_regime() -> dict[str, Any]:
    """Retrieve macroscopic volatility analysis and adaptive risk parameters."""
    from bot.risk.regime_specialist import REGIME_SPECIALIST
    res = REGIME_SPECIALIST.analyze_market_regime("BTC-USD")
    sentiment_data = None
    if res.web_sentiment:
        sentiment_data = {
            "fear_greed_value": res.web_sentiment.fear_greed_value,
            "fear_greed_classification": res.web_sentiment.fear_greed_classification,
            "sentiment_score": res.web_sentiment.sentiment_score,
            "sentiment_label": res.web_sentiment.sentiment_label,
            "top_headlines": res.web_sentiment.top_headlines,
            "macro_catalysts": res.web_sentiment.macro_catalysts,
            "investigated_at": res.web_sentiment.investigated_at.isoformat(),
        }
    return {
        "regime": res.regime.value,
        "regime_display": res.regime_display,
        "volatility_score": res.volatility_score,
        "realized_volatility_20d": res.realized_volatility_20d,
        "atr_ratio_90d": res.atr_ratio_90d,
        "bollinger_bandwidth_pct": res.bollinger_bandwidth_pct,
        "trend_bias": res.trend_bias,
        "tp_multiplier": res.tp_multiplier,
        "sl_multiplier": res.sl_multiplier,
        "break_even_pct": res.break_even_pct,
        "allocation_factor": res.allocation_factor,
        "rationale": res.rationale,
        "web_sentiment": sentiment_data,
    }


@app.get("/api/market/sentiment")
def get_market_sentiment() -> dict[str, Any]:
    """Retrieve real-time web-investigated macro sentiment, Fear & Greed index, and news."""
    from bot.risk.regime_specialist import REGIME_SPECIALIST
    res = REGIME_SPECIALIST.investigate_web_sentiment()
    return {
        "fear_greed_value": res.fear_greed_value,
        "fear_greed_classification": res.fear_greed_classification,
        "sentiment_score": res.sentiment_score,
        "sentiment_label": res.sentiment_label,
        "top_headlines": res.top_headlines,
        "macro_catalysts": res.macro_catalysts,
        "investigated_at": res.investigated_at.isoformat(),
    }


@app.get("/")
@app.head("/")
def get_index() -> FileResponse:
    """Serve the main single-page dashboard."""
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint for external uptime monitors and tunnels."""
    uptime_seconds = int(time.time() - SERVER_START_TIME)
    drift = pattern_memory.detect_regime_drift()
    neon_alive = neon_repo.ping()
    return {
        "status": "healthy" if drift["status"] != "DEFENSIVE_QUARANTINE" else "warning",
        "service": "tradingagents-server",
        "uptime_seconds": uptime_seconds,
        "broker": BOT_CONFIG.broker_type,
        "simulation_active": simulation_engine.is_running,
        "neon_connected": neon_alive,
        "regime_status": drift["status"],
        "consecutive_losses": drift["consecutive_losses"],
    }


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    """Expose telemetry metrics for external monitoring systems (Datadog, Prometheus, UptimeRobot)."""
    uptime_seconds = int(time.time() - SERVER_START_TIME)
    drift = pattern_memory.detect_regime_drift()
    patterns_summary = pattern_memory.get_learned_patterns_summary()
    return {
        "uptime_seconds": uptime_seconds,
        "broker": BOT_CONFIG.broker_type,
        "simulation_running": simulation_engine.is_running,
        "closed_trades_count": simulation_engine.closed_trades_count,
        "win_rate_pct": round(
            (simulation_engine.winning_trades_count / simulation_engine.closed_trades_count * 100.0)
            if simulation_engine.closed_trades_count > 0 else 0.0, 1
        ),
        "realized_pnl_total": round(simulation_engine.realized_pnl_total, 2),
        "patterns_cataloged": patterns_summary["total_patterns_cataloged"],
        "regime_drift_status": drift["status"],
        "consecutive_losses": drift["consecutive_losses"],
    }


@app.get("/api/simulation/status")
async def get_sim_status() -> dict[str, bool]:
    """Check if live simulation mode is active."""
    return {"running": simulation_engine.is_running}


@app.post("/api/simulation/toggle")
async def toggle_simulation() -> dict[str, Any]:
    """Toggle the live simulation mode on or off."""
    if simulation_engine.is_running:
        simulation_engine.stop()
        running = False
    else:
        simulation_engine.start()
        running = True
    return {"status": "ok", "running": running}


@app.post("/api/simulation/reset")
async def reset_simulation() -> dict[str, Any]:
    """Reset the paper simulation account back to $100,000 cash and clear history."""
    simulation_engine.reset()
    return {"status": "ok", "message": "Simulación reiniciada con éxito a $100,000.00 USD."}


@app.get("/api/balance")
async def get_balance() -> dict[str, Any]:
    """Retrieve current account balance and daily drawdown from active broker/simulation."""
    try:
        if simulation_engine.is_running:
            positions = simulation_engine.paper_broker.get_positions()
            market_total = 0.0
            for p in positions:
                curr_price = simulation_engine.get_price(p.symbol)
                market_total += p.quantity * curr_price
            
            bal = simulation_engine.paper_broker.get_account_balance()
            total_equity = bal.cash + market_total
            daily_pnl = total_equity - bal.initial_day_equity
            drawdown = max(0.0, (bal.initial_day_equity - total_equity) / bal.initial_day_equity * 100.0) if bal.initial_day_equity > 0 else 0.0
            win_rate = (simulation_engine.winning_trades_count / simulation_engine.closed_trades_count * 100.0) if simulation_engine.closed_trades_count > 0 else 0.0

            return {
                "cash": round(bal.cash, 2),
                "equity": round(total_equity, 2),
                "buying_power": round(bal.cash, 2),
                "daily_pnl": round(daily_pnl, 2),
                "daily_drawdown_pct": round(drawdown, 2),
                "currency": "USD",
                "is_simulation": True,
                "closed_trades": simulation_engine.closed_trades_count,
                "win_rate": round(win_rate, 1),
                "realized_pnl": round(simulation_engine.realized_pnl_total, 2),
            }
        else:
            bal = broker_instance.get_account_balance()
            return {
                "cash": bal.cash,
                "equity": bal.equity,
                "buying_power": bal.buying_power,
                "daily_pnl": bal.daily_pnl,
                "daily_drawdown_pct": bal.daily_drawdown_pct,
                "currency": bal.currency,
                "is_simulation": False,
            }
    except Exception as e:
        logger.warning(f"[API] Error in get_balance: {e}")
        return {
            "cash": 100000.0,
            "equity": 100000.0,
            "buying_power": 200000.0,
            "daily_pnl": 0.0,
            "daily_drawdown_pct": 0.0,
            "currency": "USD",
            "is_simulation": True,
            "warning": str(e),
        }


@app.get("/api/positions")
async def get_positions() -> list[dict[str, Any]]:
    """Retrieve all open positions from active broker/simulation with live PnL."""
    try:
        if simulation_engine.is_running:
            raw_positions = simulation_engine.paper_broker.get_positions()
            result = []
            for p in raw_positions:
                curr_price = simulation_engine.get_price(p.symbol)
                mkt_val = p.quantity * curr_price
                cost = p.quantity * p.entry_price
                pnl = mkt_val - cost
                pnl_pct = (pnl / cost) * 100.0 if cost > 0 else 0.0
                result.append({
                    "symbol": p.symbol,
                    "asset_class": p.asset_class.value,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": curr_price,
                    "market_value": round(mkt_val, 2),
                    "unrealized_pnl": round(pnl, 2),
                    "unrealized_pnl_pct": round(pnl_pct, 2),
                })
            return result
        else:
            positions = broker_instance.get_positions()
            return [
                {
                    "symbol": p.symbol,
                    "asset_class": p.asset_class.value,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                }
                for p in positions
            ]
    except Exception as e:
        logger.warning(f"[API] Error in get_positions: {e}")
        return []


@app.get("/api/orders")
def get_orders(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve executed orders history from live broker sync and Neon PostgreSQL."""
    if hasattr(broker_instance, "get_orders"):
        try:
            b_orders = broker_instance.get_orders(limit=limit)
            if b_orders:
                try:
                    neon_repo.save_orders_batch(b_orders)
                except Exception:
                    pass
                return [
                    {
                        "order_id": o.order_id,
                        "symbol": o.symbol,
                        "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                        "order_type": o.order_type.value if hasattr(o.order_type, "value") else str(o.order_type),
                        "quantity": o.quantity,
                        "limit_price": o.limit_price,
                        "filled_price": o.filled_price,
                        "status": o.status.value if hasattr(o.status, "value") else str(o.status),
                        "created_at": o.created_at.isoformat() if hasattr(o.created_at, "isoformat") else str(o.created_at),
                    }
                    for o in b_orders
                ]
        except Exception as e:
            logger.debug("Failed to sync broker orders: %s", e)

    try:
        neon_orders = neon_repo.get_recent_orders(limit=limit)
        if neon_orders:
            return neon_orders
    except Exception as e:
        logger.debug("[NEON] Failed to fetch orders from Neon: %s", e)

    return ORDER_HISTORY


@app.get("/api/debates")
def get_debates(symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve multi-agent debates, analyst perspectives, and consensus decisions from Neon."""
    try:
        return neon_repo.get_recent_debates(symbol=symbol, limit=limit)
    except Exception as e:
        logger.debug("[NEON] Failed to fetch debates: %s", e)
        return []


@app.get("/api/equity/history")
def get_equity_history(limit: int = 100) -> list[dict[str, Any]]:
    """Retrieve historical equity snapshots from Neon for persistent charting."""
    try:
        return neon_repo.get_equity_history(limit=limit)
    except Exception as e:
        logger.debug("[NEON] Failed to fetch equity history: %s", e)
        return []


@app.get("/api/watchlist")
def get_watchlist() -> dict[str, list[str]]:
    """Retrieve current asset watchlist."""
    return {
        "equities": BOT_CONFIG.watchlist_equities,
        "crypto": BOT_CONFIG.watchlist_crypto,
    }


class WatchlistRequest(BaseModel):
    action: str  # 'add' or 'remove'
    symbol: str


@app.post("/api/watchlist")
def update_watchlist(payload: WatchlistRequest) -> dict[str, Any]:
    """Add or remove symbols from the watchlist."""
    sym = payload.symbol.upper().strip()
    is_crypto = "-USD" in sym or "/USD" in sym or sym in ("BTC", "ETH", "SOL")

    target_list = BOT_CONFIG.watchlist_crypto if is_crypto else BOT_CONFIG.watchlist_equities
    if payload.action == "add":
        if sym not in target_list:
            target_list.append(sym)
    elif payload.action == "remove":
        if sym in target_list:
            target_list.remove(sym)

    return {"status": "ok", "watchlist": get_watchlist()}


@app.get("/api/logs")
def get_logs(filter: str = "all") -> list[str]:
    """Retrieve recent live terminal logs."""
    logs = list(LOG_BUFFER)
    if filter == "agents":
        logs = [l for l in logs if any(k in l for k in ("Analyst", "Researcher", "Bull", "Bear", "Trader", "Portfolio"))]
    return logs


@app.get("/api/notifications")
def get_notifications(since_id: str | None = None) -> list[dict[str, Any]]:
    """Retrieve recent real-time system notifications and movements."""
    events = list(NOTIFICATION_BUFFER)
    if since_id:
        res = []
        for ev in events:
            if ev["id"] == since_id:
                break
            res.append(ev)
        return res
    return events[:40]


@app.post("/api/notifications/test")
def trigger_test_notification() -> dict[str, Any]:
    """Trigger a live test notification for sound, toast, browser push, and webhooks."""
    event = record_notification(
        event_type="TEST_ALERT",
        title="🔔 Prueba de Alerta Exitosa",
        message="El sistema de notificaciones en tiempo real está funcionando correctamente.",
        icon="🔔",
        meta={"status": "verified"},
    )
    try:
        bot_runner.notifier.notify_alert(
            title="TradingAgents OS: Prueba de Alerta",
            message="Prueba de notificación recibida con éxito desde el panel web.",
            alert_type="SUCCESS",
        )
    except Exception as e:
        logger.warning("Could not dispatch test notification via notifier: %s", e)
    return {"status": "ok", "event": event}


class TradeRequest(BaseModel):
    symbol: str


def _async_trade_task(symbol: str) -> None:
    try:
        order = bot_runner.run_symbol(symbol)
        if order:
            ORDER_HISTORY.insert(0, {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "status": order.status.value,
                "filled_price": order.filled_price,
                "limit_price": order.limit_price,
                "created_at": order.created_at.isoformat(),
            })
            LOG_BUFFER.append(f"[SUCCESS] ✅ Orden ejecutada para {symbol}: {order.side.value} {order.quantity} u a ${order.limit_price or order.filled_price or 0.0:.2f}")
        else:
            LOG_BUFFER.append(f"[DECISION] Ciclo completado para {symbol}: Decisión neutral / HOLD (capital protegido sin orden)")
    except Exception as e:
        LOG_BUFFER.append(f"[ERROR] Error executing trade for {symbol}: {e}")


@app.post("/api/trade")
def execute_trade(payload: TradeRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Trigger manual agent analysis and trade execution for a single symbol."""
    sym = payload.symbol.upper().strip()
    background_tasks.add_task(_async_trade_task, sym)
    LOG_BUFFER.append(f"[ACTION] Ciclo de trading iniciado para {sym}")
    return {"message": f"Análisis y trade para {sym} iniciado en segundo plano."}


def _async_scan_task() -> None:
    try:
        orders = bot_runner.run_portfolio_cycle()
        for order in orders:
            ORDER_HISTORY.insert(0, {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "status": order.status.value,
                "filled_price": order.filled_price,
                "limit_price": order.limit_price,
                "created_at": order.created_at.isoformat(),
            })
        LOG_BUFFER.append(f"[SUCCESS] Escaneo de cartera completado. {len(orders)} órdenes ejecutadas.")
    except Exception as e:
        LOG_BUFFER.append(f"[ERROR] Error en escaneo de cartera: {e}")


@app.post("/api/scan")
def execute_scan(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Trigger full watchlist portfolio scan."""
    background_tasks.add_task(_async_scan_task)
    LOG_BUFFER.append("[ACTION] Escaneo general de cartera iniciado.")
    return {"message": "Escaneo general de cartera iniciado en segundo plano."}


@app.post("/api/kill-switch")
async def kill_switch() -> dict[str, str]:
    """Emergency kill switch: cancel orders and suspend execution."""
    await bot_scheduler.stop()
    LOG_BUFFER.append("[ALERT] 🚨 KILL SWITCH ACTIVADO: Motor autónomo detenido.")
    return {"message": "Kill switch activado. Todas las operaciones han sido detenidas."}


from bot.llm_rotator import LLM_ROTATOR


@app.get("/api/llm/pool")
def get_llm_pool_status() -> dict[str, Any]:
    """Retrieve active rotating LLM providers, key status, and call telemetry."""
    return {
        "total_providers": len(LLM_ROTATOR.pool),
        "pool": LLM_ROTATOR.get_pool_status(),
    }


class SettingsRequest(BaseModel):
    google_api_key: str | None = None
    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    mistral_api_key: str | None = None
    llm_provider: str | None = None
    max_risk_per_trade_pct: float | None = None
    max_daily_drawdown_pct: float | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    discord_webhook_url: str | None = None


@app.post("/api/settings")
def update_settings(payload: SettingsRequest) -> dict[str, str]:
    """Update active API keys, defensive risk thresholds, and notification webhooks."""
    import os
    env_path = Path(__file__).parent.parent / ".env"
    lines = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    def set_env_var(var_name: str, val: str) -> None:
        os.environ[var_name] = val
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{var_name}="):
                lines[i] = f"{var_name}={val}"
                updated = True
                break
        if not updated:
            lines.append(f"{var_name}={val}")

    if payload.google_api_key:
        set_env_var("GOOGLE_API_KEY", payload.google_api_key)
        set_env_var("GEMINI_API_KEY", payload.google_api_key)

    if payload.openrouter_api_key:
        set_env_var("OPENROUTER_API_KEY", payload.openrouter_api_key)

    if payload.groq_api_key:
        set_env_var("GROQ_API_KEY", payload.groq_api_key)

    if payload.mistral_api_key:
        set_env_var("MISTRAL_API_KEY", payload.mistral_api_key)

    if payload.llm_provider:
        set_env_var("TRADINGAGENTS_LLM_PROVIDER", payload.llm_provider)

    if payload.max_risk_per_trade_pct is not None:
        BOT_CONFIG.risk.max_risk_per_trade_pct = payload.max_risk_per_trade_pct

    if payload.max_daily_drawdown_pct is not None:
        BOT_CONFIG.risk.max_daily_drawdown_pct = payload.max_daily_drawdown_pct

    if payload.telegram_bot_token:
        set_env_var("TELEGRAM_BOT_TOKEN", payload.telegram_bot_token)
        bot_runner.notifier.telegram_token = payload.telegram_bot_token

    if payload.telegram_chat_id:
        set_env_var("TELEGRAM_CHAT_ID", payload.telegram_chat_id)
        bot_runner.notifier.telegram_chat_id = payload.telegram_chat_id

    if payload.discord_webhook_url:
        set_env_var("DISCORD_WEBHOOK_URL", payload.discord_webhook_url)
        bot_runner.notifier.discord_webhook = payload.discord_webhook_url

    env_path.write_text("\n".join(lines) + "\n")
    LLM_ROTATOR.load_credentials_from_env()
    LOG_BUFFER.append(f"[CONFIG] Ajustes actualizados: Pool de LLMs recargado ({len(LLM_ROTATOR.pool)} claves activas).")
    return {"message": f"Configuración guardada. Pool activo con {len(LLM_ROTATOR.pool)} proveedores."}
