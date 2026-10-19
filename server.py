"""Server launcher for TradingAgents 24/7 Web Dashboard & Daemon."""

from __future__ import annotations

import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"🚀 Iniciando servidor TradingAgents en http://{host}:{port}")
    uvicorn.run("dashboard.app:app", host=host, port=port, reload=False)
