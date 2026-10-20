"""Portfolio Intelligence & Capital Health Layer.

Calculates continuous institutional performance metrics:
1. Rolling Sharpe Ratio, Sortino Ratio, Calmar Ratio, and Mathematical Expectancy.
2. Per-asset track record tracking for dynamic capital allocation weighting.
3. Automated degradation detection & Defensive Risk Circuit Breakers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AssetPerformance:
    symbol: str
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    total_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    allocation_multiplier: float = 1.0


@dataclass
class PortfolioHealthReport:
    total_trades_analyzed: int
    rolling_win_rate: float
    rolling_sharpe: float
    rolling_sortino: float
    max_drawdown_pct: float
    profit_factor: float
    mathematical_expectancy: float
    health_status: str  # "OPTIMAL", "STABLE", "DEGRADED", "HALT_RECOMMENDED"
    sizing_multiplier: float  # 1.0 = normal, 0.5 = defensive cut, 0.0 = halt
    warnings: list[str] = field(default_factory=list)
    asset_breakdown: dict[str, AssetPerformance] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PortfolioIntelligence:
    """Prefrontal cortex for portfolio health monitoring and dynamic asset allocation."""

    def __init__(self, lookback_trades: int = 50):
        self.lookback_trades = lookback_trades
        self._cached_report: PortfolioHealthReport | None = None
        self._last_analysis_ts: float = 0.0

    def compute_metrics(
        self,
        historical_records: list[dict[str, Any]],
    ) -> PortfolioHealthReport:
        """Compute rolling risk-adjusted metrics across trades and per symbol."""
        if not historical_records:
            return PortfolioHealthReport(
                total_trades_analyzed=0,
                rolling_win_rate=50.0,
                rolling_sharpe=0.0,
                rolling_sortino=0.0,
                max_drawdown_pct=0.0,
                profit_factor=1.0,
                mathematical_expectancy=0.0,
                health_status="STABLE",
                sizing_multiplier=1.0,
                warnings=["Sin historial suficiente para análisis estadístico."],
            )

        recent = historical_records[-self.lookback_trades:]
        pnl_series = [float(r.get("pnl_pct", 0.0)) for r in recent]
        n_trades = len(pnl_series)

        # 1. Win Rate & Profit Factor
        wins = [p for p in pnl_series if p > 0]
        losses = [p for p in pnl_series if p <= 0]
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = (n_wins / n_trades * 100.0) if n_trades > 0 else 50.0

        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = abs(float(np.mean(losses))) if losses else 0.0

        # Mathematical Expectancy: (Win% * AvgWin) - (Loss% * AvgLoss)
        win_prob = n_wins / n_trades if n_trades > 0 else 0.5
        loss_prob = n_losses / n_trades if n_trades > 0 else 0.5
        expectancy = round((win_prob * avg_win) - (loss_prob * avg_loss), 4)

        gross_profits = sum(wins)
        gross_losses = abs(sum(losses))
        profit_factor = round(gross_profits / max(gross_losses, 0.001), 2) if gross_losses > 0 else round(gross_profits, 2)

        # 2. Rolling Sharpe Ratio
        mean_pnl = float(np.mean(pnl_series))
        std_pnl = float(np.std(pnl_series)) if len(pnl_series) > 1 else 1.0
        sharpe = round(mean_pnl / max(std_pnl, 0.001) * math.sqrt(n_trades), 2) if std_pnl > 0 else 0.0

        # 3. Sortino Ratio (downside volatility only)
        downside_returns = [p for p in pnl_series if p < 0]
        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 1 else max(std_pnl, 0.001)
        sortino = round(mean_pnl / max(downside_std, 0.001) * math.sqrt(n_trades), 2) if downside_std > 0 else 0.0

        # 4. Max Drawdown % of cumulative PnL curve
        cum_pnl = np.cumsum(pnl_series)
        peak = np.maximum.accumulate(cum_pnl)
        drawdown = peak - cum_pnl
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # 5. Asset breakdown & Dynamic allocation weights
        asset_map: dict[str, list[float]] = {}
        for r in recent:
            sym = str(r.get("symbol", "UNKNOWN"))
            asset_map.setdefault(sym, []).append(float(r.get("pnl_pct", 0.0)))

        breakdown: dict[str, AssetPerformance] = {}
        for sym, ret_list in asset_map.items():
            sym_wins = [x for x in ret_list if x > 0]
            sym_losses = [x for x in ret_list if x <= 0]
            s_winrate = (len(sym_wins) / len(ret_list) * 100.0)
            s_avg_win = float(np.mean(sym_wins)) if sym_wins else 0.0
            s_avg_loss = abs(float(np.mean(sym_losses))) if sym_losses else 0.0
            s_exp = round(((len(sym_wins) / len(ret_list)) * s_avg_win) - ((len(sym_losses) / len(ret_list)) * s_avg_loss), 4)

            s_mean = float(np.mean(ret_list))
            s_std = float(np.std(ret_list)) if len(ret_list) > 1 else 1.0
            s_sharpe = round(s_mean / max(s_std, 0.001) * math.sqrt(len(ret_list)), 2)

            # Dynamic allocation multiplier: [0.5, 1.35]
            if s_exp > 0.5 and s_sharpe > 0.5:
                alloc_mult = 1.25  # High edge: boost position size
            elif s_exp > 1.0 and s_sharpe > 1.0:
                alloc_mult = 1.35  # Alpha powerhouse
            elif s_exp < -0.3 or s_sharpe < -0.5:
                alloc_mult = 0.50  # Bleeding asset: cut size in half
            else:
                alloc_mult = 1.00  # Baseline

            breakdown[sym] = AssetPerformance(
                symbol=sym,
                trade_count=len(ret_list),
                win_count=len(sym_wins),
                loss_count=len(sym_losses),
                total_pnl_pct=round(sum(ret_list), 2),
                avg_win_pct=round(s_avg_win, 2),
                avg_loss_pct=round(s_avg_loss, 2),
                win_rate=round(s_winrate, 1),
                profit_factor=round(sum(sym_wins) / max(abs(sum(sym_losses)), 0.001), 2),
                expectancy=s_exp,
                sharpe_ratio=s_sharpe,
                allocation_multiplier=alloc_mult,
            )

        # 6. Global Health Status & Circuit Breaker determination
        warnings: list[str] = []
        status = "STABLE"
        sizing_mult = 1.0

        if n_trades >= 15:
            if sharpe < -0.80 or (win_rate < 30.0 and profit_factor < 0.6):
                status = "HALT_RECOMMENDED"
                sizing_mult = 0.0
                warnings.append(f"🚨 ALERTA CRÍTICA: Rolling Sharpe severamente negativo ({sharpe:.2f}) y WinRate={win_rate:.0f}%. Circuit breaker activado.")
            elif sharpe < -0.20 or win_rate < 40.0:
                status = "DEGRADED"
                sizing_mult = 0.50
                warnings.append(f"⚠️ Degradación de rendimiento detectada: Sharpe={sharpe:.2f}, WinRate={win_rate:.0f}%. Sizing reducido al 50%.")
            elif sharpe >= 1.2 and profit_factor >= 2.0:
                status = "OPTIMAL"
                sizing_mult = 1.15
            else:
                status = "STABLE"
                sizing_mult = 1.0

        report = PortfolioHealthReport(
            total_trades_analyzed=n_trades,
            rolling_win_rate=round(win_rate, 1),
            rolling_sharpe=sharpe,
            rolling_sortino=sortino,
            max_drawdown_pct=round(max_dd, 2),
            profit_factor=profit_factor,
            mathematical_expectancy=expectancy,
            health_status=status,
            sizing_multiplier=sizing_mult,
            warnings=warnings,
            asset_breakdown=breakdown,
        )
        self._cached_report = report
        return report

    def get_symbol_allocation_multiplier(self, symbol: str) -> float:
        """Return dynamic sizing multiplier [0.5, 1.35] for an asset based on historical track record."""
        if not self._cached_report or not self._cached_report.asset_breakdown:
            return 1.0
        perf = self._cached_report.asset_breakdown.get(symbol)
        if not perf:
            return 1.0
        return perf.allocation_multiplier

    def get_allocation_multiplier(self, symbol: str) -> float:
        """Alias for get_symbol_allocation_multiplier."""
        return self.get_symbol_allocation_multiplier(symbol)

    def should_halt_trading(self) -> bool:
        """Check if trading should be halted due to severe portfolio degradation."""
        if self._cached_report and self._cached_report.health_status == "HALT_RECOMMENDED":
            return True
        return False


PORTFOLIO_INTELLIGENCE = PortfolioIntelligence()

