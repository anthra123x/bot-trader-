"""Bayesian Hyperparameter Optimizer for BotRiskConfig (The Hypothalamus).

Uses Optuna with Tree-structured Parzen Estimator (TPE) sampler to dynamically
calibrate critical trading and risk parameters based on historical trade outcomes.

Tunable Category-A Parameters:
- default_stop_loss_pct: [1.5, 4.5]
- min_risk_reward_ratio: [1.8, 3.5]
- partial_take_profit_threshold_pct: [1.2, 3.5]
- max_overbought_rsi_buy: [60.0, 70.0]
- trailing_stop_distance_pct: [0.6, 2.2]
- screener_conviction_threshold: [48.0, 68.0]
- min_meta_probability: [0.52, 0.75]
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from bot.config import BotRiskConfig

logger = logging.getLogger(__name__)


class ParamOptimizer:
    """Bayesian parameter tuning engine optimizing risk and execution thresholds."""

    def __init__(
        self,
        min_samples_to_optimize: int = 20,
        cache_path: str = "data/optimized_params.json",
    ):
        self.min_samples_to_optimize = min_samples_to_optimize
        self.cache_path = cache_path
        self._cached_best_params: dict[str, float] = {}
        self._last_optimization_ts: float = 0.0

        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.load_cached_params()

    def load_cached_params(self) -> dict[str, float]:
        """Load optimized parameters from local disk or Neon DB system_config."""
        # 1. Local disk
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._cached_best_params = data.get("best_params", {})
                    return self._cached_best_params
            except Exception as e:
                logger.debug("[PARAM_OPTIMIZER] Error cargando parámetros locales: %s", e)

        # 2. Neon DB
        try:
            from bot.database.neon_repo import get_neon_repo
            repo = get_neon_repo()
            if repo.is_connected:
                raw_json = repo.get_system_config("OPTIMIZED_RISK_PARAMS")
                if raw_json:
                    data = json.loads(raw_json)
                    self._cached_best_params = data.get("best_params", {})
                    return self._cached_best_params
        except Exception as ne:
            logger.debug("[PARAM_OPTIMIZER] Error cargando parámetros desde Neon: %s", ne)

        return {}

    def save_cached_params(self, best_params: dict[str, float], best_score: float) -> None:
        """Persist optimized parameters to disk and Neon DB."""
        self._cached_best_params = best_params
        payload = {
            "best_params": best_params,
            "best_score": round(best_score, 4),
            "optimized_at": os.getenv("RENDER_GIT_COMMIT", "local"),
        }

        # 1. Local disk
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as fe:
            logger.debug("[PARAM_OPTIMIZER] Error guardando archivo local: %s", fe)

        # 2. Neon DB
        try:
            from bot.database.neon_repo import get_neon_repo
            repo = get_neon_repo()
            if repo.is_connected:
                repo.set_system_config("OPTIMIZED_RISK_PARAMS", json.dumps(payload))
                logger.info("[PARAM_OPTIMIZER] 💾 Parámetros optimizados persistidos en Neon PostgreSQL.")
        except Exception as ne:
            logger.debug("[PARAM_OPTIMIZER] Error guardando en Neon: %s", ne)

    def optimize_parameters(
        self,
        historical_records: list[dict[str, Any]],
        n_trials: int = 40,
    ) -> dict[str, float]:
        """Run Bayesian optimization over historical trades to discover optimal risk parameters."""
        if not OPTUNA_AVAILABLE:
            logger.debug("[PARAM_OPTIMIZER] Optuna no está disponible en el entorno.")
            return self._cached_best_params

        if len(historical_records) < self.min_samples_to_optimize:
            logger.info(
                "[PARAM_OPTIMIZER] ℹ️ Historial insuficiente (%d/%d requeridas) para optimización bayesiana.",
                len(historical_records), self.min_samples_to_optimize
            )
            return self._cached_best_params

        trades = historical_records[-80:]  # Focus on rolling recent window

        def objective(trial: optuna.Trial) -> float:
            sl_pct = trial.suggest_float("default_stop_loss_pct", 1.5, 4.5, step=0.1)
            rr_ratio = trial.suggest_float("min_risk_reward_ratio", 1.8, 3.5, step=0.1)
            partial_tp = trial.suggest_float("partial_take_profit_threshold_pct", 1.2, 3.5, step=0.1)
            rsi_max = trial.suggest_float("max_overbought_rsi_buy", 60.0, 70.0, step=1.0)
            trail_dist = trial.suggest_float("trailing_stop_distance_pct", 0.6, 2.2, step=0.1)
            screener_thresh = trial.suggest_float("screener_conviction_threshold", 48.0, 68.0, step=1.0)
            meta_prob = trial.suggest_float("min_meta_probability", 0.52, 0.76, step=0.02)

            simulated_pnls: list[float] = []

            for r in trades:
                rsi = float(r.get("rsi_entry", 50.0))
                mae = float(r.get("mae_pct", 0.0))
                mfe = float(r.get("mfe_pct", 0.0))
                actual_pnl = float(r.get("pnl_pct", 0.0))
                feat = r.get("features") or {}
                screener_score = float(feat.get("screener_score", 55.0))

                # 1. Entry Filter Simulation
                if rsi > rsi_max or screener_score < screener_thresh:
                    # Trade would have been filtered out (0 return)
                    continue

                # 2. Exit Simulation based on MAE and MFE
                # Target TP distance
                target_tp_pct = sl_pct * rr_ratio

                if mae >= sl_pct:
                    # Trade stopped out at candidate SL
                    sim_pnl = -sl_pct
                elif mfe >= target_tp_pct:
                    # Trade reached target TP
                    sim_pnl = target_tp_pct
                elif mfe >= partial_tp:
                    # Partial TP harvested + runner trailed
                    harvested = partial_tp * 0.35
                    runner_rem = max(0.0, mfe - trail_dist) * 0.65
                    sim_pnl = harvested + runner_rem
                else:
                    # Standard actual outcome bounded
                    sim_pnl = max(-sl_pct, min(mfe, actual_pnl))

                simulated_pnls.append(sim_pnl)

            if len(simulated_pnls) < 10:
                return -10.0

            # Compute objective metrics: Sharpe + Profit Factor
            wins = [x for x in simulated_pnls if x > 0]
            losses = [x for x in simulated_pnls if x < 0]
            gross_win = sum(wins)
            gross_loss = abs(sum(losses))
            pf = (gross_win / max(gross_loss, 0.001)) if gross_loss > 0 else 1.0

            mean_ret = float(np.mean(simulated_pnls))
            std_ret = float(np.std(simulated_pnls)) if len(simulated_pnls) > 1 else 1.0
            sharpe = (mean_ret / max(std_ret, 0.001)) * math.sqrt(len(simulated_pnls)) if std_ret > 0 else 0.0

            # Objective combines Sharpe, Profit Factor, and net return
            score = (sharpe * 1.5) + (min(pf, 3.5) * 1.0) + (mean_ret * 0.2)
            return float(score)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=10.0)

        best_params = {k: round(v, 2) for k, v in study.best_params.items()}
        best_score = float(study.best_value)

        self.save_cached_params(best_params, best_score)
        logger.info(
            "[PARAM_OPTIMIZER] 🎯 Optimización Bayesiana concluida (Score: %.2f). Parámetros óptimos: %s",
            best_score, best_params
        )
        return best_params

    def apply_to_config(self, config: BotRiskConfig, params: dict[str, float] | None = None) -> int:
        """Dynamically apply optimized parameters to an active BotRiskConfig instance."""
        target_params = params or self._cached_best_params
        if not target_params:
            return 0

        applied_count = 0
        for key, val in target_params.items():
            if hasattr(config, key):
                current_val = getattr(config, key)
                setattr(config, key, val)
                applied_count += 1
                logger.debug(
                    "[PARAM_OPTIMIZER] Parámetro actualizado: %s = %.2f (Anterior: %s)",
                    key, val, current_val
                )

        if applied_count > 0:
            logger.info(
                "[PARAM_OPTIMIZER] ⚙️ %d parámetros de riesgo auto-calibrados aplicados dinámicamente al motor.",
                applied_count
            )
        return applied_count


PARAM_OPTIMIZER = ParamOptimizer()
