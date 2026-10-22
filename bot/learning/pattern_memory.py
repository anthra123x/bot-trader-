"""Pattern Memory & Continuous Learning Feedback Engine.

Inspired by Stefan Jansen's 'Machine Learning for Trading' (3rd Edition):
Implements an operational feedback loop where past mistakes (losses, Stop-Loss triggers,
regime whipsaws) are captured with their technical feature signatures, stored,
and used to veto or downscale prospective trades matching known loss patterns.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bot.models import AssetClass, OrderSide, PatternCheckResult, TradeOutcome

logger = logging.getLogger(__name__)


class PatternMemory:
    """Persistent post-mortem feedback and pattern recognition memory."""

    def __init__(self, db_path: str = "data/pattern_memory.db"):
        self.db_path = db_path
        self._mem_conn: sqlite3.Connection | None = None
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:")
                self._mem_conn.row_factory = sqlite3.Row
            return self._mem_conn
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_patterns (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    pnl_dollars REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    rsi_entry REAL NOT NULL,
                    ema_ratio REAL NOT NULL,
                    volatility_atr_pct REAL NOT NULL,
                    market_regime TEXT NOT NULL,
                    mistake_signature TEXT,
                    lesson_learned TEXT,
                    created_at TEXT NOT NULL,
                    mae_dollars REAL DEFAULT 0.0,
                    mae_pct REAL DEFAULT 0.0,
                    mfe_dollars REAL DEFAULT 0.0,
                    mfe_pct REAL DEFAULT 0.0,
                    reward_score REAL DEFAULT 0.0,
                    exit_reason TEXT DEFAULT 'MANUAL',
                    duration_hours REAL DEFAULT 0.0,
                    features_json TEXT
                )
            """)
            # Automatic schema migration for preexisting SQLite databases
            try:
                cursor = conn.execute("PRAGMA table_info(trade_patterns)")
                existing_cols = {row["name"] for row in cursor.fetchall()}
                new_columns = [
                    ("mae_dollars", "REAL DEFAULT 0.0"),
                    ("mae_pct", "REAL DEFAULT 0.0"),
                    ("mfe_dollars", "REAL DEFAULT 0.0"),
                    ("mfe_pct", "REAL DEFAULT 0.0"),
                    ("reward_score", "REAL DEFAULT 0.0"),
                    ("exit_reason", "TEXT DEFAULT 'MANUAL'"),
                    ("duration_hours", "REAL DEFAULT 0.0"),
                    ("features_json", "TEXT"),
                ]
                for col_name, col_def in new_columns:
                    if col_name not in existing_cols:
                        conn.execute(f"ALTER TABLE trade_patterns ADD COLUMN {col_name} {col_def}")
            except Exception as me:
                logger.debug("Notice during SQLite schema migration: %s", me)
            conn.commit()

    @staticmethod
    def diagnose_mistake(
        side: OrderSide,
        rsi: float,
        ema_ratio: float,
        atr_pct: float,
        pnl_pct: float,
    ) -> tuple[str, str]:
        """Automatically diagnose and classify a trade mistake signature."""
        if pnl_pct >= 0:
            return (
                "VALIDATED_MOMENTUM_EDGE",
                "Estructura técnica validada con volatilidad contenida y ratio R/R positivo."
            )

        # Loss diagnostics based on entry conditions
        if rsi >= 65.0:
            return (
                "BULL_TRAP_OVERBOUGHT",
                f"Entrada en sobrecompra extrema (RSI={rsi:.1f}). Agotamiento comprador y trampa alcista institucional."
            )
        elif ema_ratio <= 0.96:
            return (
                "COUNTER_TREND_FALLING_KNIFE",
                f"Compra contra-tendencia agresiva por debajo de la EMA-50 (ratio {ema_ratio:.2f}). Evitar atrapar cuchillos cayendo."
            )
        elif atr_pct >= 4.5:
            return (
                "HIGH_VOLATILITY_WHIPSAW",
                f"Volatilidad excesiva (ATR={atr_pct:.1f}%). El ruido del mercado expulsó la posición en el Stop-Loss antes del desarrollo."
            )
        elif rsi <= 38.0 and side == OrderSide.BUY:
            return (
                "PREMATURE_BOTTOM_FISHING",
                f"Intento de adivinar suelo sin confirmación de volumen ni divergencia alcista consolidada (RSI={rsi:.1f})."
            )
        else:
            return (
                "CHOPPY_RANGE_BREAKDOWN",
                "Falso rompimiento en zona de consolidación lateral sin seguimiento institucional."
            )

    def record_trade_outcome(self, outcome: TradeOutcome) -> None:
        """Store settled trade outcome and register its pattern signature."""
        if not outcome.mistake_signature or not outcome.lesson_learned:
            sig, lesson = self.diagnose_mistake(
                side=outcome.side,
                rsi=outcome.rsi_entry,
                ema_ratio=outcome.ema_ratio,
                atr_pct=outcome.volatility_atr_pct,
                pnl_pct=outcome.pnl_pct,
            )
            outcome.mistake_signature = sig
            outcome.lesson_learned = lesson

        features_dict = outcome.features.model_dump() if outcome.features else None
        features_json_str = json.dumps(features_dict) if features_dict else None

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trade_patterns (
                    trade_id, symbol, asset_class, side, entry_price, exit_price,
                    quantity, pnl_dollars, pnl_pct, outcome, rsi_entry, ema_ratio,
                    volatility_atr_pct, market_regime, mistake_signature,
                    lesson_learned, created_at, mae_dollars, mae_pct, mfe_dollars,
                    mfe_pct, reward_score, exit_reason, duration_hours, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.trade_id,
                outcome.symbol,
                outcome.asset_class.value if hasattr(outcome.asset_class, "value") else str(outcome.asset_class),
                outcome.side.value if hasattr(outcome.side, "value") else str(outcome.side),
                outcome.entry_price,
                outcome.exit_price,
                outcome.quantity,
                outcome.pnl_dollars,
                outcome.pnl_pct,
                outcome.outcome,
                outcome.rsi_entry,
                outcome.ema_ratio,
                outcome.volatility_atr_pct,
                outcome.market_regime,
                outcome.mistake_signature,
                outcome.lesson_learned,
                outcome.created_at.isoformat(),
                outcome.mae_dollars,
                outcome.mae_pct,
                outcome.mfe_dollars,
                outcome.mfe_pct,
                outcome.reward_score,
                outcome.exit_reason,
                outcome.duration_hours,
                features_json_str,
            ))
            conn.commit()

        # Dual-write pattern to Neon PostgreSQL
        try:
            from bot.database.neon_repo import get_neon_repo
            neon_repo = get_neon_repo()
            metrics = {
                "trade_id": outcome.trade_id,
                "asset_class": outcome.asset_class.value if hasattr(outcome.asset_class, "value") else str(outcome.asset_class),
                "entry_price": outcome.entry_price,
                "exit_price": outcome.exit_price,
                "quantity": outcome.quantity,
                "pnl_dollars": outcome.pnl_dollars,
                "pnl_pct": outcome.pnl_pct,
                "rsi_entry": outcome.rsi_entry,
                "ema_ratio": outcome.ema_ratio,
                "volatility_atr_pct": outcome.volatility_atr_pct,
                "mae_dollars": outcome.mae_dollars,
                "mae_pct": outcome.mae_pct,
                "mfe_dollars": outcome.mfe_dollars,
                "mfe_pct": outcome.mfe_pct,
                "reward_score": outcome.reward_score,
                "exit_reason": outcome.exit_reason,
                "duration_hours": outcome.duration_hours,
                "features": features_dict,
            }
            neon_repo.save_learned_pattern(
                regime=outcome.market_regime,
                pattern_type=outcome.outcome,
                symbol=outcome.symbol,
                outcome=outcome.outcome,
                mistake_signature=outcome.mistake_signature or "",
                lesson_learned=outcome.lesson_learned or "",
                metrics=metrics,
            )
        except Exception as e:
            logger.debug("[NEON] Could not sync pattern to Neon: %s", e)

        logger.info(
            "Registered trade pattern [%s] for %s: %s (PnL: %.2f%%)",
            outcome.outcome, outcome.symbol, outcome.mistake_signature, outcome.pnl_pct
        )

        # Online Evolution: Retrain AdaptiveModel, update Portfolio Intelligence & Bayesian Optimizer
        try:
            matrix = self.get_historical_feature_matrix(limit=500)
            if len(matrix) >= 10:
                from bot.learning.portfolio_intelligence import PORTFOLIO_INTELLIGENCE
                PORTFOLIO_INTELLIGENCE.compute_metrics(matrix)

            if len(matrix) >= 25:
                from bot.learning.adaptive_model import ADAPTIVE_MODEL
                ADAPTIVE_MODEL.train_on_history(matrix)

            if len(matrix) >= 20 and len(matrix) % 10 == 0:
                from bot.learning.param_optimizer import PARAM_OPTIMIZER
                PARAM_OPTIMIZER.optimize_parameters(matrix, n_trials=35)
        except Exception as ae:
            logger.debug("[EVOLUTION] Online loop notice: %s", ae)

    def evaluate_setup(
        self,
        symbol: str,
        asset_class: AssetClass,
        rsi: float,
        ema_ratio: float,
        atr_pct: float,
    ) -> PatternCheckResult:
        """Compare prospective trade setup against known loss patterns in the database.
        
        Returns PatternCheckResult with veto (multiplier=0.0) or sizing penalty (multiplier=0.5)
        if the setup matches a previously failed pattern above threshold similarity.
        """
        candidate_patterns = []
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT mistake_signature, lesson_learned, rsi_entry, ema_ratio, volatility_atr_pct, pnl_pct
                    FROM trade_patterns
                    WHERE outcome = 'LOSS'
                    ORDER BY created_at DESC
                    LIMIT 50
                """).fetchall()
                for r in rows:
                    candidate_patterns.append({
                        "mistake_signature": r["mistake_signature"],
                        "lesson_learned": r["lesson_learned"],
                        "rsi_entry": float(r["rsi_entry"]),
                        "ema_ratio": float(r["ema_ratio"]),
                        "volatility_atr_pct": float(r["volatility_atr_pct"]),
                        "pnl_pct": float(r["pnl_pct"]),
                    })
        except Exception as e:
            logger.debug("SQLite pattern read failed: %s", e)

        # Dual-check with Neon PostgreSQL cloud patterns (crucial for ephemeral cloud instances)
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                from bot.database.neon_repo import get_neon_repo
                neon_repo = get_neon_repo()
                if neon_repo and neon_repo.is_connected:
                    neon_rows = neon_repo.get_learned_patterns(limit=50)
                    for nr in neon_rows:
                        if nr.get("outcome") == "LOSS":
                            metrics = nr.get("metrics") or {}
                            candidate_patterns.append({
                                "mistake_signature": nr.get("mistake_signature"),
                                "lesson_learned": nr.get("lesson_learned"),
                                "rsi_entry": float(metrics.get("rsi_entry", 50.0)),
                                "ema_ratio": float(metrics.get("ema_ratio", 1.0)),
                                "volatility_atr_pct": float(metrics.get("volatility_atr_pct", 2.0)),
                                "pnl_pct": float(metrics.get("pnl_pct", -1.0)),
                            })
            except Exception as e:
                logger.debug("Failed to augment patterns from Neon: %s", e)

        if not candidate_patterns:
            return PatternCheckResult(is_safe=True, risk_multiplier=1.0)

        highest_similarity = 0.0
        worst_match = None

        for r in candidate_patterns:
            p_rsi = float(r["rsi_entry"])
            p_ema = float(r["ema_ratio"])
            p_atr = float(r["volatility_atr_pct"])

            # Weighted normalized Euclidean distance between current setup and past mistake
            d_rsi = (rsi - p_rsi) / 15.0
            d_ema = (ema_ratio - p_ema) / 0.05
            d_atr = (atr_pct - p_atr) / 1.5

            dist = math.sqrt((d_rsi ** 2) * 0.4 + (d_ema ** 2) * 0.35 + (d_atr ** 2) * 0.25)
            # Similarity scaled between 0.0 and 1.0
            sim = max(0.0, 1.0 - dist)

            if sim > highest_similarity:
                highest_similarity = sim
                worst_match = r

        if highest_similarity >= 0.82 and worst_match:
            sig = worst_match["mistake_signature"]
            lesson = worst_match["lesson_learned"]
            logger.warning(
                "Trade vetoed by Pattern Memory: %s setup matches '%s' (similarity: %.1f%%)",
                symbol, sig, highest_similarity * 100.0
            )
            return PatternCheckResult(
                is_safe=False,
                risk_multiplier=0.0,
                matched_mistake_tag=sig,
                similarity_score=round(highest_similarity * 100.0, 1),
                warning_message=f"Setup para {symbol} coincide en {highest_similarity * 100:.1f}% con patrón fallido: {sig}",
                learned_lesson=lesson,
            )

        elif highest_similarity >= 0.68 and worst_match:
            sig = worst_match["mistake_signature"]
            lesson = worst_match["lesson_learned"]
            logger.info(
                "Trade downscaled by Pattern Memory: %s setup has partial match to '%s' (similarity: %.1f%%)",
                symbol, sig, highest_similarity * 100.0
            )
            return PatternCheckResult(
                is_safe=True,
                risk_multiplier=0.5,
                matched_mistake_tag=sig,
                similarity_score=round(highest_similarity * 100.0, 1),
                warning_message=f"Precaución: Setup moderadamente similar ({highest_similarity * 100:.1f}%) a fallo previo: {sig}. Asignación reducida al 50%.",
                learned_lesson=lesson,
            )

        return PatternCheckResult(
            is_safe=True,
            risk_multiplier=1.0,
            similarity_score=round(highest_similarity * 100.0, 1),
        )

    def detect_regime_drift(self, window_size: int = 10) -> dict[str, Any]:
        """Detect Concept/Regime Drift (alpha decay) based on rolling execution window.
        
        If the rolling win rate degrades or a loss streak occurs, signals a defensive quarantine.
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT outcome, pnl_pct, mistake_signature
                FROM trade_patterns
                ORDER BY created_at DESC
                LIMIT ?
            """, (window_size,)).fetchall()

        if len(rows) < 3:
            return {
                "drift_detected": False,
                "status": "INSUFFICIENT_HISTORY",
                "sample_size": len(rows),
                "win_rate_pct": 100.0,
                "consecutive_losses": 0,
                "recommendation": "Continuar acumulando historial de ejecución.",
            }

        wins = sum(1 for r in rows if r["outcome"] == "WIN")
        losses = sum(1 for r in rows if r["outcome"] == "LOSS")
        win_rate = (wins / len(rows)) * 100.0

        # Calculate consecutive losses from latest backward
        loss_streak = 0
        for r in rows:
            if r["outcome"] == "LOSS":
                loss_streak += 1
            else:
                break

        # Regime drift triggers: win rate < 40% OR 3 consecutive losses
        drift_detected = (win_rate < 40.0) or (loss_streak >= 3)
        status = "DEFENSIVE_QUARANTINE" if drift_detected else "STABLE_PERFORMANCE"

        recommendation = (
            "⚠️ DERIVA DE RÉGIMEN DETECTADA: Pausar entradas agresivas, exigir consenso unánime "
            "de 4 analistas y reducir el tamaño máximo por operación al 50%."
            if drift_detected else
            "✅ RÉGIMEN ESTABLE: Los patrones analizados mantienen rentabilidad estadística esperada."
        )

        return {
            "drift_detected": drift_detected,
            "status": status,
            "sample_size": len(rows),
            "win_rate_pct": round(win_rate, 1),
            "consecutive_losses": loss_streak,
            "recommendation": recommendation,
        }

    def get_empirical_meta_calibration(self) -> dict[str, Any]:
        """Empirically evaluate recent closed trade performance to calibrate Meta-Labeling threshold."""
        drift = self.detect_regime_drift(window_size=12)
        status = drift.get("status", "INSUFFICIENT_HISTORY")
        win_rate = drift.get("win_rate_pct", 100.0)
        streak = drift.get("consecutive_losses", 0)
        sample_size = drift.get("sample_size", 0)

        if sample_size < 3:
            return {
                "threshold_adjustment": 0.0,
                "rolling_win_rate": win_rate,
                "consecutive_losses": streak,
                "regime_status": status,
                "reason": "Historial incipiente (<3 trades). Calibración nominal base.",
            }

        if status == "DEFENSIVE_QUARANTINE" or win_rate < 45.0 or streak >= 2:
            adj = +0.07  # Tighten quality gate from 0.62 to 0.69
            reason = f"Régimen de ruido/corrección (WinRate: {win_rate:.0f}%, Pérdidas seguidas: {streak}). Calibración ajustada a +{adj*100:.0f}% (mayor exigencia de confluencia)."
        elif win_rate >= 70.0 and streak == 0:
            adj = -0.04  # Ease slightly from 0.62 to 0.58 to catch more runners
            reason = f"Régimen de alta persistencia/tendencia (WinRate: {win_rate:.0f}%). Calibración ajustada a {adj*100:.0f}% para maximizar captura de runners."
        else:
            adj = 0.0
            reason = f"Régimen equilibrado (WinRate: {win_rate:.0f}%). Calibración nominal base."

        return {
            "threshold_adjustment": adj,
            "rolling_win_rate": win_rate,
            "consecutive_losses": streak,
            "regime_status": status,
            "reason": reason,
        }

    def get_learned_patterns_summary(self) -> dict[str, Any]:
        """Return aggregated summary of learned mistake signatures and positive patterns."""
        with self._get_connection() as conn:
            mistakes = conn.execute("""
                SELECT mistake_signature, lesson_learned, COUNT(*) as frequency, AVG(pnl_pct) as avg_loss_pct
                FROM trade_patterns
                WHERE outcome = 'LOSS'
                GROUP BY mistake_signature
                ORDER BY frequency DESC
            """).fetchall()

            wins = conn.execute("""
                SELECT mistake_signature, COUNT(*) as frequency, AVG(pnl_pct) as avg_win_pct
                FROM trade_patterns
                WHERE outcome = 'WIN'
                GROUP BY mistake_signature
                ORDER BY frequency DESC
            """).fetchall()

            total_trades = conn.execute("SELECT COUNT(*) as c FROM trade_patterns").fetchone()["c"]

        return {
            "total_patterns_cataloged": total_trades,
            "mistake_patterns": [
                {
                    "signature": m["mistake_signature"],
                    "frequency": m["frequency"],
                    "avg_loss_pct": round(m["avg_loss_pct"], 2),
                    "lesson": m["lesson_learned"],
                }
                for m in mistakes
            ],
            "winning_patterns": [
                {
                    "signature": w["mistake_signature"],
                    "frequency": w["frequency"],
                    "avg_win_pct": round(w["avg_win_pct"], 2),
                }
                for w in wins
            ],
            "drift_metrics": self.detect_regime_drift(),
        }

    def get_historical_feature_matrix(self, limit: int = 500) -> list[dict[str, Any]]:
        """Retrieve historical trades with full feature snapshots and rewards for ML training."""
        records: list[dict[str, Any]] = []
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM (
                        SELECT trade_id, symbol, outcome, pnl_pct, reward_score,
                               rsi_entry, ema_ratio, volatility_atr_pct, market_regime,
                               mae_pct, mfe_pct, exit_reason, duration_hours, features_json,
                               created_at
                        FROM trade_patterns
                        ORDER BY created_at DESC
                        LIMIT ?
                    ) ORDER BY created_at ASC
                """, (limit,)).fetchall()
                for r in rows:
                    rec = dict(r)
                    if rec.get("features_json"):
                        try:
                            rec["features"] = json.loads(rec["features_json"])
                        except Exception:
                            rec["features"] = {}
                    else:
                        rec["features"] = {}
                    records.append(rec)
        except Exception as e:
            logger.debug("Failed to fetch historical feature matrix: %s", e)
        return records

    def get_asset_track_record(
        self,
        symbol: str,
        lookback: int = 10,
        min_trades: int = 2,
        min_win_rate_pct: float = 40.0,
    ) -> dict[str, Any]:
        """Query rolling performance for a specific asset over recent closed trades.
        
        Calculates win rate, trade count, and whether asset qualifies or fails quality gate (<40% win rate).
        """
        norm_sym = symbol.replace("-", "").replace("/", "").strip().upper()
        records: list[dict[str, Any]] = []
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT outcome, pnl_pct, symbol
                    FROM trade_patterns
                    ORDER BY created_at DESC
                """).fetchall()
                for r in rows:
                    row_sym = str(r["symbol"]).replace("-", "").replace("/", "").strip().upper()
                    if row_sym == norm_sym:
                        records.append({
                            "outcome": str(r["outcome"]).upper(),
                            "pnl_pct": float(r["pnl_pct"]),
                        })
                        if len(records) >= lookback:
                            break
        except Exception as e:
            logger.debug("Failed to query asset track record for %s: %s", symbol, e)

        # Fallback to Neon if SQLite has insufficient records and not in tests
        if len(records) < min_trades and not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                from bot.database.neon_repo import get_neon_repo
                repo = get_neon_repo()
                if repo and repo.is_connected:
                    neon_rows = repo.get_learned_patterns(limit=50)
                    for nr in neon_rows:
                        nr_sym = str(nr.get("symbol", "")).replace("-", "").replace("/", "").strip().upper()
                        if nr_sym == norm_sym:
                            metrics = nr.get("metrics") or {}
                            records.append({
                                "outcome": str(nr.get("outcome", "")).upper(),
                                "pnl_pct": float(metrics.get("pnl_pct", 0.0)),
                            })
                            if len(records) >= lookback:
                                break
            except Exception as ne:
                logger.debug("Notice querying Neon for asset track record: %s", ne)

        trade_count = len(records)
        if trade_count == 0:
            return {
                "symbol": symbol,
                "trade_count": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_pct": 50.0,
                "avg_pnl_pct": 0.0,
                "is_disqualified": False,
                "reason": "Sin historial previo.",
            }

        wins = sum(1 for r in records if r["outcome"] == "WIN" or r["pnl_pct"] > 0)
        losses = trade_count - wins
        win_rate = (wins / trade_count) * 100.0
        avg_pnl = sum(r["pnl_pct"] for r in records) / trade_count

        is_disqualified = (trade_count >= min_trades) and (win_rate < min_win_rate_pct)
        reason = (
            f"Bajo Win Rate histórico ({win_rate:.1f}% en últimas {trade_count} operaciones < {min_win_rate_pct:.0f}%). Excluido de nuevas compras."
            if is_disqualified else
            f"Win Rate aceptable ({win_rate:.1f}% en {trade_count} operaciones)."
        )

        return {
            "symbol": symbol,
            "trade_count": trade_count,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 1),
            "avg_pnl_pct": round(avg_pnl, 2),
            "is_disqualified": is_disqualified,
            "reason": reason,
        }

    def evaluate_asset_quality(
        self,
        symbol: str,
        lookback: int = 10,
        min_trades: int = 2,
        min_win_rate_pct: float = 40.0,
    ) -> PatternCheckResult:
        """Evaluate whether an asset qualifies under historical win-rate filtering."""
        track = self.get_asset_track_record(
            symbol=symbol,
            lookback=lookback,
            min_trades=min_trades,
            min_win_rate_pct=min_win_rate_pct,
        )
        if track.get("is_disqualified"):
            return PatternCheckResult(
                is_safe=False,
                risk_multiplier=0.0,
                matched_mistake_tag="LOW_HISTORICAL_WIN_RATE",
                similarity_score=100.0,
                warning_message=f"Activo {symbol} vetado: Win Rate {track['win_rate_pct']:.1f}% < {min_win_rate_pct:.0f}% en últimas {track['trade_count']} operaciones.",
                learned_lesson=f"Excluir activos con bajo desempeño ({symbol}) para enfocar capital en líderes consistentes.",
            )
        return PatternCheckResult(is_safe=True, risk_multiplier=1.0)

