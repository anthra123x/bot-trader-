"""Adaptive Machine Learning Model (LightGBM) for Lopez de Prado Meta-Labeling.

Replaces static linear heuristic weights with a self-calibrating Gradient Boosted Decision Tree
trained online on real historical trade outcomes and continuous reward feedback.

Architecture:
1. Online Feature Extractor: Transforms 20+ sensory signals into normalized vector.
2. Binary Classifier: LightGBM shallow trees (max_depth=3) with sample weighting based on continuous reward magnitude.
3. Fallback Mechanism: If historical samples < min_samples_to_train (default 25), seamlessly falls back
   to the calibrated heuristic while accumulating data.
4. Feature Importance Tracker: Quantifies which technical/microstructure indicators drive alpha.
5. Dual Persistence: Local disk cache + Neon PostgreSQL system_config for zero-loss cloud reboots.
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
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from bot.models import AssetClass, OrderSide, TradeFeatureSnapshot, TradeSignal

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "rsi_14",
    "ema_ratio_50",
    "ema_ratio_200",
    "atr_pct",
    "band_pct",
    "hurst_exponent",
    "volume_delta_flow",
    "is_cvd_divergent",
    "l2_imbalance_ratio",
    "spread_pct",
    "fear_greed_value",
    "sentiment_score",
    "hour_utc",
    "day_of_week",
    "screener_score",
    "signal_confidence",
    "market_regime_code",
    "mtf_confluence_code",
    "asset_class_code",
]


def encode_regime(regime_str: str) -> float:
    """Encode market regime into an ordinal scale."""
    r = (regime_str or "").upper()
    if "TRENDING_BULL" in r or "EXPANSION" in r or "MOMENTUM" in r:
        return 2.0
    elif "BULL" in r or "RALLY" in r:
        return 1.0
    elif "NEUTRAL" in r or "CHOP" in r:
        return 0.0
    elif "BEAR" in r or "BREAKDOWN" in r or "DEFENSIVE" in r:
        return -2.0
    return 0.0


def encode_confluence(mtf_str: str) -> float:
    c = (mtf_str or "").upper()
    if "ALIGNED" in c or "BULL" in c:
        return 1.0
    elif "OPPOSED" in c or "BEAR" in c:
        return -1.0
    return 0.0


def extract_features_vector(
    signal: TradeSignal,
    market_regime: str = "NEUTRAL",
    snapshot: TradeFeatureSnapshot | None = None,
) -> np.ndarray:
    """Extract ordered 19-dimensional float vector for model inference or training."""
    metrics = signal.metrics or {}

    if snapshot:
        rsi = snapshot.rsi_14
        ema50 = snapshot.ema_ratio_50
        ema200 = snapshot.ema_ratio_200
        atr = snapshot.atr_pct
        band = snapshot.band_pct
        hurst = snapshot.hurst_exponent
        cvd = snapshot.volume_delta_flow
        cvd_div = 1.0 if snapshot.is_cvd_divergent else 0.0
        l2 = snapshot.l2_imbalance_ratio
        spread = snapshot.spread_pct
        fg = float(snapshot.fear_greed_value)
        sent = snapshot.sentiment_score
        hour = float(snapshot.hour_utc)
        dow = float(snapshot.day_of_week)
        score = snapshot.screener_score
        reg = encode_regime(snapshot.market_regime)
        mtf = encode_confluence(snapshot.mtf_confluence)
    else:
        rsi = float(metrics.get("rsi", 50.0))
        ema50 = float(metrics.get("ema_ratio", 1.0))
        ema200 = float(metrics.get("ema_ratio_200", 1.0))
        atr = float(metrics.get("atr_pct", 2.0))
        band = float(metrics.get("band_pct", 0.5))
        hurst = float(metrics.get("hurst_exponent", 0.5))
        cvd = float(metrics.get("volume_delta_flow", 0.0))
        cvd_div = float(metrics.get("is_cvd_divergent", 0.0))
        l2 = float(metrics.get("l2_imbalance_ratio", 1.0))
        spread = float(metrics.get("spread_pct", 0.05))
        fg = float(metrics.get("fear_greed_value", 50))
        sent = float(metrics.get("sentiment_score", 0.0))
        hour = float(metrics.get("hour_utc", 12))
        dow = float(metrics.get("day_of_week", 2))
        score = float(metrics.get("screener_score", metrics.get("score", 50.0)))
        reg = encode_regime(str(metrics.get("regime") or market_regime))
        mtf = encode_confluence(str(metrics.get("mtf_confluence", "NEUTRAL")))

    conf = float(signal.confidence) if signal.confidence else 0.5
    ac = 1.0 if signal.asset_class == AssetClass.CRYPTO else 0.0

    vec = [
        rsi,
        ema50,
        ema200,
        atr,
        band,
        hurst,
        cvd,
        cvd_div,
        l2,
        spread,
        fg,
        sent,
        hour,
        dow,
        score,
        conf,
        reg,
        mtf,
        ac,
    ]
    return np.array(vec, dtype=np.float32)


class AdaptiveModel:
    """Self-learning online LightGBM model for Meta-Labeling trade probability prediction."""

    def __init__(
        self,
        min_samples_to_train: int = 25,
        model_path: str = "data/adaptive_model.lgb",
        meta_path: str = "data/adaptive_model_meta.json",
    ):
        self.min_samples_to_train = min_samples_to_train
        self.model_path = model_path
        self.meta_path = meta_path
        self._classifier: Any = None
        self._is_trained: bool = False
        self._sample_count: int = 0
        self._train_accuracy: float = 0.0
        self._feature_importances: dict[str, float] = {}
        self._last_trained_ts: float = 0.0

        # Create dir
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        self.load_model()

    @property
    def is_trained(self) -> bool:
        return self._is_trained and self._classifier is not None

    def load_model(self) -> bool:
        """Load persisted LightGBM model and metadata from disk or Neon DB."""
        if not LIGHTGBM_AVAILABLE:
            logger.debug("[ADAPTIVE_MODEL] LightGBM no disponible en el entorno.")
            return False

        # 1. Try local disk
        if os.path.exists(self.model_path) and os.path.exists(self.meta_path):
            try:
                clf = lgb.Booster(model_file=self.model_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._classifier = clf
                self._sample_count = int(meta.get("sample_count", 0))
                self._train_accuracy = float(meta.get("train_accuracy", 0.0))
                self._feature_importances = meta.get("feature_importances", {})
                self._is_trained = True
                logger.info(
                    "[ADAPTIVE_MODEL] 🧠 Modelo LightGBM cargado desde disco (%d muestras, precisión: %.1f%%)",
                    self._sample_count, self._train_accuracy * 100.0
                )
                return True
            except Exception as e:
                logger.debug("[ADAPTIVE_MODEL] Error cargando modelo local: %s", e)

        # 2. Try Neon system_config
        try:
            from bot.database.neon_repo import get_neon_repo
            repo = get_neon_repo()
            if repo.is_connected:
                raw_json = repo.get_system_config("ADAPTIVE_MODEL_STATE")
                if raw_json:
                    state = json.loads(raw_json)
                    model_str = state.get("model_str")
                    if model_str:
                        clf = lgb.Booster(model_str=model_str)
                        self._classifier = clf
                        self._sample_count = int(state.get("sample_count", 0))
                        self._train_accuracy = float(state.get("train_accuracy", 0.0))
                        self._feature_importances = state.get("feature_importances", {})
                        self._is_trained = True
                        logger.info(
                            "[ADAPTIVE_MODEL] 🧠 Modelo LightGBM restaurado desde Neon PostgreSQL (%d muestras)",
                            self._sample_count
                        )
                        return True
        except Exception as ne:
            logger.debug("[ADAPTIVE_MODEL] Error restaurando modelo desde Neon: %s", ne)

        return False

    def save_model(self) -> bool:
        """Persist trained model to disk and Neon DB system_config."""
        if not self._classifier:
            return False

        # 1. Save to local disk
        try:
            self._classifier.save_model(self.model_path)
            meta = {
                "sample_count": self._sample_count,
                "train_accuracy": self._train_accuracy,
                "feature_importances": self._feature_importances,
                "feature_names": FEATURE_NAMES,
            }
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as fe:
            logger.debug("[ADAPTIVE_MODEL] Error guardando archivo local: %s", fe)

        # 2. Save to Neon PostgreSQL
        try:
            from bot.database.neon_repo import get_neon_repo
            repo = get_neon_repo()
            if repo.is_connected:
                model_str = self._classifier.model_to_string()
                payload = {
                    "sample_count": self._sample_count,
                    "train_accuracy": self._train_accuracy,
                    "feature_importances": self._feature_importances,
                    "model_str": model_str,
                }
                repo.set_system_config("ADAPTIVE_MODEL_STATE", json.dumps(payload))
                logger.info("[ADAPTIVE_MODEL] 💾 Estado del modelo sincronizado con Neon PostgreSQL.")
                return True
        except Exception as ne:
            logger.debug("[ADAPTIVE_MODEL] Error persistiendo a Neon: %s", ne)

        return True

    def train_on_history(
        self,
        historical_records: list[dict[str, Any]],
        force_retrain: bool = False,
    ) -> bool:
        """Train or update LightGBM classifier on historical trade outcomes.
        
        Args:
            historical_records: List of dicts from PatternMemory.get_historical_feature_matrix()
            force_retrain: If True, retrains even if sample count hasn't changed.
        """
        if not LIGHTGBM_AVAILABLE:
            return False

        if len(historical_records) < self.min_samples_to_train:
            logger.info(
                "[ADAPTIVE_MODEL] ℹ️ Muestras insuficientes para entrenamiento (%d/%d requeridas). Operando en modo heurístico adaptativo.",
                len(historical_records), self.min_samples_to_train
            )
            return False

        X_list: list[list[float]] = []
        y_list: list[int] = []
        weights_list: list[float] = []

        for r in historical_records:
            feat_dict = r.get("features") or {}
            # Reconstruct or parse feature vector
            rsi = float(feat_dict.get("rsi_14", r.get("rsi_entry", 50.0)))
            ema50 = float(feat_dict.get("ema_ratio_50", r.get("ema_ratio", 1.0)))
            ema200 = float(feat_dict.get("ema_ratio_200", 1.0))
            atr = float(feat_dict.get("atr_pct", r.get("volatility_atr_pct", 2.0)))
            band = float(feat_dict.get("band_pct", 0.5))
            hurst = float(feat_dict.get("hurst_exponent", 0.5))
            cvd = float(feat_dict.get("volume_delta_flow", 0.0))
            cvd_div = 1.0 if feat_dict.get("is_cvd_divergent") else 0.0
            l2 = float(feat_dict.get("l2_imbalance_ratio", 1.0))
            spread = float(feat_dict.get("spread_pct", 0.05))
            fg = float(feat_dict.get("fear_greed_value", 50))
            sent = float(feat_dict.get("sentiment_score", 0.0))
            hour = float(feat_dict.get("hour_utc", 12))
            dow = float(feat_dict.get("day_of_week", 2))
            score = float(feat_dict.get("screener_score", 50.0))
            conf = 0.70  # Default historical confidence
            reg = encode_regime(str(r.get("market_regime", "NEUTRAL")))
            mtf = encode_confluence(str(feat_dict.get("mtf_confluence", "NEUTRAL")))
            ac = 1.0 if "CRYPTO" in str(r.get("asset_class", "")).upper() else 0.0

            x = [
                rsi, ema50, ema200, atr, band, hurst, cvd, cvd_div, l2, spread,
                fg, sent, hour, dow, score, conf, reg, mtf, ac
            ]
            X_list.append(x)

            # Target label: 1 if WIN / positive return, 0 if LOSS / negative
            outcome_str = str(r.get("outcome", "")).upper()
            pnl = float(r.get("pnl_pct", 0.0))
            reward = float(r.get("reward_score", 0.0))
            is_win = 1 if (outcome_str == "WIN" or pnl > 0 or reward > 0) else 0
            y_list.append(is_win)

            # Sample weight based on continuous reward magnitude
            weight = max(0.2, min(2.5, 1.0 + abs(reward) + (abs(pnl) / 4.0)))
            weights_list.append(weight)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        weights = np.array(weights_list, dtype=np.float32)

        # Require at least both classes present (win and loss)
        if len(np.unique(y)) < 2:
            logger.info("[ADAPTIVE_MODEL] Solo una clase presente en los datos. Esperando diversidad de operaciones.")
            return False

        try:
            train_data = lgb.Dataset(X, label=y, weight=weights, feature_name=FEATURE_NAMES)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.05,
                "num_leaves": 7,
                "max_depth": 3,
                "min_data_in_leaf": max(3, len(X) // 10),
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "verbose": -1,
                "seed": 42,
            }

            booster = lgb.train(
                params,
                train_data,
                num_boost_round=40,
            )

            # Evaluate training accuracy
            preds = booster.predict(X)
            pred_labels = (preds >= 0.5).astype(int)
            acc = float(np.mean(pred_labels == y))

            # Feature importance
            raw_importances = booster.feature_importance(importance_type="gain")
            tot_imp = float(np.sum(raw_importances)) if np.sum(raw_importances) > 0 else 1.0
            feat_imp_dict = {
                FEATURE_NAMES[i]: round(float(raw_importances[i]) / tot_imp * 100.0, 2)
                for i in range(len(FEATURE_NAMES))
            }
            # Sort descending
            feat_imp_dict = dict(sorted(feat_imp_dict.items(), key=lambda item: item[1], reverse=True))

            self._classifier = booster
            self._is_trained = True
            self._sample_count = len(X)
            self._train_accuracy = acc
            self._feature_importances = feat_imp_dict
            self.save_model()

            top3 = list(feat_imp_dict.items())[:3]
            top3_str = ", ".join(f"{k}: {v:.1f}%" for k, v in top3)
            logger.info(
                "[ADAPTIVE_MODEL] 🚀 Modelo LightGBM re-entrenado exitosamente (%d muestras | Precisión: %.1f%%). Top predictores: %s",
                self._sample_count, acc * 100.0, top3_str
            )
            return True
        except Exception as te:
            logger.error("[ADAPTIVE_MODEL] Error durante entrenamiento de LightGBM: %s", te)
            return False

    def predict_p_success(
        self,
        signal: TradeSignal,
        market_regime: str = "NEUTRAL",
        snapshot: TradeFeatureSnapshot | None = None,
    ) -> tuple[float, str]:
        """Predict probability of trade success P(Success) using trained LightGBM or graceful heuristic fallback.
        
        Returns: (probability, explanation)
        """
        x_vec = extract_features_vector(signal, market_regime=market_regime, snapshot=snapshot)

        # 1. If trained LightGBM available, use it!
        if self.is_trained:
            try:
                x_2d = x_vec.reshape(1, -1)
                prob_raw = float(self._classifier.predict(x_2d)[0])
                prob = float(np.clip(prob_raw, 0.05, 0.95))

                top_features = list(self._feature_importances.items())[:2]
                top_str = " & ".join(f"{k} ({v:.0f}%)" for k, v in top_features)
                reason = f"LightGBM Neural Predictor (Muestras={self._sample_count}, Top factores: {top_str})"
                return prob, reason
            except Exception as pe:
                logger.debug("[ADAPTIVE_MODEL] Error inferencia LightGBM, recurriendo a fallback: %s", pe)

        # 2. Graceful Heuristic Fallback (Lopez de Prado statistical rules)
        prob = 0.50
        metrics = signal.metrics or {}

        # Signal confidence
        sig_conf = float(signal.confidence) if signal.confidence else 0.50
        prob += (sig_conf - 0.50) * 0.25

        # Hurst Exponent (fractal persistence)
        h_val = float(metrics.get("hurst_exponent", 0.50))
        if h_val >= 0.56:
            prob += 0.12
        elif h_val >= 0.52:
            prob += 0.05
        elif h_val < 0.48:
            prob -= 0.15

        # CVD Volume delta
        cvd_flow = float(metrics.get("volume_delta_flow", 0.0))
        is_cvd_div = bool(metrics.get("is_cvd_divergent", False))
        if is_cvd_div:
            prob -= 0.18
        elif cvd_flow > 0.20:
            prob += 0.10

        # L2 Imbalance
        l2 = float(metrics.get("l2_imbalance_ratio", 1.0))
        if l2 >= 1.25:
            prob += 0.08
        elif l2 <= 0.60:
            prob -= 0.15

        # Regime
        reg_code = encode_regime(market_regime)
        if reg_code > 0:
            prob += 0.06
        elif reg_code < 0:
            prob -= 0.18

        prob = float(np.clip(prob, 0.05, 0.95))
        reason = f"Heurística Estadística (Muestras de entrenamiento en acumulación: {self._sample_count}/{self.min_samples_to_train})"
        return prob, reason

    def get_feature_importances(self) -> dict[str, float]:
        """Return percentage feature importance dictionary."""
        return self._feature_importances


ADAPTIVE_MODEL = AdaptiveModel()
