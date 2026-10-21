import json
import logging
import math
import os
import time
from typing import Any
import numpy as np
from bot.config import BotRiskConfig
from bot.learning.pattern_memory import PatternMemory
from bot.models import AccountBalance, AssetClass, OrderSide, Position, RiskValidationResult, TradeSignal
from bot.symbols import is_crypto

logger = logging.getLogger(__name__)


def normalize_symbol(symbol: str) -> str:
    """Normalize trading symbols for unified comparisons (e.g. BTC-USD, BTC/USD, BTCUSD -> BTCUSD)."""
    return symbol.replace("-", "").replace("/", "").strip().upper()


class RiskEngine:
    """Evaluates trade signals against strict quantitative capital protection rules."""

    def __init__(self, config: BotRiskConfig | None = None, pattern_memory: PatternMemory | None = None):
        self.config = config or BotRiskConfig()
        self.pattern_memory = pattern_memory or PatternMemory()
        # QuantConnect LEAN: High-water mark tracking for trailing stop
        self._peak_prices: dict[str, float] = {}
        # Neural Feature Store: Trough tracking for MAE (Maximum Adverse Excursion)
        self._trough_prices: dict[str, float] = {}
        # Freqtrade: Symbol cooldown tracker after exits
        self._symbol_cooldowns: dict[str, float] = {}
        # Freqtrade: Rolling stoploss guard timestamps
        self._recent_stoploss_events: list[float] = []
        # Tiered Partial Take-Profit tracker
        self._partial_tp_executed: set[str] = set()
        self._partial_tp_stages: dict[str, int] = {}
        # Dynamic Runner Target Extension tracker
        self._runner_extended: set[str] = set()
        # Persistent position targets tracker (Take-Profit & Stop-Loss)
        self._position_targets: dict[str, dict[str, float]] = {}
        # Entry context tracking (RSI, EMA, ATR, Regime) for post-mortem learning
        self.entry_contexts: dict[str, dict[str, Any]] = {}
        # Portfolio-level Peak Equity High-Water Mark tracker
        self._peak_equity: float = 0.0
        # Permanent Profit Quarantine Vault Ratchet (never fluctuates down)
        self._peak_quarantined_profit: float = 0.0
        # In-flight orders tracker: symbol -> list of dicts (timestamp, qty, cost, asset_class)
        self._in_flight_orders: dict[str, list[dict[str, Any]]] = {}
        # Position entry duration tracker for Opportunity Cost Time-Stop
        self._position_entry_times: dict[str, float] = {}
        # Consecutive loss streak per symbol for Extended Quarantine
        self._symbol_loss_streak: dict[str, int] = {}
        # Hydrate active positions and risk states from Neon DB / local cache
        self.hydrate_active_positions_state()
        # Apply Bayesian optimized parameters if available (skip during pytest to preserve test config)
        if "PYTEST_CURRENT_TEST" not in os.environ:
            try:
                from bot.learning.param_optimizer import PARAM_OPTIMIZER
                PARAM_OPTIMIZER.apply_to_config(self.config)
            except Exception as oe:
                logger.debug("[PARAM_OPTIMIZER] Init notice: %s", oe)

    def register_order_in_flight(self, symbol: str, quantity: float, price: float, asset_class: AssetClass) -> None:
        """Track newly submitted order in-flight to prevent duplicate rapid pyramiding."""
        norm_sym = normalize_symbol(symbol)
        now = time.time()
        if norm_sym not in self._in_flight_orders:
            self._in_flight_orders[norm_sym] = []
        self._in_flight_orders[norm_sym].append({
            "timestamp": now,
            "quantity": quantity,
            "cost": quantity * price,
            "asset_class": asset_class,
        })

    def get_in_flight_capital(self, symbol: str | None = None, asset_class: AssetClass | None = None) -> float:
        """Return total in-flight capital for a symbol or asset class (pruning orders older than 180s)."""
        now = time.time()
        total = 0.0
        norm_target = normalize_symbol(symbol) if symbol else None
        for sym, orders in list(self._in_flight_orders.items()):
            active = [o for o in orders if now - o["timestamp"] < 180.0]
            self._in_flight_orders[sym] = active
            for o in active:
                if norm_target and sym != norm_target:
                    continue
                if asset_class and o.get("asset_class") != asset_class:
                    continue
                total += o.get("cost", 0.0)
        return total

    # --------------------------------------------------------------------------
    # ACTIVE POSITIONS STATE PERSISTENCE (Cloud Reboot Resilience)
    # --------------------------------------------------------------------------

    def save_active_positions_state(self) -> None:
        """Persist current active positions' risk state (peaks, TP stages, targets, contexts) to Neon & local cache."""
        if not getattr(self.config, "enable_position_state_persistence", True):
            return
        if "PYTEST_CURRENT_TEST" in os.environ:
            return

        state: dict[str, Any] = {}
        all_symbols = set(self._position_targets.keys()) | set(self._peak_prices.keys()) | set(self._trough_prices.keys()) | set(self._position_entry_times.keys())
        for sym in all_symbols:
            norm_sym = normalize_symbol(sym)
            targets = self._position_targets.get(norm_sym, {})
            state[norm_sym] = {
                "peak_price": self._peak_prices.get(norm_sym, 0.0),
                "trough_price": self._trough_prices.get(norm_sym, 0.0),
                "tp_stage": self._partial_tp_stages.get(norm_sym, 0),
                "partial_tp_executed": norm_sym in self._partial_tp_executed,
                "runner_extended": norm_sym in self._runner_extended,
                "stop_loss": targets.get("stop_loss", 0.0),
                "take_profit": targets.get("take_profit"),
                "entry_price": targets.get("entry_price", 0.0),
                "entry_context": self.entry_contexts.get(norm_sym, {}),
                "entry_time": self._position_entry_times.get(norm_sym, 0.0),
                "cooldown_until": self._symbol_cooldowns.get(norm_sym, 0.0),
                "loss_streak": self._symbol_loss_streak.get(norm_sym, 0),
            }

        try:
            json_str = json.dumps(state)
        except Exception as je:
            logger.debug("[PERSISTENCE] JSON serialization notice: %s", je)
            return

        # 1. Persist to Neon DB system_config
        try:
            from bot.database.neon_repo import get_neon_repo
            repo = get_neon_repo()
            if repo.is_connected:
                repo.set_system_config("ACTIVE_POSITIONS_STATE", json_str)
        except Exception as ne:
            logger.debug("[PERSISTENCE] Neon position state save notice: %s", ne)

        # 2. Persist to local filesystem fallback
        try:
            cache_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "active_positions_state.json")
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(json_str)
        except Exception as fe:
            logger.debug("[PERSISTENCE] File position state save notice: %s", fe)

    def hydrate_active_positions_state(self, open_positions_symbols: list[str] | None = None) -> int:
        """Hydrate in-memory risk trackers from Neon DB / local cache to recover state after reboot."""
        if not getattr(self.config, "enable_position_state_persistence", True):
            return 0
        if "PYTEST_CURRENT_TEST" in os.environ and not getattr(self, "_force_test_hydration", False):
            return 0

        raw_json = None
        # Try Neon first
        try:
            from bot.database.neon_repo import get_neon_repo
            repo = get_neon_repo()
            if repo.is_connected:
                raw_json = repo.get_system_config("ACTIVE_POSITIONS_STATE")
        except Exception as ne:
            logger.debug("[PERSISTENCE] Neon position state load notice: %s", ne)

        # Fallback to local file
        if not raw_json:
            cache_file = os.path.join(os.getcwd(), "data", "active_positions_state.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        raw_json = f.read()
                except Exception as fe:
                    logger.debug("[PERSISTENCE] File position state load notice: %s", fe)

        if not raw_json:
            return 0

        try:
            state = json.loads(raw_json)
        except Exception:
            return 0

        filter_symbols = {normalize_symbol(s) for s in open_positions_symbols} if open_positions_symbols is not None else None
        hydrated_count = 0

        for norm_sym, data in state.items():
            if filter_symbols is not None and norm_sym not in filter_symbols:
                if data.get("cooldown_until"):
                    self._symbol_cooldowns[norm_sym] = float(data["cooldown_until"])
                if data.get("loss_streak"):
                    self._symbol_loss_streak[norm_sym] = int(data["loss_streak"])
                continue

            if data.get("peak_price", 0.0) > 0:
                self._peak_prices[norm_sym] = float(data["peak_price"])
            if data.get("trough_price", 0.0) > 0:
                self._trough_prices[norm_sym] = float(data["trough_price"])
            if data.get("tp_stage", 0) > 0:
                self._partial_tp_stages[norm_sym] = int(data["tp_stage"])
            if data.get("partial_tp_executed"):
                self._partial_tp_executed.add(norm_sym)
            if data.get("runner_extended"):
                self._runner_extended.add(norm_sym)
            if data.get("stop_loss", 0.0) > 0 or data.get("take_profit"):
                self._position_targets[norm_sym] = {
                    "entry_price": float(data.get("entry_price", 0.0)),
                    "stop_loss": float(data.get("stop_loss", 0.0)),
                    "take_profit": float(data["take_profit"]) if data.get("take_profit") else None,
                }
            if data.get("entry_context"):
                self.entry_contexts[norm_sym] = data["entry_context"]
            if data.get("entry_time", 0.0) > 0:
                self._position_entry_times[norm_sym] = float(data["entry_time"])
            if data.get("cooldown_until"):
                self._symbol_cooldowns[norm_sym] = float(data["cooldown_until"])
            if data.get("loss_streak"):
                self._symbol_loss_streak[norm_sym] = int(data["loss_streak"])

            hydrated_count += 1

        if hydrated_count > 0:
            logger.info(
                "[PERSISTENCE] 💾 Estado de posiciones activas hidratado exitosamente (%d posiciones restauradas).",
                hydrated_count
            )
        return hydrated_count

    def register_entry_context(self, symbol: str, context: dict[str, Any]) -> None:
        """Register technical metrics at entry for post-trade feedback and learning."""
        norm_sym = normalize_symbol(symbol)
        existing = self.entry_contexts.get(norm_sym) or self.entry_contexts.get(symbol) or {}
        merged = {**existing, **context}
        self.entry_contexts[norm_sym] = merged
        self.entry_contexts[symbol] = merged
        self.save_active_positions_state()

    def get_entry_context(self, symbol: str) -> dict[str, Any]:
        """Retrieve technical metrics captured when position was entered."""
        norm_sym = normalize_symbol(symbol)
        return self.entry_contexts.get(norm_sym) or self.entry_contexts.get(symbol) or {}

    def register_position_targets(self, symbol: str, entry_price: float, stop_loss: float, take_profit: float | None) -> None:
        """Register active target take-profit and stop-loss levels for a position."""
        norm_sym = normalize_symbol(symbol)
        if take_profit is None:
            default_sl_dist = entry_price * (self.config.default_stop_loss_pct / 100.0)
            take_profit = round(entry_price + (default_sl_dist * self.config.min_risk_reward_ratio), 4)
        self._position_targets[norm_sym] = {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        self.save_active_positions_state()

    def get_position_targets(self, symbol: str, entry_price: float) -> dict[str, float]:
        """Retrieve target TP and SL for a position. If not tracked, generates based on R:R config."""
        norm_sym = normalize_symbol(symbol)
        targets = self._position_targets.get(norm_sym)
        if targets:
            return targets

        # Fallback calculation for preexisting positions or after process reboot
        default_sl_distance = entry_price * (self.config.default_stop_loss_pct / 100.0)
        fallback_sl = round(entry_price - default_sl_distance, 4)
        min_reward = default_sl_distance * self.config.min_risk_reward_ratio
        fallback_tp = round(entry_price + min_reward, 4)

        # Palanca C: Elastic take-profit expansion if volatility ATR is tracked
        ctx = self.get_entry_context(symbol)
        atr_pct = float(ctx.get("atr_pct", 0.0))
        if atr_pct <= 0.0 and ctx.get("atr"):
            atr_val = float(ctx.get("atr", 0.0))
            atr_pct = (atr_val / entry_price * 100.0) if entry_price > 0 else 0.0

        if getattr(self.config, "enable_adaptive_atr_targets", True) and atr_pct > 0.0:
            atr_tp_dist = entry_price * ((atr_pct * getattr(self.config, "atr_tp_multiplier_stage2", 3.0)) / 100.0)
            if atr_tp_dist > min_reward:
                fallback_tp = round(entry_price + atr_tp_dist, 4)

        self._position_targets[norm_sym] = {
            "entry_price": entry_price,
            "stop_loss": fallback_sl,
            "take_profit": fallback_tp,
        }
        self.save_active_positions_state()
        return self._position_targets[norm_sym]

    def update_position_stop(self, symbol: str, new_stop: float) -> None:
        """Update ratcheted stop loss for an open position."""
        norm_sym = normalize_symbol(symbol)
        if norm_sym in self._position_targets:
            curr_sl = self._position_targets[norm_sym]["stop_loss"]
            if new_stop > curr_sl:
                self._position_targets[norm_sym]["stop_loss"] = new_stop
                self.save_active_positions_state()

    def set_position_target_tp(self, symbol: str, new_tp: float) -> None:
        """Update target take profit (e.g. for dynamic runner target extension)."""
        norm_sym = normalize_symbol(symbol)
        if norm_sym in self._position_targets:
            self._position_targets[norm_sym]["take_profit"] = new_tp
        if symbol in self._position_targets:
            self._position_targets[symbol]["take_profit"] = new_tp
        self.save_active_positions_state()

    def register_runner_extended(self, symbol: str) -> None:
        """Mark runner target extension as active for current position lifecycle."""
        norm_sym = normalize_symbol(symbol)
        self._runner_extended.add(norm_sym)
        self._runner_extended.add(symbol)
        self.save_active_positions_state()

    def is_runner_extended(self, symbol: str) -> bool:
        """Check if runner target extension has already occurred for this symbol."""
        norm_sym = normalize_symbol(symbol)
        return norm_sym in self._runner_extended or symbol in self._runner_extended

    def register_trade_exit(self, symbol: str, is_loss: bool = False, is_stoploss_event: bool = True) -> None:
        """Register trade exit to activate Freqtrade CooldownPeriod and StoplossGuard."""
        now = time.time()
        norm_sym = normalize_symbol(symbol)
        self._peak_prices.pop(norm_sym, None)
        self._peak_prices.pop(symbol, None)
        self._trough_prices.pop(norm_sym, None)
        self._trough_prices.pop(symbol, None)
        self._partial_tp_executed.discard(norm_sym)
        self._partial_tp_executed.discard(symbol)
        self._partial_tp_stages.pop(norm_sym, None)
        self._partial_tp_stages.pop(symbol, None)
        self._runner_extended.discard(norm_sym)
        self._runner_extended.discard(symbol)
        self._position_targets.pop(norm_sym, None)
        self._position_targets.pop(symbol, None)
        self.entry_contexts.pop(norm_sym, None)
        self.entry_contexts.pop(symbol, None)
        self._position_entry_times.pop(norm_sym, None)
        self._position_entry_times.pop(symbol, None)
        if is_loss:
            self._symbol_cooldowns[norm_sym] = now
            if is_stoploss_event:
                self._recent_stoploss_events.append(now)
                # Prune events older than 1 hour
                self._recent_stoploss_events = [t for t in self._recent_stoploss_events if now - t < 3600.0]

            curr_streak = self._symbol_loss_streak.get(norm_sym, 0) + 1
            self._symbol_loss_streak[norm_sym] = curr_streak
            self._symbol_loss_streak[symbol] = curr_streak
            thresh_losses = getattr(self.config, "symbol_consecutive_losses_threshold", 2)
            if curr_streak >= thresh_losses and getattr(self.config, "enable_extended_symbol_quarantine", True):
                ext_seconds = getattr(self.config, "extended_quarantine_seconds", 14400)
                logger.warning(
                    "[EXTENDED_QUARANTINE] 🚨 %s acumula %d pérdidas consecutivas. Entrando en cuarentena defensiva por %.1f horas.",
                    symbol, curr_streak, ext_seconds / 3600.0
                )
        else:
            self._symbol_loss_streak.pop(norm_sym, None)
            self._symbol_loss_streak.pop(symbol, None)
        self.save_active_positions_state()

    def register_position_entry(self, symbol: str, entry_price: float = 0.0, timestamp: float | None = None) -> None:
        """Record entry timestamp and initial price extremes for position excursion tracking."""
        norm_sym = normalize_symbol(symbol)
        ts = timestamp if timestamp is not None and timestamp > 0 else time.time()
        self._position_entry_times[norm_sym] = ts
        self._position_entry_times[symbol] = ts
        if entry_price > 0:
            peak = max(self._peak_prices.get(norm_sym, entry_price), entry_price)
            self._peak_prices[norm_sym] = peak
            self._peak_prices[symbol] = peak
            trough = min(self._trough_prices.get(norm_sym, entry_price), entry_price)
            self._trough_prices[norm_sym] = trough
            self._trough_prices[symbol] = trough
        self.save_active_positions_state()

    def update_position_extremes(self, symbol: str, current_price: float, entry_price: float = 0.0) -> None:
        """Update ongoing high-water mark (peak) and low-water mark (trough) for MAE/MFE calculation."""
        if current_price <= 0:
            return
        norm_sym = normalize_symbol(symbol)
        base_ref = entry_price if entry_price > 0 else current_price
        peak = max(self._peak_prices.get(norm_sym, self._peak_prices.get(symbol, base_ref)), current_price)
        self._peak_prices[norm_sym] = peak
        self._peak_prices[symbol] = peak
        trough = min(self._trough_prices.get(norm_sym, self._trough_prices.get(symbol, base_ref)), current_price)
        self._trough_prices[norm_sym] = trough
        self._trough_prices[symbol] = trough

    def get_position_excursions(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        quantity: float = 1.0,
    ) -> dict[str, float]:
        """Compute Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)."""
        norm_sym = normalize_symbol(symbol)
        safe_entry = max(entry_price, 0.0001)
        safe_qty = max(quantity, 0.0)

        peak = max(self._peak_prices.get(norm_sym, self._peak_prices.get(symbol, safe_entry)), current_price, safe_entry)
        trough = min(self._trough_prices.get(norm_sym, self._trough_prices.get(symbol, safe_entry)), current_price, safe_entry)

        mfe_dollars = max(0.0, (peak - safe_entry) * safe_qty)
        mfe_pct = max(0.0, ((peak - safe_entry) / safe_entry) * 100.0)
        mae_dollars = max(0.0, (safe_entry - trough) * safe_qty)
        mae_pct = max(0.0, ((safe_entry - trough) / safe_entry) * 100.0)

        return {
            "mae_dollars": round(mae_dollars, 4),
            "mae_pct": round(mae_pct, 4),
            "mfe_dollars": round(mfe_dollars, 4),
            "mfe_pct": round(mfe_pct, 4),
            "peak_price": round(peak, 4),
            "trough_price": round(trough, 4),
        }

    def get_position_duration_hours(self, symbol: str) -> float:
        """Return the holding duration in hours for a symbol, or 0.0 if not tracked."""
        norm_sym = normalize_symbol(symbol)
        entry_ts = self._position_entry_times.get(norm_sym) or self._position_entry_times.get(symbol)
        if not entry_ts:
            return 0.0
        return max(0.0, (time.time() - entry_ts) / 3600.0)

    def evaluate_time_stop_exit(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        current_qty: float,
        entry_timestamp: float | None = None,
    ) -> tuple[bool, str]:
        """Evaluate if position has exceeded stagnation time threshold (Opportunity Cost Time-Decay).
        
        Only applies to 24/7 crypto assets. If a crypto position remains trapped within a narrow
        stagnation band (<= stagnant_range_pct, e.g. between stop loss and +1.2%) for >= max_crypto_stagnant_hours,
        it triggers an exit to rotate capital into explosive momentum movers.
        """
        if not getattr(self.config, "enable_time_stop_rotation", True):
            return False, ""
        if current_price <= 0 or entry_price <= 0 or current_qty <= 0:
            return False, ""

        if not is_crypto(symbol):
            # Equities do not trade 24/7; closed exchange hours must NOT be penalized as stagnation
            return False, ""

        norm_sym = normalize_symbol(symbol)
        now = time.time()

        # Resolve entry timestamp
        entry_ts = entry_timestamp
        if entry_ts is None or entry_ts <= 0:
            entry_ts = self._position_entry_times.get(norm_sym) or self._position_entry_times.get(symbol)

        if not entry_ts:
            # First time seeing this position; record current timestamp and let observation window run
            self.register_position_entry(symbol, entry_price, now)
            return False, ""

        elapsed_hours = (now - entry_ts) / 3600.0
        max_hours = getattr(self.config, "max_crypto_stagnant_hours", 4.0)

        if elapsed_hours < max_hours:
            return False, ""

        unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
        stagnant_band = getattr(self.config, "stagnant_range_pct", 1.2)

        # Stagnant condition: Has not achieved breakout profit (> stagnant_band)
        # e.g., hovering in chop between stop loss and +1.2%
        if unrealized_pnl_pct <= stagnant_band:
            reason = (
                f"Capital Rotation Time-Stop: {symbol} estancado por {elapsed_hours:.1f}h "
                f"(PnL: {unrealized_pnl_pct:+.2f}%, umbral <= +{stagnant_band:.1f}%). "
                f"Rotando liquidez hacia líderes de momentum."
            )
            return True, reason

        return False, ""

    def is_in_cooldown(self, symbol: str) -> tuple[bool, float]:
        """Check if symbol is currently in cooldown after a stop-loss exit (Freqtrade CooldownPeriod)."""
        norm_sym = normalize_symbol(symbol)
        now = time.time()
        if norm_sym in self._symbol_cooldowns:
            elapsed = now - self._symbol_cooldowns[norm_sym]
            streak = self._symbol_loss_streak.get(norm_sym, 0)
            thresh_losses = getattr(self.config, "symbol_consecutive_losses_threshold", 2)
            if streak >= thresh_losses and getattr(self.config, "enable_extended_symbol_quarantine", True):
                req_cd = getattr(self.config, "extended_quarantine_seconds", 14400)
            else:
                req_cd = self.config.cooldown_period_seconds

            if elapsed < req_cd:
                remaining = req_cd - elapsed
                return True, remaining
        return False, 0.0

    def get_calibrated_meta_threshold(self) -> tuple[float, str]:
        """Compute dynamically calibrated Meta-Labeling probability threshold from empirical performance."""
        base_thresh = getattr(self.config, "min_meta_probability", 0.62)
        if not hasattr(self, "pattern_memory") or not self.pattern_memory:
            return base_thresh, "Base estática"

        now = time.time()
        if hasattr(self, "_meta_calib_cache") and (now - self._meta_calib_cache[0]) < 60.0:
            return self._meta_calib_cache[1], self._meta_calib_cache[2]

        try:
            calib = self.pattern_memory.get_empirical_meta_calibration()
            adj = float(calib.get("threshold_adjustment", 0.0))
            calibrated = float(np.clip(base_thresh + adj, 0.55, 0.80))
            reason = calib.get("reason", "Calibración empírica nominal")
            self._meta_calib_cache = (now, calibrated, reason)
            return calibrated, reason
        except Exception as e:
            logger.debug("Notice calculating empirical meta calibration: %s", e)
            return base_thresh, "Base fallback"

    def evaluate_meta_labeling(
        self,
        signal: TradeSignal,
        market_regime: str = "UNKNOWN",
    ) -> tuple[bool, float, str]:
        """Meta-Labeling Binary Quality Filter based on Marcos Lopez de Prado's framework.
        
        Calculates the estimated probability of trade success P(Success) under current
        microstructure noise, fractal persistence, and volume flow.
        
        Threshold: P(Success) >= calibrated_meta_threshold (empirically adapted in real-time).
        """
        if not getattr(self.config, "enable_meta_labeling", True):
            return True, 1.0, ""

        metrics = signal.metrics or {}
        # If test signal without microstructure metrics, pass through
        if not metrics or ("hurst_exponent" not in metrics and "volume_delta_flow" not in metrics and "l2_imbalance_ratio" not in metrics):
            return True, 0.75, "Meta-labeling passthrough (no microstructure metrics provided)"

        from bot.learning.adaptive_model import ADAPTIVE_MODEL
        p_success, model_source = ADAPTIVE_MODEL.predict_p_success(
            signal=signal,
            market_regime=market_regime,
        )

        # Pattern Memory Loss Similarity check
        if hasattr(self, "pattern_memory") and self.pattern_memory:
            try:
                rsi_val = float(metrics.get("rsi", 50.0))
                ema_ratio = float(metrics.get("ema_ratio", 1.0))
                atr_pct = float(metrics.get("atr_pct", 2.0))
                pcheck = self.pattern_memory.evaluate_setup(
                    symbol=signal.symbol,
                    asset_class=signal.asset_class,
                    rsi=rsi_val,
                    ema_ratio=ema_ratio,
                    atr_pct=atr_pct,
                )
                if not pcheck.is_safe:
                    p_success -= 0.25
                elif pcheck.risk_multiplier < 1.0:
                    p_success -= 0.10
            except Exception:
                pass

        p_success = float(np.clip(p_success, 0.05, 0.95))
        min_prob, calib_reason = self.get_calibrated_meta_threshold()

        if p_success < min_prob:
            h_str = f"H={metrics.get('hurst_exponent')}" if 'hurst_exponent' in metrics else ""
            l2_str = f"L2={metrics.get('l2_imbalance_ratio')}x" if 'l2_imbalance_ratio' in metrics else ""
            noise_ctx = f" ({', '.join(filter(None, [h_str, l2_str]))})" if (h_str or l2_str) else ""
            reason = (
                f"Filtro Meta-Labeling [{model_source}]: Probabilidad de éxito estimada "
                f"P(Success)={p_success:.1%} inferior al umbral calibrado ({min_prob:.1%} - {calib_reason})"
                f"{noise_ctx}. Falso positivo neutralizado para proteger capital."
            )
            return False, p_success, reason

        return True, p_success, f"Meta-Labeling Aprobado [{model_source}] (P(Success)={p_success:.1%} >= {min_prob:.1%} - {calib_reason})"


    def evaluate_partial_take_profit(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        current_qty: float,
    ) -> tuple[bool, float]:
        """Evaluate if position qualifies for tiered partial profit harvesting (Scale-Out)."""
        if not self.config.enable_partial_take_profit or current_price <= 0 or entry_price <= 0 or current_qty <= 0:
            return False, 0.0

        norm_sym = normalize_symbol(symbol)
        stage = self._partial_tp_stages.get(norm_sym, 0)
        unrealized_profit_pct = ((current_price - entry_price) / entry_price) * 100.0

        # Palanca C: Adaptive Volatility ATR Elastic Take-Profit Floors
        ctx = self.get_entry_context(symbol)
        atr_pct = float(ctx.get("atr_pct", 0.0))
        if atr_pct <= 0.0 and ctx.get("atr"):
            atr_val = float(ctx.get("atr", 0.0))
            atr_pct = (atr_val / entry_price * 100.0) if entry_price > 0 else 0.0

        stage1_thresh = self.config.partial_take_profit_threshold_pct
        stage2_thresh = getattr(self.config, "stage2_take_profit_threshold_pct", 4.5)
        stage3_thresh = getattr(self.config, "stage3_take_profit_threshold_pct", 8.0)

        if getattr(self.config, "enable_adaptive_atr_targets", True) and atr_pct > 0.0:
            stage1_thresh = max(stage1_thresh, atr_pct * getattr(self.config, "atr_tp_multiplier_stage1", 1.5))
            stage2_thresh = max(stage2_thresh, atr_pct * getattr(self.config, "atr_tp_multiplier_stage2", 3.0))
            stage3_thresh = max(stage3_thresh, atr_pct * getattr(self.config, "atr_tp_multiplier_stage3", 5.0))

        # Stage 1: Initial scale-out threshold
        if stage == 0 and unrealized_profit_pct >= stage1_thresh:
            partial_qty = round(current_qty * self.config.partial_take_profit_ratio, 6)
            if 0 < partial_qty < current_qty:
                return True, partial_qty

        # Stage 2: Secondary scale-out threshold
        if stage == 1 and getattr(self.config, "enable_multi_stage_tp", True):
            stage2_ratio = getattr(self.config, "stage2_take_profit_ratio", 0.45)
            if unrealized_profit_pct >= stage2_thresh:
                stage2_qty = round(current_qty * stage2_ratio, 6)
                if 0 < stage2_qty < current_qty:
                    return True, stage2_qty

        # Stage 3: Tertiary scale-out threshold
        if stage == 2 and getattr(self.config, "enable_stage3_tp", True):
            stage3_ratio = getattr(self.config, "stage3_take_profit_ratio", 0.50)
            if unrealized_profit_pct >= stage3_thresh:
                stage3_qty = round(current_qty * stage3_ratio, 6)
                if 0 < stage3_qty < current_qty:
                    return True, stage3_qty

        return False, 0.0

    def register_partial_tp(self, symbol: str) -> None:
        """Mark partial take profit stage as executed for current position lifecycle."""
        norm_sym = normalize_symbol(symbol)
        curr_stage = self._partial_tp_stages.get(norm_sym, 0)
        new_stage = curr_stage + 1
        self._partial_tp_stages[norm_sym] = new_stage
        self._partial_tp_stages[symbol] = new_stage
        self._partial_tp_executed.add(norm_sym)
        self._partial_tp_executed.add(symbol)
        self.save_active_positions_state()

    def get_tp_stage(self, symbol: str) -> int:
        """Return current take profit stage executed for symbol (0=none, 1=stage 1, 2=stage 2)."""
        norm_sym = normalize_symbol(symbol)
        return self._partial_tp_stages.get(norm_sym, 0)

    def evaluate_portfolio_rebalancing(
        self,
        position: Position,
        current_price: float,
        total_equity: float,
    ) -> tuple[bool, float, str]:
        """Evaluate if an overweight profitable position should be trimmed to liberate liquidity for asset rotation."""
        if not self.config.enable_dynamic_rebalancing or total_equity <= 0 or current_price <= 0 or position.quantity <= 0:
            return False, 0.0, ""

        position_val = position.quantity * current_price
        concentration_pct = (position_val / total_equity) * 100.0
        unrealized_profit_pct = ((current_price - position.entry_price) / position.entry_price) * 100.0

        # Only trim when overweight (> max_concentration_threshold_pct) AND currently in profit
        if concentration_pct > self.config.max_concentration_threshold_pct and unrealized_profit_pct > 0.2:
            excess_pct = concentration_pct - self.config.target_concentration_pct
            excess_dollars = (excess_pct / 100.0) * total_equity
            trim_qty = round(excess_dollars / current_price, 6)
            if 0 < trim_qty < position.quantity:
                reason = (
                    f"Concentración excesiva ({concentration_pct:.1f}% > {self.config.max_concentration_threshold_pct:.1f}% máx). "
                    f"Recorte de {trim_qty} u (${excess_dollars:.2f}) para liberar liquidez en beneficio (+{unrealized_profit_pct:.2f}%)."
                )
                return True, trim_qty, reason

        return False, 0.0, ""

    def calculate_protected_cash_reserve(self, balance: AccountBalance) -> tuple[float, float, float]:
        """Calculate dynamic protected cash reserve: (total_protected_cash, base_cash_reserve, quarantined_profit).
        
        Guarantees that 40% of each completed profit milestone chunk (e.g. cada $200 USD) is permanently locked
        into an untouchable cash vault ratchet that NEVER fluctuates down, leaving the rest actively working.
        """
        min_cash_pct = getattr(self.config, "min_cash_reserve_pct", 20.0)
        base_cash_reserve = balance.equity * (min_cash_pct / 100.0)
        quarantined_profit = 0.0
        if getattr(self.config, "enable_profit_quarantine_vault", True) and balance.initial_day_equity > 0:
            net_profit = max(0.0, balance.equity - balance.initial_day_equity)
            quarantine_ratio = getattr(self.config, "profit_quarantine_ratio", 0.40)
            milestone_step = getattr(self.config, "profit_quarantine_milestone_step", 200.0)
            if milestone_step > 0:
                completed_milestones = int(net_profit // milestone_step)
                target_quarantine = completed_milestones * (milestone_step * quarantine_ratio)
            else:
                target_quarantine = net_profit * quarantine_ratio

            # Permanent Ratchet Floor: The locked vault profit can ONLY scale up, NEVER fluctuate down!
            self._peak_quarantined_profit = max(self._peak_quarantined_profit, round(target_quarantine, 4))
            quarantined_profit = self._peak_quarantined_profit

        total_protected = base_cash_reserve + quarantined_profit
        return total_protected, base_cash_reserve, quarantined_profit

    def calculate_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        current_stop_loss: float,
    ) -> tuple[float, bool]:
        """QuantConnect LEAN High-Water Mark Trailing Stop with asset-specific volatility adaptation.
        
        Returns: (updated_stop_loss, was_ratcheted_up)
        """
        if not self.config.enable_trailing_stop or current_price <= 0 or entry_price <= 0:
            return current_stop_loss, False

        # 1. Update high-water mark peak and trough price
        norm_sym = normalize_symbol(symbol)
        peak = max(self._peak_prices.get(norm_sym, self._peak_prices.get(symbol, entry_price)), current_price)
        self._peak_prices[norm_sym] = peak
        self._peak_prices[symbol] = peak
        trough = min(self._trough_prices.get(norm_sym, self._trough_prices.get(symbol, entry_price)), current_price)
        self._trough_prices[norm_sym] = trough
        self._trough_prices[symbol] = trough

        # 2. Check Freqtrade Trailing Stop & Break-Even Ratchet
        unrealized_profit_pct = ((peak - entry_price) / entry_price) * 100.0

        # Stage 1: Dynamic Break-Even Protection (adapted per asset volatility)
        bench = norm_sym.replace("/", "-")
        from bot.risk.regime_specialist import REGIME_SPECIALIST
        try:
            regime_analysis = REGIME_SPECIALIST.analyze_market_regime(bench)
            dynamic_be_pct = regime_analysis.break_even_pct
            sl_multiplier = getattr(regime_analysis, "sl_multiplier", 1.0)
        except Exception:
            dynamic_be_pct = 0.8
            sl_multiplier = 1.0

        dynamic_distance = self.config.trailing_stop_distance_pct
        from bot.risk.regime_specialist import MarketRegime
        if getattr(regime_analysis, "regime", None) == MarketRegime.HIGH_VOLATILITY_EXPANSION:
            dynamic_distance = round(self.config.trailing_stop_distance_pct * 1.25, 2)
        elif getattr(regime_analysis, "regime", None) == MarketRegime.DEFENSIVE_BEAR_BREAKDOWN:
            dynamic_distance = round(self.config.trailing_stop_distance_pct * 0.75, 2)

        candidate_sl = current_stop_loss
        was_updated = False

        # Stage 1: Break-even check
        break_even_sl = round(entry_price * 1.0015, 4)
        if unrealized_profit_pct >= dynamic_be_pct and break_even_sl > candidate_sl:
            candidate_sl = break_even_sl
            was_updated = True
            logger.info(
                "[BREAK_EVEN] 🛡️ %s en ganancia (+%.2f%% >= %.1f%% BE). Stop Loss ratcheted a Break-Even: $%.2f (Entrada: $%.2f)",
                symbol, unrealized_profit_pct, dynamic_be_pct, break_even_sl, entry_price
            )

        # Stage 2: Multi-Tier Profit Vault Ratchet (Absolute Floor Guarantees as peaks expand)
        if getattr(self.config, "enable_profit_vault_ratchet", True):
            vault_tiers = [
                (getattr(self.config, "vault_tier6_activation_pct", 12.0), getattr(self.config, "vault_tier6_floor_pct", 9.5), 6),
                (getattr(self.config, "vault_tier5_activation_pct", 8.5), getattr(self.config, "vault_tier5_floor_pct", 6.5), 5),
                (getattr(self.config, "vault_tier4_activation_pct", 6.0), getattr(self.config, "vault_tier4_floor_pct", 4.2), 4),
                (getattr(self.config, "vault_tier3_activation_pct", 4.0), getattr(self.config, "vault_tier3_floor_pct", 2.4), 3),
                (getattr(self.config, "vault_tier2_activation_pct", 2.5), getattr(self.config, "vault_tier2_floor_pct", 1.2), 2),
                (getattr(self.config, "vault_tier1_activation_pct", 1.5), getattr(self.config, "vault_tier1_floor_pct", 0.4), 1),
            ]
            for act_pct, floor_pct, tier_num in vault_tiers:
                if unrealized_profit_pct >= act_pct:
                    vault_sl = round(entry_price * (1.0 + (floor_pct / 100.0)), 4)
                    if vault_sl > candidate_sl:
                        candidate_sl = vault_sl
                        was_updated = True
                        logger.info(
                            "[PROFIT_VAULT] 🔒 %s Bóveda de Ganancias asegurada en Piso Nivel %d: Stop Loss ratcheted a $%.2f "
                            "(+%.2f%% garantizado sobre entrada $%.2f). Pico alcanzado: +%.2f%%",
                            symbol, tier_num, vault_sl, floor_pct, entry_price, unrealized_profit_pct,
                        )
                    break

        # Stage 3: Dynamic LEAN Trailing Stop (trailing high-water mark behind peak)
        if unrealized_profit_pct >= self.config.trailing_stop_activation_pct:
            trailing_sl = round(peak * (1.0 - (dynamic_distance / 100.0)), 4)
            if trailing_sl > candidate_sl:
                candidate_sl = trailing_sl
                was_updated = True
                logger.info(
                    "[TRAILING_STOP] 📈 %s Trailing Stop ratcheted a $%.2f (Peak: $%.2f, Dist: %.2f%%)",
                    symbol, candidate_sl, peak, dynamic_distance
                )

        if was_updated:
            self.save_active_positions_state()

        return candidate_sl, was_updated

    def evaluate(
        self,
        signal: TradeSignal,
        balance: AccountBalance,
        positions: list[Position],
    ) -> RiskValidationResult:
        """Run all safety and sizing checks on a trade signal."""
        rejection_reasons: list[str] = []
        warnings: list[str] = []

        now = time.time()

        # Freqtrade Protection: StoplossGuard
        recent_losses = len([t for t in self._recent_stoploss_events if now - t < 3600.0])
        if signal.action == OrderSide.BUY and recent_losses >= self.config.max_consecutive_stoplosses:
            rejection_reasons.append(
                f"Freqtrade StoplossGuard activo: {recent_losses} pérdidas en la última hora. "
                f"Compras pausadas para proteger capital frente a ruido de mercado."
            )
            return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

        # Freqtrade Protection: CooldownPeriod per asset / Extended Quarantine
        norm_sym = normalize_symbol(signal.symbol)
        in_cd, remaining_sec = self.is_in_cooldown(signal.symbol)
        if signal.action == OrderSide.BUY and in_cd:
            remaining_min = remaining_sec / 60.0
            rejection_reasons.append(
                f"Freqtrade CooldownPeriod activo para {signal.symbol}: "
                f"esperando {remaining_min:.1f}m antes de re-entrada tras pérdidas previas."
            )
            return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

        # 1. Action check
        if signal.action == OrderSide.HOLD:
            return RiskValidationResult(
                is_approved=False,
                rejection_reasons=["Trade signal action is HOLD; no execution required."],
            )

        # 2. Peak Equity Tracking & High-Water Mark Trailing Drawdown Protection
        if balance.equity > self._peak_equity:
            self._peak_equity = balance.equity

        if getattr(self.config, "enable_portfolio_trailing_drawdown_guard", True) and self._peak_equity > 0:
            trailing_dd_pct = ((self._peak_equity - balance.equity) / self._peak_equity) * 100.0
            max_dd_peak = getattr(self.config, "max_trailing_drawdown_from_peak_pct", 2.5)
            if trailing_dd_pct >= max_dd_peak and signal.action == OrderSide.BUY:
                rejection_reasons.append(
                    f"Portfolio Peak Drawdown Guard activo: retroceso del {trailing_dd_pct:.2f}% "
                    f"desde el pico histórico (${self._peak_equity:,.2f} -> ${balance.equity:,.2f}, máx permitido: {max_dd_peak:.1f}%). "
                    f"Compras bloqueadas para blindar beneficios acumulados."
                )
                return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

        # 2b. Minimum Cash Reserve Floor & Profit Quarantine Vault (50% Retained Profit Lockbox)
        min_cash_reserve, base_cash_reserve, quarantined_profit = self.calculate_protected_cash_reserve(balance)
        available_cash_reserve = max(0.0, balance.cash - min_cash_reserve)

        if quarantined_profit > 0:
            quarantine_pct = getattr(self.config, "profit_quarantine_ratio", 0.40) * 100
            warnings.append(
                f"[PROFIT_QUARANTINE] 🏦 Bóveda de Ganancias Activa: ${quarantined_profit:,.2f} "
                f"({quarantine_pct:.0f}% de ganancia neta) selladas de forma intocable en efectivo."
            )

        if signal.action == OrderSide.BUY and balance.cash <= min_cash_reserve:
            min_cash_pct = getattr(self.config, "min_cash_reserve_pct", 20.0)
            quarantine_pct = getattr(self.config, "profit_quarantine_ratio", 0.40) * 100
            rejection_reasons.append(
                f"Piso de Liquidez / Reserva de Efectivo activa: Efectivo actual (${balance.cash:,.2f}) "
                f"en o bajo la reserva protegida (${min_cash_reserve:,.2f} = base {min_cash_pct:.0f}% (${base_cash_reserve:,.2f}) + ${quarantined_profit:,.2f} en bóveda de ganancias {quarantine_pct:.0f}%). "
                f"Compras suspendidas para blindar las ganancias cosechadas."
            )
            return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

        # 2c. Kill-Switch check (daily drawdown) - Blocks new BUYs, but permits protective SELLs
        if balance.daily_drawdown_pct >= self.config.max_daily_drawdown_pct and signal.action == OrderSide.BUY:
            rejection_reasons.append(
                f"Kill-switch active: daily drawdown ({balance.daily_drawdown_pct:.2f}%) "
                f"exceeds limit ({self.config.max_daily_drawdown_pct:.2f}%). BUY orders prohibited."
            )
            return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons)

        # 3. Palanca A: Defensive Regime Veto (Strictly blocks BUYs during market crashes/defensive regimes)
        regime_val = signal.metrics.get("regime")
        alloc_factor = float(signal.metrics.get("allocation_factor", 1.0))
        if signal.action == OrderSide.BUY and getattr(self.config, "enable_defensive_regime_veto", True):
            if regime_val == "DEFENSIVE_BEAR_BREAKDOWN" or alloc_factor == 0.0:
                rejection_reasons.append(
                    f"Veto por Régimen Defensivo: Mercado en desplome/fase bajista correctiva "
                    f"(Régimen: {regime_val}, Alloc Factor: {alloc_factor:.1f}). Compras prohibidas para resguardar capital en efectivo."
                )
                return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

        # 4. Real-Time Market Research & Prudence Gatekeeper
        market_prudence = signal.metrics.get("market_prudence")
        prudence_directive = signal.metrics.get("prudence_directive") or "Condiciones adversas detectadas por investigación de mercado"
        prudence_multiplier = float(signal.metrics.get("prudence_multiplier", 1.0))
        if signal.action == OrderSide.BUY:
            if market_prudence == "IMPRUDENT_ADVERSE_VETO":
                rejection_reasons.append(
                    f"VETO DE INVESTIGACIÓN DE MERCADO: El subagente de investigación dictamina "
                    f"entorno IMPRUDENTE ({prudence_directive}). Capital protegido al 100% en liquidez."
                )
                return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)
            elif market_prudence == "CAUTION_HEADWINDS" or prudence_multiplier < 1.0:
                warnings.append(
                    f"ADVERTENCIA DE MERCADO: Subagente de investigación detecta turbulencia/viento en contra "
                    f"({prudence_directive}). Asignación prudencial ajustada al {prudence_multiplier*100:.0f}%."
                )

        # 5. Palanca C: Techo Anti-Trampas (BULL_TRAP_OVERBOUGHT: RSI >= 64.0)
        max_rsi_ceiling = getattr(self.config, "max_overbought_rsi_buy", 64.0)
        signal_rsi = float(signal.metrics.get("rsi", 50.0))
        if signal.action == OrderSide.BUY and signal_rsi >= max_rsi_ceiling:
            rejection_reasons.append(
                f"Veto Anti-Trampa (BULL_TRAP_OVERBOUGHT): Entrada en sobrecompra extrema (RSI {signal_rsi:.1f} >= {max_rsi_ceiling:.1f}). "
                f"Agotamiento comprador inminente; capital preservado."
            )
            return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

        # 6. Palanca B: Filtro Macro Institucional SMA-200 (Paul Tudor Jones Rule)
        sma200 = float(signal.metrics.get("sma200", 0.0))
        div_type = signal.metrics.get("divergence", "NONE")
        if signal.action == OrderSide.BUY and getattr(self.config, "enable_macro_sma200_filter", True) and sma200 > 0:
            if signal.current_price < sma200 and div_type != "BULLISH":
                rejection_reasons.append(
                    f"Filtro Macro SMA-200: Precio (${signal.current_price:.2f}) bajo media móvil de 200 períodos (${sma200:.2f}) "
                    f"sin divergencia alcista confirmada. Compras contra-tendencia bajista vetadas."
                )
                return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

        # 7. High-Conviction Institutional Quality Gate
        min_required_conf = getattr(self.config, "min_signal_confidence", 0.0)
        if signal.action == OrderSide.BUY and min_required_conf > 0 and signal.confidence < min_required_conf:
            rejection_reasons.append(
                f"Quality Gate: Convicción ({signal.confidence*100:.0f}%) inferior al umbral institucional "
                f"mínimo ({min_required_conf*100:.0f}%). Solo se ejecutan operaciones A+ de alta probabilidad."
            )
            return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

        # 8. Correlated Crypto Exposure Gatekeeper (Max concurrent crypto positions & Aggregate Cap)
        if signal.action == OrderSide.BUY and signal.asset_class == AssetClass.CRYPTO:
            max_crypto_pos = getattr(self.config, "max_concurrent_crypto_positions", 2)
            crypto_positions_count = sum(1 for p in positions if p.asset_class == AssetClass.CRYPTO and p.quantity > 0)
            existing_pos = next((p for p in positions if normalize_symbol(p.symbol) == norm_sym), None)
            if not existing_pos and crypto_positions_count >= max_crypto_pos:
                rejection_reasons.append(
                    f"Límite de correlación cripto alcanzado: {crypto_positions_count} activos en cartera "
                    f"(máx: {max_crypto_pos}). Capital preservado contra riesgo sistémico de mercado."
                )

            # Hard Aggregate Crypto Exposure Cap (e.g. max 25% or 30% of portfolio in crypto)
            max_crypto_cap_pct = getattr(self.config, "max_total_crypto_allocation_pct", 25.0)
            max_crypto_dollars = balance.equity * (max_crypto_cap_pct / 100.0)
            current_crypto_val = sum(
                p.market_value for p in positions if p.asset_class == AssetClass.CRYPTO and p.quantity > 0
            )
            in_flight_crypto_val = self.get_in_flight_capital(asset_class=AssetClass.CRYPTO)
            total_active_crypto = current_crypto_val + in_flight_crypto_val

            if total_active_crypto >= max_crypto_dollars:
                rejection_reasons.append(
                    f"Techo de Exposición Agregada Cripto alcanzado: ${total_active_crypto:,.2f} en cripto "
                    f"({(total_active_crypto / balance.equity) * 100.0:.1f}% >= {max_crypto_cap_pct:.1f}% máx de ${max_crypto_dollars:,.2f}). "
                    f"Portafolio protegido contra sobre-concentración y volatilidad sistémica."
                )

            if rejection_reasons:
                return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

        # 8a. Spread Guard: Microstructure Slippage & Flash Spread Defense
        spread_pct = float(signal.metrics.get("spread_pct", 0.0))
        max_allowed_spread = getattr(self.config, "max_allowed_spread_pct", 0.45)
        if signal.action == OrderSide.BUY and getattr(self.config, "enable_spread_guard", True) and spread_pct > max_allowed_spread:
            rejection_reasons.append(
                f"[SPREAD_GUARD] 🛡️ Spread Bid/Ask excesivo ({spread_pct:.2f}% > {max_allowed_spread:.2f}% máx). "
                f"Entrada bloqueada para proteger margen contra creadores de mercado."
            )
            return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)
        elif spread_pct > 0 and signal.action == OrderSide.BUY:
            warnings.append(f"[SPREAD_GUARD] 🔍 Spread Bid/Ask verificado seguro ({spread_pct:.2f}% <= {max_allowed_spread:.2f}%).")

        # 8b. Meta-Labeling Binary Quality Filter (Marcos López de Prado)
        if signal.action == OrderSide.BUY and getattr(self.config, "enable_meta_labeling", True):
            is_meta_ok, p_success, meta_reason = self.evaluate_meta_labeling(
                signal=signal,
                market_regime=str(signal.metrics.get("regime") or "UNKNOWN"),
            )
            if not is_meta_ok:
                rejection_reasons.append(meta_reason)
                return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)
            else:
                warnings.append(f"[META_LABELING] 🎯 Filtro de Precisión superado: {meta_reason}")

        # 9. Pattern Learning & Mistake Veto (ML4T Continuous Feedback)
        pattern_multiplier = 1.0
        if signal.action == OrderSide.BUY and self.pattern_memory:
            # Check for regime drift / loss fatigue
            drift = self.pattern_memory.detect_regime_drift()
            if drift.get("drift_detected"):
                if signal.confidence < 0.80:
                    rejection_reasons.append(
                        f"Régimen Adverso / Deriva de Alpha detectada ({drift.get('recommendation')}). "
                        f"Se requiere convicción >= 80% (actual: {signal.confidence * 100:.0f}%)."
                    )
                    return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)
                else:
                    warnings.append("Operando bajo Régimen Defensivo: Convicción alta validada.")

            # Check matching mistake patterns with REAL technical metrics from signal
            real_rsi = float(signal.metrics.get("rsi", 55.0))
            real_atr = float(signal.metrics.get("atr", signal.current_price * 0.02))
            real_atr_pct = (real_atr / signal.current_price * 100.0) if signal.current_price > 0 else 2.0
            real_ema_ratio = float(signal.metrics.get("ema_ratio", 1.01))

            pattern_check = self.pattern_memory.evaluate_setup(
                symbol=signal.symbol,
                asset_class=signal.asset_class,
                rsi=real_rsi,
                ema_ratio=real_ema_ratio,
                atr_pct=real_atr_pct,
            )
            if not pattern_check.is_safe:
                rejection_reasons.append(
                    f"VETO DE APRENDIZAJE: {pattern_check.warning_message}. "
                    f"Lección: '{pattern_check.learned_lesson}'"
                )
                return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)
            elif pattern_check.risk_multiplier < 1.0:
                pattern_multiplier = pattern_check.risk_multiplier
                warnings.append(pattern_check.warning_message or f"Asignación reducida al {pattern_multiplier*100:.0f}% por similitud con patrón fallido.")

        # 3. Position check for SELL orders and allocation limits
        existing_position = next((p for p in positions if normalize_symbol(p.symbol) == norm_sym), None)
        if signal.action == OrderSide.SELL:
            if not existing_position or existing_position.quantity <= 0:
                rejection_reasons.append(f"Cannot execute SELL for {signal.symbol}: no open position held.")
                return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons)
            
            # Close existing position
            return RiskValidationResult(
                is_approved=True,
                adjusted_quantity=existing_position.quantity,
                allocated_capital=existing_position.quantity * signal.current_price,
            )

        # 4. BUY Validation: Stop Loss verification & sanitization
        current_price = signal.current_price
        if current_price <= 0:
            rejection_reasons.append(f"Invalid current price for {signal.symbol}: {current_price}")
            return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons)

        stop_loss = signal.stop_loss
        default_sl_distance = current_price * (self.config.default_stop_loss_pct / 100.0)
        
        if stop_loss is None or stop_loss >= current_price or stop_loss <= 0:
            stop_loss = current_price - default_sl_distance
            warnings.append(
                f"LLM stop loss missing or invalid ({signal.stop_loss}). "
                f"Applied default {self.config.default_stop_loss_pct}% stop: ${stop_loss:,.2f}"
            )

        risk_per_share = current_price - stop_loss
        risk_pct = (risk_per_share / current_price) * 100.0

        if risk_pct > self.config.max_stop_loss_pct:
            stop_loss = current_price * (1 - self.config.max_stop_loss_pct / 100.0)
            risk_per_share = current_price - stop_loss
            warnings.append(
                f"Requested stop loss too wide ({risk_pct:.1f}%). "
                f"Clamped to max allowable {self.config.max_stop_loss_pct}%: ${stop_loss:,.2f}"
            )

        # 5. Position Sizing
        # a) Max capital allowed to risk per trade (e.g. 1.5% of total equity)
        max_risk_dollars = balance.equity * (self.config.max_risk_per_trade_pct / 100.0)
        qty_by_risk = max_risk_dollars / risk_per_share

        # b) Fractional Kelly Sizing: scale allocation between 4% (conviction 0.5) and 10% (conviction >= 0.85)
        conviction = max(0.5, min(1.0, signal.confidence))
        alloc_scale = 0.4 + 0.6 * ((conviction - 0.5) / 0.5)
        effective_alloc_pct = self.config.max_position_size_pct * alloc_scale

        # Free-Roll Winner Pyramiding: permit controlled compounding into confirmed winning positions
        is_free_roll_pyramiding = False
        if existing_position and existing_position.quantity > 0 and getattr(self.config, "enable_winner_pyramiding", True):
            existing_unrealized_profit_pct = (
                ((current_price - existing_position.entry_price) / existing_position.entry_price) * 100.0
            ) if existing_position.entry_price > 0 else 0.0
            existing_sl = existing_position.stop_loss_price or 0.0
            is_stop_locked_in_profit = existing_sl >= existing_position.entry_price * 1.002
            min_pyramid_profit = getattr(self.config, "min_profit_pct_for_pyramiding", 2.8)

            if existing_unrealized_profit_pct >= min_pyramid_profit and is_stop_locked_in_profit and conviction >= 0.70:
                is_free_roll_pyramiding = True
                boost = getattr(self.config, "pyramiding_max_allocation_boost", 1.35)
                effective_alloc_pct = effective_alloc_pct * boost
                warnings.append(
                    f"[PYRAMIDING_ACTIVE] 🚀 Posición ganadora en {signal.symbol} (+{existing_unrealized_profit_pct:.2f}% "
                    f"con Stop asegurado en ${existing_sl:,.2f} > Entrada ${existing_position.entry_price:,.2f}). "
                    f"Escalando ganadores (cap ampliado a {effective_alloc_pct:.1f}%)."
                )

        max_capital_allocation = balance.equity * (effective_alloc_pct / 100.0)
        
        # Deduct already invested capital in this asset + in-flight pending orders
        existing_val = existing_position.market_value if existing_position else 0.0
        in_flight_sym_val = self.get_in_flight_capital(symbol=signal.symbol)
        total_sym_exposure = existing_val + in_flight_sym_val

        if total_sym_exposure >= max_capital_allocation:
            rejection_reasons.append(
                f"Position allocation limit reached for {signal.symbol}: "
                f"currently holding/pending ${total_sym_exposure:,.2f} (max permitted: ${max_capital_allocation:,.2f}). "
                f"Capital preserved against over-concentration."
            )
            return RiskValidationResult(
                is_approved=False,
                rejection_reasons=rejection_reasons,
                risk_warnings=warnings,
            )

        remaining_alloc = max(0.0, max_capital_allocation - total_sym_exposure)

        # For crypto, further constrain remaining allocation by available room under aggregate crypto cap
        if signal.asset_class == AssetClass.CRYPTO:
            max_crypto_cap_pct = getattr(self.config, "max_total_crypto_allocation_pct", 30.0)
            max_crypto_dollars = balance.equity * (max_crypto_cap_pct / 100.0)
            current_crypto_val = sum(
                p.market_value for p in positions if p.asset_class == AssetClass.CRYPTO and p.quantity > 0
            )
            in_flight_crypto_val = self.get_in_flight_capital(asset_class=AssetClass.CRYPTO)
            remaining_crypto_room = max(0.0, max_crypto_dollars - (current_crypto_val + in_flight_crypto_val))
            remaining_alloc = min(remaining_alloc, remaining_crypto_room)

        if total_sym_exposure > 0 and remaining_alloc < 50.0:
            rejection_reasons.append(
                f"Posición para {signal.symbol} ya completada (${total_sym_exposure:,.2f} de ${max_capital_allocation:,.2f} máx). "
                f"Capital preservado contra sobre-exposición."
            )
            return RiskValidationResult(
                is_approved=False,
                rejection_reasons=rejection_reasons,
                risk_warnings=warnings,
            )

        qty_by_alloc = remaining_alloc / current_price

        # c) Available liquidity (Bounded by Cash Reserve Floor, Buying Power, and Venue Cash in Hybrid mode)
        available_liquidity = available_cash_reserve
        if signal.asset_class == AssetClass.EQUITY:
            if balance.buying_power > 0:
                available_liquidity = min(available_liquidity, balance.buying_power)
            if balance.equity_cash is not None:
                available_liquidity = min(available_liquidity, balance.equity_cash)
        elif signal.asset_class == AssetClass.CRYPTO:
            if balance.crypto_cash is not None:
                available_liquidity = min(available_liquidity, balance.crypto_cash)

        # Apply 2% safety buffer for slippage & fees
        safe_liquidity = max(0.0, available_liquidity * 0.98)
        qty_by_cash = safe_liquidity / current_price


        # Palanca D: Order Book Imbalance / Level 2 Depth Adjustment
        l2_multiplier = 1.0
        if getattr(self.config, "enable_order_book_imbalance", True) and signal.metrics:
            l2_imbalance = float(signal.metrics.get("l2_imbalance_ratio", 1.0))
            if l2_imbalance <= getattr(self.config, "max_seller_overhang_ratio", 0.60):
                l2_multiplier = 0.55  # Clamp capital allocation by 45% when heavy seller wall overhangs
                warnings.append(
                    f"[L2_DEPTH] 🧱 Muro vendedor pesado detectado ({l2_imbalance:.2f}x). "
                    f"Asignación ajustada preventivamente al 55% para no comprar contra pared de oferta."
                )
            elif l2_imbalance >= getattr(self.config, "min_order_book_imbalance_ratio", 1.25):
                warnings.append(
                    f"[L2_DEPTH] 🚀 Absorción compradora confirmada en L2 ({l2_imbalance:.2f}x bids vs asks)."
                )

        regime_sizing_factor = alloc_factor if 0.0 < alloc_factor < 1.0 else 1.0

        # Portfolio Intelligence: Check overall portfolio health and circuit breakers
        portfolio_health_mult = 1.0
        asset_alloc_mult = 1.0
        try:
            from bot.learning.portfolio_intelligence import PORTFOLIO_INTELLIGENCE
            if hasattr(self, "pattern_memory") and self.pattern_memory:
                records = self.pattern_memory.get_historical_feature_matrix(limit=50)
                if records:
                    health = PORTFOLIO_INTELLIGENCE.compute_metrics(records)
                    portfolio_health_mult = health.sizing_multiplier
                    if health.health_status == "HALT_RECOMMENDED":
                        halt_msg = health.warnings[0] if health.warnings else "Riesgo excesivo de cartera"
                        rejection_reasons.append(
                            f"Circuit Breaker de Cartera: Salud crítica ({halt_msg}). Operaciones pausadas preventivamente."
                        )
                        return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)
                    elif health.health_status == "DEGRADED":
                        warnings.append(f"[PORTFOLIO_INTELLIGENCE] ⚠️ Rendimiento degradado: Sizing global al {portfolio_health_mult * 100:.0f}%.")

                    asset_alloc_mult = PORTFOLIO_INTELLIGENCE.get_symbol_allocation_multiplier(signal.symbol)
                    if asset_alloc_mult > 1.0:
                        warnings.append(f"[PORTFOLIO_INTELLIGENCE] 🚀 Alpha Edge confirmado para {signal.symbol}: Asignación amplificada {asset_alloc_mult:.2f}x.")
                    elif asset_alloc_mult < 1.0:
                        warnings.append(f"[PORTFOLIO_INTELLIGENCE] 🛡️ Drawdown histórico en {signal.symbol}: Asignación reducida a {asset_alloc_mult:.2f}x.")
        except Exception as pie:
            logger.debug("[PORTFOLIO_INTELLIGENCE] Notice en evaluación: %s", pie)

        target_qty = (
            min(qty_by_risk, qty_by_alloc, qty_by_cash)
            * pattern_multiplier
            * regime_sizing_factor
            * prudence_multiplier
            * l2_multiplier
            * portfolio_health_mult
            * asset_alloc_mult
        )

        # Quantize shares (Equities: integer shares; Crypto: up to 6 decimals)
        if signal.asset_class == AssetClass.EQUITY:
            final_qty = math.floor(target_qty)
        else:
            final_qty = round(target_qty, 6)

        allocated_capital = final_qty * current_price

        if final_qty <= 0 or allocated_capital < 10.0:  # Minimum $10 trade
            rejection_reasons.append(
                f"Calculated position size is negligible or below $10 minimum "
                f"(Liquid Capital: ${available_liquidity:,.2f}, Max Alloc: ${max_capital_allocation:,.2f}, Target qty: {target_qty:.4f})"
            )
            return RiskValidationResult(
                is_approved=False,
                rejection_reasons=rejection_reasons,
                risk_warnings=warnings,
            )

        # 6. Asymmetrical Risk/Reward Enforcement & Palanca C Adaptive ATR Targets
        take_profit = signal.target_price
        min_reward = risk_per_share * self.config.min_risk_reward_ratio
        min_take_profit = round(current_price + min_reward, 4)

        # Palanca C: Elastic target calculation using ATR%
        atr_pct = float(signal.metrics.get("atr_pct", 0.0))
        if atr_pct <= 0.0 and signal.metrics.get("atr"):
            atr_val = float(signal.metrics.get("atr", 0.0))
            atr_pct = (atr_val / current_price * 100.0) if current_price > 0 else 0.0

        elastic_take_profit = min_take_profit
        if getattr(self.config, "enable_adaptive_atr_targets", True) and atr_pct > 0.0:
            atr_target_dist = current_price * ((atr_pct * getattr(self.config, "atr_tp_multiplier_stage2", 3.0)) / 100.0)
            elastic_take_profit = max(min_take_profit, round(current_price + atr_target_dist, 4))

        if take_profit is None or take_profit <= current_price:
            take_profit = elastic_take_profit
            warnings.append(
                f"Take-Profit calibrado asimétricamente con ATR ({atr_pct:.1f}% ATR) a ${take_profit:,.2f}"
            )
        else:
            reward = take_profit - current_price
            rr_ratio = reward / risk_per_share
            if rr_ratio < self.config.min_risk_reward_ratio:
                take_profit = min_take_profit
                warnings.append(
                    f"R:R original ({rr_ratio:.2f}:1) elevado asimétricamente a {self.config.min_risk_reward_ratio:.1f}:1 (${take_profit:,.2f}) para maximizar margen de ganancia."
                )
            if take_profit < elastic_take_profit:
                warnings.append(
                    f"Take-Profit ampliado con Palanca C (ATR {atr_pct:.1f}%): ${take_profit:,.2f} -> ${elastic_take_profit:,.2f}"
                )
                take_profit = elastic_take_profit

        final_sl = round(stop_loss, 4)
        final_tp = round(take_profit, 4) if take_profit else None

        # Register position target levels for persistent tracking and exit execution
        self.register_position_targets(
            symbol=signal.symbol,
            entry_price=current_price,
            stop_loss=final_sl,
            take_profit=final_tp,
        )
        entry_ctx = dict(signal.metrics or {})
        entry_ctx["atr_pct"] = atr_pct
        self.register_entry_context(signal.symbol, entry_ctx)

        return RiskValidationResult(
            is_approved=True,
            adjusted_quantity=final_qty,
            allocated_capital=allocated_capital,
            stop_loss_price=final_sl,
            take_profit_price=final_tp,
            risk_warnings=warnings,
        )

