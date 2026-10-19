import logging
import math
import time
from typing import Any
from bot.config import BotRiskConfig
from bot.learning.pattern_memory import PatternMemory
from bot.models import AccountBalance, AssetClass, OrderSide, Position, RiskValidationResult, TradeSignal

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
        # Freqtrade: Symbol cooldown tracker after exits
        self._symbol_cooldowns: dict[str, float] = {}
        # Freqtrade: Rolling stoploss guard timestamps
        self._recent_stoploss_events: list[float] = []
        # Tiered Partial Take-Profit tracker
        self._partial_tp_executed: set[str] = set()
        self._partial_tp_stages: dict[str, int] = {}
        # Persistent position targets tracker (Take-Profit & Stop-Loss)
        self._position_targets: dict[str, dict[str, float]] = {}
        # Entry context tracking (RSI, EMA, ATR, Regime) for post-mortem learning
        self.entry_contexts: dict[str, dict[str, Any]] = {}

    def register_entry_context(self, symbol: str, context: dict[str, Any]) -> None:
        """Register technical metrics at entry for post-trade feedback and learning."""
        norm_sym = normalize_symbol(symbol)
        self.entry_contexts[norm_sym] = context
        self.entry_contexts[symbol] = context

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

        self._position_targets[norm_sym] = {
            "entry_price": entry_price,
            "stop_loss": fallback_sl,
            "take_profit": fallback_tp,
        }
        return self._position_targets[norm_sym]

    def update_position_stop(self, symbol: str, new_stop: float) -> None:
        """Update ratcheted stop loss for an open position."""
        norm_sym = normalize_symbol(symbol)
        if norm_sym in self._position_targets:
            curr_sl = self._position_targets[norm_sym]["stop_loss"]
            if new_stop > curr_sl:
                self._position_targets[norm_sym]["stop_loss"] = new_stop

    def register_trade_exit(self, symbol: str, is_loss: bool = False) -> None:
        """Register trade exit to activate Freqtrade CooldownPeriod and StoplossGuard."""
        now = time.time()
        norm_sym = normalize_symbol(symbol)
        self._peak_prices.pop(norm_sym, None)
        self._peak_prices.pop(symbol, None)
        self._partial_tp_executed.discard(norm_sym)
        self._partial_tp_executed.discard(symbol)
        self._partial_tp_stages.pop(norm_sym, None)
        self._partial_tp_stages.pop(symbol, None)
        self._position_targets.pop(norm_sym, None)
        self._position_targets.pop(symbol, None)
        self.entry_contexts.pop(norm_sym, None)
        self.entry_contexts.pop(symbol, None)
        if is_loss:
            self._symbol_cooldowns[norm_sym] = now
            self._recent_stoploss_events.append(now)
            # Prune events older than 1 hour
            self._recent_stoploss_events = [t for t in self._recent_stoploss_events if now - t < 3600.0]


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

        # Stage 1: Initial scale-out threshold (+1.2%)
        if stage == 0 and unrealized_profit_pct >= self.config.partial_take_profit_threshold_pct:
            partial_qty = round(current_qty * self.config.partial_take_profit_ratio, 6)
            if 0 < partial_qty < current_qty:
                return True, partial_qty

        # Stage 2: Secondary scale-out threshold (+3.0%)
        if stage == 1 and getattr(self.config, "enable_multi_stage_tp", True):
            stage2_thresh = getattr(self.config, "stage2_take_profit_threshold_pct", 3.0)
            stage2_ratio = getattr(self.config, "stage2_take_profit_ratio", 0.45)
            if unrealized_profit_pct >= stage2_thresh:
                stage2_qty = round(current_qty * stage2_ratio, 6)
                if 0 < stage2_qty < current_qty:
                    return True, stage2_qty

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

        # 1. Update high-water mark peak price
        norm_sym = normalize_symbol(symbol)
        peak = max(self._peak_prices.get(norm_sym, self._peak_prices.get(symbol, entry_price)), current_price)
        self._peak_prices[norm_sym] = peak
        self._peak_prices[symbol] = peak

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

        # Stage 2: Dynamic LEAN Trailing Stop (if higher than break-even / candidate_sl)
        if unrealized_profit_pct >= self.config.trailing_stop_activation_pct:
            trailing_sl = round(peak * (1.0 - (dynamic_distance / 100.0)), 4)
            if trailing_sl > candidate_sl:
                candidate_sl = trailing_sl
                was_updated = True
                logger.info(
                    "[TRAILING_STOP] 📈 %s Trailing Stop ratcheted a $%.2f (Peak: $%.2f, Dist: %.2f%%)",
                    symbol, candidate_sl, peak, dynamic_distance
                )

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

        # Freqtrade Protection: CooldownPeriod per asset
        norm_sym = normalize_symbol(signal.symbol)
        if signal.action == OrderSide.BUY and norm_sym in self._symbol_cooldowns:
            elapsed = now - self._symbol_cooldowns[norm_sym]
            if elapsed < self.config.cooldown_period_seconds:
                remaining_min = (self.config.cooldown_period_seconds - elapsed) / 60.0
                rejection_reasons.append(
                    f"Freqtrade CooldownPeriod activo para {signal.symbol}: "
                    f"esperando {remaining_min:.1f}m antes de re-entrada tras pérdida previa."
                )
                return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

        # 1. Action check
        if signal.action == OrderSide.HOLD:
            return RiskValidationResult(
                is_approved=False,
                rejection_reasons=["Trade signal action is HOLD; no execution required."],
            )

        # 2. Kill-Switch check (daily drawdown) - Blocks new BUYs, but permits protective SELLs
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

        # 8. Correlated Crypto Exposure Gatekeeper (Max concurrent crypto positions)
        if signal.action == OrderSide.BUY and signal.asset_class == AssetClass.CRYPTO:
            max_crypto_pos = getattr(self.config, "max_concurrent_crypto_positions", 4)
            crypto_positions_count = sum(1 for p in positions if p.asset_class == AssetClass.CRYPTO and p.quantity > 0)
            existing_pos = next((p for p in positions if normalize_symbol(p.symbol) == norm_sym), None)
            if not existing_pos and crypto_positions_count >= max_crypto_pos:
                rejection_reasons.append(
                    f"Límite de correlación cripto alcanzado: {crypto_positions_count} activos en cartera "
                    f"(máx: {max_crypto_pos}). Capital preservado contra riesgo sistémico de mercado."
                )
                return RiskValidationResult(is_approved=False, rejection_reasons=rejection_reasons, risk_warnings=warnings)

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
        max_capital_allocation = balance.equity * ((self.config.max_position_size_pct * alloc_scale) / 100.0)
        
        # Deduct already invested capital in this asset if any
        existing_val = existing_position.market_value if existing_position else 0.0
        if existing_val >= max_capital_allocation:
            rejection_reasons.append(
                f"Position allocation limit reached for {signal.symbol}: "
                f"currently holding ${existing_val:,.2f} (max permitted: ${max_capital_allocation:,.2f}). "
                f"Capital preserved against over-concentration."
            )
            return RiskValidationResult(
                is_approved=False,
                rejection_reasons=rejection_reasons,
                risk_warnings=warnings,
            )

        remaining_alloc = max(0.0, max_capital_allocation - existing_val)
        if existing_val > 0 and remaining_alloc < 50.0:
            rejection_reasons.append(
                f"Posición para {signal.symbol} ya completada (${existing_val:,.2f} de ${max_capital_allocation:,.2f} máx). "
                f"Capital preservado contra sobre-exposición."
            )
            return RiskValidationResult(
                is_approved=False,
                rejection_reasons=rejection_reasons,
                risk_warnings=warnings,
            )

        qty_by_alloc = remaining_alloc / current_price

        # c) Available liquidity (Bounded by Buying Power for equities to avoid broker rejection)
        available_liquidity = balance.cash
        if signal.asset_class == AssetClass.EQUITY and balance.buying_power > 0:
            available_liquidity = min(available_liquidity, balance.buying_power)
        # Apply 2% safety buffer for slippage & fees
        safe_liquidity = max(0.0, available_liquidity * 0.98)
        qty_by_cash = safe_liquidity / current_price

        regime_sizing_factor = alloc_factor if 0.0 < alloc_factor < 1.0 else 1.0
        target_qty = min(qty_by_risk, qty_by_alloc, qty_by_cash) * pattern_multiplier * regime_sizing_factor * prudence_multiplier

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

        # 6. Asymmetrical Risk/Reward Enforcement (Profit Expectancy >> Loss Expectancy)
        take_profit = signal.target_price
        min_reward = risk_per_share * self.config.min_risk_reward_ratio
        min_take_profit = round(current_price + min_reward, 4)

        if take_profit is None or take_profit <= current_price:
            take_profit = min_take_profit
            warnings.append(
                f"Take-Profit calibrado asimétricamente a R:R {self.config.min_risk_reward_ratio:.1f}:1 (${take_profit:,.2f})"
            )
        else:
            reward = take_profit - current_price
            rr_ratio = reward / risk_per_share
            if rr_ratio < self.config.min_risk_reward_ratio:
                take_profit = min_take_profit
                warnings.append(
                    f"R:R original ({rr_ratio:.2f}:1) elevado asimétricamente a {self.config.min_risk_reward_ratio:.1f}:1 (${take_profit:,.2f}) para maximizar margen de ganancia."
                )

        final_sl = round(stop_loss, 4)
        final_tp = round(take_profit, 4) if take_profit else None

        # Register position target levels for persistent tracking and exit execution
        self.register_position_targets(
            symbol=signal.symbol,
            entry_price=current_price,
            stop_loss=final_sl,
            take_profit=final_tp,
        )

        return RiskValidationResult(
            is_approved=True,
            adjusted_quantity=final_qty,
            allocated_capital=allocated_capital,
            stop_loss_price=final_sl,
            take_profit_price=final_tp,
            risk_warnings=warnings,
        )

