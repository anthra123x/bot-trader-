"""Institutional Execution Guard: Reverse-Engineered from Hummingbot & Freqtrade.

Implements institutional-grade quantitative execution controls:
1. SpreadGuard (Hummingbot): Rejects orders when bid-ask spread is too wide, protecting against illiquidity and toxic order flow.
2. PriceDriftGuard (Hummingbot): Blocks execution if live market price has drifted significantly from the screened price.
3. ConsecutiveLossCooldownGuard (Freqtrade): Enforces lockout periods on assets that suffer sequential stop-outs.
4. OrderLifecycleTracker (Hummingbot): Deterministic state machine governing order progression:
   PENDING -> SUBMITTED -> FILLED / REJECTED / CANCELLED -> EXITED.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from bot.models import OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class OrderState(StrEnum):
    PENDING_CREATE = "PENDING_CREATE"
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXITED = "EXITED"


@dataclass
class GuardEvaluationResult:
    """Result of institutional pre-flight execution evaluation."""
    is_approved: bool
    veto_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class InstitutionalExecutionGuard:
    """Institutional execution safety layer inspired by Hummingbot & Freqtrade."""

    def __init__(
        self,
        max_spread_pct: float = 0.50,          # 0.50% max spread (Hummingbot standard)
        max_price_drift_pct: float = 1.25,     # 1.25% max drift between screen and execution
        cooldown_seconds: int = 1800,          # 30-minute cooldown on sequential stop-loss (Freqtrade)
        max_consecutive_losses: int = 2,       # Max consecutive losses before extended lockout
        extended_cooldown_seconds: int = 3600, # 1-hour extended lockout
        risk_engine: Any = None,
    ):
        self.max_spread_pct = max_spread_pct
        self.max_price_drift_pct = max_price_drift_pct
        self.cooldown_seconds = cooldown_seconds
        self.max_consecutive_losses = max_consecutive_losses
        self.extended_cooldown_seconds = extended_cooldown_seconds
        self.risk_engine = risk_engine

        # Internal state tracking (fallback if risk_engine is not provided)
        self._symbol_cooldowns: dict[str, float] = {}
        self._consecutive_losses: dict[str, int] = {}
        self._order_states: dict[str, OrderState] = {}
        self._order_state_history: dict[str, list[tuple[OrderState, float]]] = {}

    def _normalize(self, symbol: str) -> str:
        return symbol.upper().replace("/", "-").strip()

    # --------------------------------------------------------------------------
    # 1. SPREAD & DRIFT GUARDS (Hummingbot)
    # --------------------------------------------------------------------------

    def evaluate_spread(
        self,
        symbol: str,
        bid_price: float,
        ask_price: float,
    ) -> tuple[bool, float, str | None]:
        """Verify that current market spread does not exceed institutional tolerance."""
        if bid_price <= 0 or ask_price <= 0 or ask_price < bid_price:
            return False, 0.0, f"Invalid quote prices for {symbol}: Bid ${bid_price:.4f}, Ask ${ask_price:.4f}"

        mid_price = (bid_price + ask_price) / 2.0
        spread_dollars = ask_price - bid_price
        spread_pct = (spread_dollars / mid_price) * 100.0

        if spread_pct > self.max_spread_pct:
            return (
                False,
                spread_pct,
                f"Hummingbot SpreadGuard VETO: Spread for {symbol} ({spread_pct:.2f}%) "
                f"exceeds institutional limit ({self.max_spread_pct:.2f}%). Execution blocked.",
            )
        return True, spread_pct, None

    def evaluate_price_drift(
        self,
        symbol: str,
        screen_price: float,
        current_price: float,
        side: OrderSide = OrderSide.BUY,
    ) -> tuple[bool, float, str | None]:
        """Check if execution price drifted adversely away from the screening baseline."""
        if screen_price <= 0 or current_price <= 0:
            return True, 0.0, None

        drift_pct = abs(current_price - screen_price) / screen_price * 100.0
        is_adverse = (side == OrderSide.BUY and current_price > screen_price) or (
            side == OrderSide.SELL and current_price < screen_price
        )

        if is_adverse and drift_pct > self.max_price_drift_pct:
            return (
                False,
                drift_pct,
                f"Hummingbot PriceDriftGuard VETO: Price for {symbol} drifted adversely by "
                f"{drift_pct:.2f}% (${screen_price:.4f} -> ${current_price:.4f}) beyond "
                f"tolerance ({self.max_price_drift_pct:.2f}%).",
            )
        return True, drift_pct, None

    # --------------------------------------------------------------------------
    # 2. COOLDOWN & LOSS LOCKOUT (Freqtrade)
    # --------------------------------------------------------------------------

    def register_trade_outcome(self, symbol: str, is_loss: bool) -> None:
        """Register completed trade outcome for dynamic Freqtrade cooldown."""
        if self.risk_engine and hasattr(self.risk_engine, "register_trade_exit"):
            self.risk_engine.register_trade_exit(symbol, is_loss=is_loss)

        norm = self._normalize(symbol)
        now = time.time()
        if is_loss:
            losses = self._consecutive_losses.get(norm, 0) + 1
            self._consecutive_losses[norm] = losses
            cooldown = self.extended_cooldown_seconds if losses >= self.max_consecutive_losses else self.cooldown_seconds
            self._symbol_cooldowns[norm] = now + cooldown
            logger.warning(
                "[INSTITUTIONAL_GUARD] 🛑 Trade loss recorded for %s (streak: %d). Cooldown applied: %.1f minutes.",
                norm, losses, cooldown / 60.0,
            )
        else:
            self._consecutive_losses[norm] = 0
            self._symbol_cooldowns.pop(norm, None)
            logger.info("[INSTITUTIONAL_GUARD] ✅ Win recorded for %s. Cooldown reset.", norm)

    def is_in_cooldown(self, symbol: str) -> tuple[bool, float]:
        """Check if symbol is currently in an active Freqtrade cooldown window."""
        # Single authoritative source of truth: delegate to RiskEngine when available
        if self.risk_engine and hasattr(self.risk_engine, "is_in_cooldown"):
            return self.risk_engine.is_in_cooldown(symbol)

        norm = self._normalize(symbol)
        now = time.time()
        expiry = self._symbol_cooldowns.get(norm, 0.0)
        if now < expiry:
            remaining_seconds = expiry - now
            return True, remaining_seconds
        return False, 0.0

    # --------------------------------------------------------------------------
    # 3. COMPREHENSIVE PRE-FLIGHT EVALUATION
    # --------------------------------------------------------------------------

    def evaluate_preflight(
        self,
        symbol: str,
        side: OrderSide,
        current_price: float,
        screen_price: float | None = None,
        bid_price: float | None = None,
        ask_price: float | None = None,
    ) -> GuardEvaluationResult:
        """Run complete institutional pre-flight validation suite."""
        vetoes: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}

        # 1. Cooldown Check
        in_cd, remaining = self.is_in_cooldown(symbol)
        if in_cd and side == OrderSide.BUY:
            vetoes.append(
                f"Freqtrade CooldownGuard VETO: {symbol} in active loss lockout. "
                f"Remaining: {remaining / 60.0:.1f} minutes."
            )
        metrics["cooldown_remaining_sec"] = remaining

        # 2. Price Drift Check
        if screen_price and screen_price > 0:
            ok_drift, drift_pct, drift_err = self.evaluate_price_drift(symbol, screen_price, current_price, side)
            metrics["price_drift_pct"] = drift_pct
            if not ok_drift and drift_err:
                vetoes.append(drift_err)

        # 3. Spread Check
        if bid_price and ask_price and bid_price > 0 and ask_price > 0:
            ok_spread, spread_pct, spread_err = self.evaluate_spread(symbol, bid_price, ask_price)
            metrics["spread_pct"] = spread_pct
            if not ok_spread and spread_err:
                vetoes.append(spread_err)

        return GuardEvaluationResult(
            is_approved=len(vetoes) == 0,
            veto_reasons=vetoes,
            warnings=warnings,
            metrics=metrics,
        )

    # --------------------------------------------------------------------------
    # 4. ORDER LIFECYCLE STATE MACHINE (Hummingbot)
    # --------------------------------------------------------------------------

    def transition_order_state(self, order_id: str, new_state: OrderState) -> bool:
        """Advance an order state with strict Hummingbot lifecycle invariants."""
        current = self._order_states.get(order_id, OrderState.PENDING_CREATE)

        # State transition validation table
        valid_transitions: dict[OrderState, set[OrderState]] = {
            OrderState.PENDING_CREATE: {OrderState.SUBMITTED, OrderState.REJECTED, OrderState.CANCELLED},
            OrderState.SUBMITTED: {OrderState.OPEN, OrderState.FILLED, OrderState.PARTIALLY_FILLED, OrderState.REJECTED, OrderState.CANCELLED},
            OrderState.OPEN: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED},
            OrderState.PARTIALLY_FILLED: {OrderState.FILLED, OrderState.CANCELLED},
            OrderState.FILLED: {OrderState.EXITED},
            OrderState.CANCELLED: set(),
            OrderState.REJECTED: set(),
            OrderState.EXITED: set(),
        }

        if new_state not in valid_transitions.get(current, set()):
            logger.warning(
                "[ORDER_LIFECYCLE] Invalid state transition for order %s: %s -> %s",
                order_id, current, new_state,
            )
            return False

        self._order_states[order_id] = new_state
        if order_id not in self._order_state_history:
            self._order_state_history[order_id] = []
        self._order_state_history[order_id].append((new_state, time.time()))
        return True

    def get_order_state(self, order_id: str) -> OrderState:
        return self._order_states.get(order_id, OrderState.PENDING_CREATE)
