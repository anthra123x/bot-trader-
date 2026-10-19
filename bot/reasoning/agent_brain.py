"""AgentBrain (ReasoningEngine): Orchestrator of the REA-inspired Cognitive Trading Pipeline.

Implements the three-phase reverse-engineering workflow:
1. Decompile / Inspect: Reads unadulterated live portfolio and market telemetry.
2. Understand / Deliberate: Builds mathematical derivations, probabilistic inferences,
   and explicit invalidation conditions within an immutable Evidence Ledger.
3. Recreate / Execute: Controlled execution through strict risk engine constraints
   and broker dispatch with a persistent audit trail in Neon PostgreSQL.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from bot.brokers.base import BaseBroker
from bot.config import BOT_CONFIG, BotConfig
from bot.database.neon_repo import NeonRepository, get_neon_repo
from bot.models import (
    AccountBalance,
    AssetClass,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskValidationResult,
    TradeSignal,
)
from bot.reasoning.contracts import (
    CognitivePhase,
    EvidenceLedger,
    MathematicalDerivation,
    Observation,
    RiskLimitation,
    TradingDecision,
    TradingInference,
)
from bot.risk.institutional_guard import InstitutionalExecutionGuard, OrderState
from bot.risk.order_simulator import ExecutionSimulator
from bot.risk.quant_screener import QuantScreener, QuantScreenResult
from bot.risk.regime_specialist import REGIME_SPECIALIST, MarketPrudence
from bot.risk.risk_engine import RiskEngine

logger = logging.getLogger(__name__)


class AgentBrain:
    """Core Cognitive Adapter emulating advanced agent reasoning flows (Claude/Cursor/Gemini)."""

    def __init__(
        self,
        config: BotConfig | None = None,
        broker: BaseBroker | None = None,
        risk_engine: RiskEngine | None = None,
        db: NeonRepository | None = None,
        quant_screener: QuantScreener | None = None,
        institutional_guard: InstitutionalExecutionGuard | None = None,
        execution_simulator: ExecutionSimulator | None = None,
    ):
        self.config = config or BOT_CONFIG
        from bot.brokers import get_broker
        self.broker = broker or get_broker(self.config.broker_type)
        self.risk_engine = risk_engine or RiskEngine(self.config.risk)
        self.db = db or get_neon_repo()
        self.quant_screener = quant_screener or QuantScreener()
        self.institutional_guard = institutional_guard or InstitutionalExecutionGuard(
            cooldown_seconds=self.config.risk.cooldown_period_seconds,
        )
        self.execution_simulator = execution_simulator or ExecutionSimulator(
            min_risk_reward_ratio=2.0,
        )

    def identify_asset_class(self, symbol: str) -> AssetClass:
        """Categorize instrument into Equity or Crypto."""
        normalized = symbol.upper().strip()
        if (
            "-USD" in normalized
            or normalized in [s.upper() for s in self.config.watchlist_crypto]
            or normalized in ("BTC", "ETH", "SOL", "USDT")
        ):
            return AssetClass.CRYPTO
        return AssetClass.EQUITY

    # --------------------------------------------------------------------------
    # PHASE 1: DECOMPILE & INSPECT
    # --------------------------------------------------------------------------

    def decompile_state(self, symbol: str, trade_date: str | None = None) -> EvidenceLedger:
        """Decompile live account and market state into immutable, factual observations."""
        ledger = EvidenceLedger(
            symbol=symbol,
            phase=CognitivePhase.DECOMPILE,
            created_at=datetime.now(timezone.utc),
        )

        # 1. Decompile Account Telemetry
        try:
            balance: AccountBalance = self.broker.get_account_balance()
            positions = self.broker.get_positions()
            ledger.observations.append(
                Observation(
                    source="AlpacaBroker",
                    facet="account_balance",
                    data={
                        "cash": balance.cash,
                        "equity": balance.equity,
                        "buying_power": balance.buying_power,
                        "daily_drawdown_pct": balance.daily_drawdown_pct,
                    },
                )
            )
            ledger.observations.append(
                Observation(
                    source="AlpacaBroker",
                    facet="open_positions",
                    data={
                        "count": len(positions),
                        "positions": [
                            {
                                "symbol": p.symbol,
                                "quantity": p.quantity,
                                "entry_price": p.entry_price,
                                "current_price": p.current_price,
                                "unrealized_pnl": p.unrealized_pnl,
                            }
                            for p in positions
                        ],
                    },
                )
            )

            # Mathematical Derivation: 50% Profit Quarantine Vault
            protected_floor, _, _ = self.risk_engine.calculate_protected_cash_reserve(balance)
            free_cash = max(0.0, balance.cash - protected_floor)
            ledger.derivations.append(
                MathematicalDerivation(
                    metric_name="protected_cash_reserve",
                    formula="PeakEquity - (TotalProfit * 0.50)",
                    inputs={"equity": balance.equity, "cash": balance.cash},
                    value=protected_floor,
                    unit="USD",
                )
            )
            ledger.derivations.append(
                MathematicalDerivation(
                    metric_name="unrestricted_free_cash",
                    formula="max(0.0, Cash - ProtectedCashReserve)",
                    inputs={"cash": balance.cash, "protected_cash_reserve": protected_floor},
                    value=free_cash,
                    unit="USD",
                )
            )
        except Exception as e:
            logger.warning("[AGENT_BRAIN] Could not read live broker balance: %s", e)
            ledger.limitations.append(
                RiskLimitation(
                    factor="BrokerTelemetryUnavailable",
                    severity="HIGH",
                    mitigation="Fallback to zero free cash until broker responds",
                    source_warning=str(e),
                )
            )

        # 2. Decompile Market Telemetry & Quantitative Setup
        try:
            screen_res: QuantScreenResult = self.quant_screener.screen(symbol, trade_date)
            ledger.observations.append(
                Observation(
                    source="QuantScreener",
                    facet="quant_metrics",
                    data={
                        "setup_name": screen_res.setup_name,
                        "bias": screen_res.bias,
                        "score": screen_res.score,
                        "current_price": screen_res.current_price,
                        "rsi": screen_res.metrics.get("rsi"),
                        "atr": screen_res.metrics.get("atr"),
                        "band_pct": screen_res.metrics.get("band_pct"),
                        "suggested_sl": screen_res.suggested_stop_loss,
                        "suggested_tp": screen_res.suggested_take_profit,
                    },
                )
            )

            # Mathematical Derivation: Risk-to-Reward Ratio
            if screen_res.suggested_stop_loss and screen_res.suggested_take_profit and screen_res.current_price > 0:
                sl_dist = abs(screen_res.current_price - screen_res.suggested_stop_loss)
                tp_dist = abs(screen_res.suggested_take_profit - screen_res.current_price)
                rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0
                ledger.derivations.append(
                    MathematicalDerivation(
                        metric_name="risk_reward_ratio",
                        formula="(TakeProfit - CurrentPrice) / (CurrentPrice - StopLoss)",
                        inputs={"current_price": screen_res.current_price, "sl": screen_res.suggested_stop_loss, "tp": screen_res.suggested_take_profit},
                        value=rr,
                        unit="ratio",
                    )
                )
        except Exception as e:
            logger.debug("[AGENT_BRAIN] Quant screening notice for %s: %s", symbol, e)

        # 3. Decompile Macroeconomic Regime & Market Research Telemetry
        try:
            reg_obj = REGIME_SPECIALIST.analyze_market_regime(symbol.replace("/", "-"))
            ledger.observations.append(
                Observation(
                    source="RegimeSpecialist",
                    facet="macro_regime",
                    data={
                        "regime": reg_obj.regime.value,
                        "volatility_score": reg_obj.volatility_score,
                        "prudence_status": reg_obj.prudence.value,
                        "prudence_directive": reg_obj.prudence_directive,
                        "prudence_multiplier": reg_obj.prudence_multiplier,
                        "web_sentiment": reg_obj.web_sentiment.sentiment_label if reg_obj.web_sentiment else None,
                        "fear_greed_value": reg_obj.web_sentiment.fear_greed_value if reg_obj.web_sentiment else None,
                    },
                )
            )

            if reg_obj.prudence == MarketPrudence.IMPRUDENT_ADVERSE_VETO:
                ledger.limitations.append(
                    RiskLimitation(
                        factor="MarketPrudenceVeto",
                        severity="CRITICAL_VETO",
                        mitigation="Enforce 100% Cash Liquidity; Veto any new Long exposure",
                        source_warning=reg_obj.prudence_directive,
                    )
                )
        except Exception as e:
            logger.debug("[AGENT_BRAIN] Regime specialist notice for %s: %s", symbol, e)

        # 4. Decompile Continuous Learning Lessons
        try:
            rsi_val = 50.0
            for obs in ledger.observations:
                if obs.facet == "quant_metrics" and "rsi" in obs.data:
                    rsi_val = float(obs.data["rsi"] or 50.0)
            pat_check = self.risk_engine.pattern_memory.query_matching_patterns(
                symbol=symbol,
                side=OrderSide.BUY,
                rsi=rsi_val,
                ema_ratio=1.0,
                atr_pct=2.0,
                market_regime="STABLE",
            )
            if pat_check.warning_message:
                ledger.limitations.append(
                    RiskLimitation(
                        factor="RecurringMistakePattern",
                        severity="HIGH",
                        mitigation=pat_check.learned_lesson,
                        source_warning=pat_check.warning_message,
                    )
                )
        except Exception as e:
            logger.debug("[AGENT_BRAIN] Pattern query notice: %s", e)

        ledger.provenance = {
            "source": "AgentBrain.decompile_state",
            "fingerprint": ledger.compute_fingerprint(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return ledger

    # --------------------------------------------------------------------------
    # PHASE 2: UNDERSTAND & DELIBERATE
    # --------------------------------------------------------------------------

    def deliberate_and_synthesize(
        self,
        ledger: EvidenceLedger,
        candidate_rating: str,
        confidence: float,
        target_price: float | None = None,
        stop_loss: float | None = None,
        rationale: str = "",
    ) -> EvidenceLedger:
        """Formulate probabilistic inferences with explicit invalidation bounds and synthesize decision."""
        ledger.phase = CognitivePhase.UNDERSTAND

        # Retrieve current price and available cash from observations
        curr_price = 0.0
        free_cash = 0.0
        for obs in ledger.observations:
            if obs.facet == "quant_metrics" and "current_price" in obs.data:
                curr_price = float(obs.data["current_price"])
        for der in ledger.derivations:
            if der.metric_name == "unrestricted_free_cash":
                free_cash = der.value

        # Check for Critical Vetoes in limitations
        has_critical_veto = any(lim.severity == "CRITICAL_VETO" for lim in ledger.limitations)
        has_recurring_mistake = any(lim.factor == "RecurringMistakePattern" for lim in ledger.limitations)

        # Formulate Invalidation Level
        inv_price = stop_loss if stop_loss else (curr_price * 0.96 if curr_price > 0 else 0.0)
        invalidation_condition = f"Market price breaks below numerical invalidation level ${inv_price:.2f}"

        thesis_text = (
            f"Thesis based on empirical confluence for {ledger.symbol}: "
            f"Rating '{candidate_rating}' with conviction {confidence*100.0:.0f}%. "
            f"Rationale: {rationale or 'Algorithmic confluence confirmed'}"
        )

        ledger.inferences.append(
            TradingInference(
                thesis=thesis_text,
                invalidation_condition=invalidation_condition,
                invalidation_price=inv_price,
                target_price=target_price,
                confidence=confidence,
                time_horizon_bars=5,
            )
        )

        # Decision Synthesis
        if has_critical_veto:
            decision = TradingDecision(
                rating="Hold",
                action="HOLD",
                conviction=0.0,
                entry_price=curr_price,
                suggested_stop_loss=stop_loss,
                suggested_take_profit=target_price,
                suggested_quantity=0.0,
                rationale="[VETO_PREVENTIVO] Veto institucional de investigación de mercado; capital 100% resguardado.",
            )
        elif candidate_rating.upper() in ("BUY", "OVERWEIGHT") and confidence < 0.65:
            decision = TradingDecision(
                rating="Hold",
                action="HOLD",
                conviction=confidence,
                entry_price=curr_price,
                suggested_stop_loss=stop_loss,
                suggested_take_profit=target_price,
                suggested_quantity=0.0,
                rationale=f"Convicción {confidence*100.0:.0f}% insuficiente para justificar riesgo de capital.",
            )
        else:
            action = "BUY" if candidate_rating.upper() in ("BUY", "OVERWEIGHT") else ("SELL" if candidate_rating.upper() in ("SELL", "UNDERWEIGHT") else "HOLD")
            decision = TradingDecision(
                rating=candidate_rating,
                action=action,
                conviction=confidence,
                entry_price=curr_price,
                suggested_stop_loss=stop_loss,
                suggested_take_profit=target_price,
                suggested_quantity=0.0,  # Will be adjusted by RiskEngine in Phase 3
                rationale=rationale or thesis_text,
            )

        ledger.decision = decision
        ledger.provenance["deliberated_at"] = datetime.now(timezone.utc).isoformat()
        ledger.provenance["fingerprint"] = ledger.compute_fingerprint()
        return ledger

    def deliberate_with_cognitive_loop(
        self,
        symbol: str,
        trade_date: str | None = None,
        screen_res: QuantScreenResult | None = None,
        ledger: EvidenceLedger | None = None,
    ) -> EvidenceLedger:
        """Execute the cognitive deliberation loop reverse-engineered from Claude Code & Aider.
        
        Orchestrates the 5 cognitive scratchpad phases:
        1. Telemetry Verification: Read unadulterated facts from EvidenceLedger.
        2. Invalidation Boundary: Formulate the mathematical failure level.
        3. Expectancy Simulation: Backtrader slippage and VectorBT mathematical edge.
        4. Institutional Pre-Flight: Hummingbot drift/spread and Freqtrade cooldown.
        5. Synthesis: Authoritative TradingDecision with cryptographic ledger signature.
        """
        if ledger is None:
            ledger = self.decompile_state(symbol, trade_date)

        curr_price = 0.0
        for obs in ledger.observations:
            if obs.facet == "quant_metrics" and "current_price" in obs.data:
                curr_price = float(obs.data["current_price"])

        if curr_price <= 0 and screen_res and screen_res.current_price > 0:
            curr_price = screen_res.current_price

        # Check for Critical Vetoes in decompiled limitations
        has_critical_veto = any(lim.severity == "CRITICAL_VETO" for lim in ledger.limitations)
        if has_critical_veto:
            logger.info("[COGNITIVE_LOOP] 🛡️ Critical veto active in decompiled state for %s; enforcing HOLD.", symbol)
            return self.deliberate_and_synthesize(
                ledger=ledger,
                candidate_rating="Hold",
                confidence=0.0,
                rationale="Institutional research or prudence veto active; 100% capital preservation enforced.",
            )

        # Baseline Candidate Proposal from Quantitative Screener
        if screen_res and screen_res.is_actionable and screen_res.bias in ("BUY", "Buy"):
            candidate_rating = "Buy"
            confidence = round(screen_res.score / 100.0, 2)
            sl = screen_res.suggested_stop_loss
            tp = screen_res.suggested_take_profit
            base_rationale = (
                f"Quantitative setup '{screen_res.setup_name}' confirmed with Score {screen_res.score:.0f}. "
                f"R:R: {screen_res.risk_reward_ratio:.2f}:1. {screen_res.rationale}"
            )
        else:
            candidate_rating = "Hold"
            confidence = 0.50
            sl = curr_price * 0.96 if curr_price > 0 else None
            tp = curr_price * 1.08 if curr_price > 0 else None
            base_rationale = "No actionable quantitative setup identified; maintaining neutral stance."

        # Step 3: Run Execution Simulator (Backtrader & VectorBT Expectancy)
        if candidate_rating == "Buy" and sl and tp and curr_price > 0:
            sim = self.execution_simulator.simulate_trade_execution(
                nominal_price=curr_price,
                suggested_stop_loss=sl,
                suggested_take_profit=tp,
                quantity=1.0,
                side=OrderSide.BUY,
                asset_class=self.identify_asset_class(symbol),
                historical_win_rate=max(0.50, confidence),
            )
            ledger.derivations.append(
                MathematicalDerivation(
                    metric_name="statistical_expectancy",
                    formula="(WinRate * NetReward) - ((1 - WinRate) * NetRisk)",
                    inputs={"win_rate": max(0.50, confidence), "net_rr": sim.net_risk_reward_ratio},
                    value=sim.statistical_expectancy,
                    unit="USD/unit",
                )
            )
            ledger.derivations.append(
                MathematicalDerivation(
                    metric_name="net_risk_reward_ratio",
                    formula="NetRewardAfterFriction / NetRiskAfterFriction",
                    inputs={"net_rr": sim.net_risk_reward_ratio},
                    value=sim.net_risk_reward_ratio,
                    unit="ratio",
                )
            )

            if not sim.is_positive_expectancy or not sim.is_convex_asymmetric:
                logger.info("[COGNITIVE_LOOP] ⚠️ Expectancy simulation rejected %s: %s", symbol, sim.rejection_reason)
                ledger.limitations.append(
                    RiskLimitation(
                        factor="SimulationExpectancyVeto",
                        severity="HIGH",
                        mitigation="Downgraded to HOLD due to negative edge after slippage and fees",
                        source_warning=sim.rejection_reason,
                    )
                )
                candidate_rating = "Hold"
                confidence = 0.40
                base_rationale = f"Simulation edge check rejected: {sim.rejection_reason}"

        # Step 4: Run Institutional Preflight Guard (Hummingbot & Freqtrade)
        if candidate_rating == "Buy":
            preflight = self.institutional_guard.evaluate_preflight(
                symbol=symbol,
                side=OrderSide.BUY,
                current_price=curr_price,
                screen_price=screen_res.current_price if screen_res else curr_price,
            )
            if not preflight.is_approved:
                err_msg = "; ".join(preflight.veto_reasons)
                logger.info("[COGNITIVE_LOOP] 🛑 Institutional Guard rejected %s: %s", symbol, err_msg)
                ledger.limitations.append(
                    RiskLimitation(
                        factor="InstitutionalGuardVeto",
                        severity="CRITICAL_VETO",
                        mitigation="Execution vetoed by institutional risk guard",
                        source_warning=err_msg,
                    )
                )
                candidate_rating = "Hold"
                confidence = 0.0
                base_rationale = f"Institutional guard veto: {err_msg}"

        return self.deliberate_and_synthesize(
            ledger=ledger,
            candidate_rating=candidate_rating,
            confidence=confidence,
            target_price=tp,
            stop_loss=sl,
            rationale=base_rationale,
        )

    # --------------------------------------------------------------------------
    # PHASE 3: RECREATE & CONTROLLED EXECUTION
    # --------------------------------------------------------------------------

    def recreate_and_execute(self, ledger: EvidenceLedger) -> tuple[Order | None, EvidenceLedger]:
        """Validate against RiskEngine, Institutional Guards, and Simulator; dispatch order to broker."""
        ledger.phase = CognitivePhase.RECREATE
        symbol = ledger.symbol
        asset_class = self.identify_asset_class(symbol)

        if not ledger.decision or ledger.decision.action == "HOLD":
            logger.info("[AGENT_BRAIN] 🛡️ Decisión HOLD para %s; capital resguardado.", symbol)
            self._record_audit_in_db(ledger, order=None)
            return None, ledger

        # 1. Institutional Pre-Flight Guard Check (Hummingbot & Freqtrade)
        action = OrderSide.BUY if ledger.decision.action == "BUY" else OrderSide.SELL
        entry_price = ledger.decision.entry_price
        sl_price = ledger.decision.suggested_stop_loss
        tp_price = ledger.decision.suggested_take_profit

        preflight = self.institutional_guard.evaluate_preflight(
            symbol=symbol,
            side=action,
            current_price=entry_price,
            screen_price=entry_price,
        )
        if not preflight.is_approved:
            err_msg = "; ".join(preflight.veto_reasons)
            logger.info("[AGENT_BRAIN] 🛑 Orden vetada por InstitutionalExecutionGuard para %s: %s", symbol, err_msg)
            ledger.limitations.append(
                RiskLimitation(
                    factor="InstitutionalGuardVeto",
                    severity="CRITICAL_VETO",
                    mitigation="Order execution blocked by institutional guard",
                    source_warning=err_msg,
                )
            )
            self._record_audit_in_db(ledger, order=None)
            return None, ledger

        # 2. Backtrader & VectorBT Expectancy Check
        if action == OrderSide.BUY and sl_price and tp_price and entry_price > 0:
            sim = self.execution_simulator.simulate_trade_execution(
                nominal_price=entry_price,
                suggested_stop_loss=sl_price,
                suggested_take_profit=tp_price,
                quantity=1.0,
                side=action,
                asset_class=asset_class,
                historical_win_rate=ledger.decision.conviction,
            )
            if not sim.is_positive_expectancy or not sim.is_convex_asymmetric:
                rejection = sim.rejection_reason or "Negative mathematical expectancy or non-convex R:R."
                logger.info("[AGENT_BRAIN] 🛑 Orden vetada por ExecutionSimulator para %s: %s", symbol, rejection)
                ledger.limitations.append(
                    RiskLimitation(
                        factor="SimulationExpectancyVeto",
                        severity="HIGH",
                        mitigation="Order execution blocked due to insufficient statistical edge",
                        source_warning=rejection,
                    )
                )
                self._record_audit_in_db(ledger, order=None)
                return None, ledger

        # 3. Build TradeSignal for RiskEngine
        signal = TradeSignal(
            symbol=symbol,
            asset_class=asset_class,
            trade_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            rating=ledger.decision.rating,
            action=action,
            confidence=ledger.decision.conviction,
            current_price=ledger.decision.entry_price,
            target_price=ledger.decision.suggested_take_profit,
            stop_loss=ledger.decision.suggested_stop_loss,
            reasoning=ledger.decision.rationale,
        )

        balance = self.broker.get_account_balance()
        positions = self.broker.get_positions()
        risk_result: RiskValidationResult = self.risk_engine.evaluate(signal, balance, positions)

        if not risk_result.is_approved or risk_result.adjusted_quantity <= 0:
            reasons = ", ".join(risk_result.rejection_reasons) if risk_result.rejection_reasons else "Risk veto"
            logger.info("[AGENT_BRAIN] 🛡️ Orden vetada por RiskEngine para %s: %s", symbol, reasons)
            ledger.limitations.append(
                RiskLimitation(
                    factor="RiskEngineVeto",
                    severity="HIGH",
                    mitigation="Order execution cancelled",
                    source_warning=reasons,
                )
            )
            self._record_audit_in_db(ledger, order=None)
            return None, ledger

        # 4. Construct and Dispatch Order
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            asset_class=asset_class,
            side=action,
            order_type=OrderType.MARKET,
            quantity=risk_result.adjusted_quantity,
            stop_loss_price=risk_result.stop_loss_price,
            take_profit_price=risk_result.take_profit_price,
            status=OrderStatus.PENDING,
        )

        self.institutional_guard.transition_order_state(order.order_id, OrderState.PENDING_CREATE)
        executed_order = self.broker.submit_order(order)
        self.institutional_guard.transition_order_state(order.order_id, OrderState.SUBMITTED)

        if action == OrderSide.BUY:
            if hasattr(self.risk_engine, "register_entry_context"):
                self.risk_engine.register_entry_context(symbol, {
                    "confidence": signal.confidence,
                    "evidence_id": ledger.ledger_id,
                })
            if hasattr(self.risk_engine, "register_order_in_flight"):
                self.risk_engine.register_order_in_flight(symbol, risk_result.adjusted_quantity, ledger.decision.entry_price, asset_class)

        logger.info(
            "[AGENT_BRAIN] 🚀 ¡Orden ejecutada en broker para %s! %s %.4f u @ $%.2f (Ledger: %s)",
            symbol, action.value, risk_result.adjusted_quantity, ledger.decision.entry_price, ledger.ledger_id
        )

        self._record_audit_in_db(ledger, order=executed_order)
        return executed_order, ledger

    def _record_audit_in_db(self, ledger: EvidenceLedger, order: Order | None = None) -> None:
        """Persist the full Evidence Ledger and order into Neon PostgreSQL."""
        try:
            if order:
                self.db.save_order(order)
            
            analyst_signals = {
                "evidence_ledger_id": ledger.ledger_id,
                "phase": ledger.phase.value,
                "observations_count": len(ledger.observations),
                "derivations_count": len(ledger.derivations),
                "inferences_count": len(ledger.inferences),
                "limitations_count": len(ledger.limitations),
                "observations": [o.model_dump(mode="json") for o in ledger.observations],
                "derivations": [d.model_dump(mode="json") for d in ledger.derivations],
                "inferences": [i.model_dump(mode="json") for i in ledger.inferences],
                "limitations": [l.model_dump(mode="json") for l in ledger.limitations],
                "fingerprint": ledger.compute_fingerprint(),
            }
            reasoning = ledger.decision.rationale if ledger.decision else "Deliberation completed"
            rating = ledger.decision.rating if ledger.decision else "Hold"
            action = ledger.decision.action if ledger.decision else "HOLD"
            price = ledger.decision.entry_price if ledger.decision else 0.0

            self.db.save_debate(
                symbol=ledger.symbol,
                trade_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                rating=rating,
                action=action,
                current_price=price,
                analyst_signals=analyst_signals,
                reasoning=reasoning,
            )
        except Exception as e:
            logger.debug("[AGENT_BRAIN] Audit trail save notice: %s", e)

    # --------------------------------------------------------------------------
    # PROMPT GENERATION FOR EXTERNAL LLMs (CLAUDE / CURSOR / GEMINI CLI)
    # --------------------------------------------------------------------------

    def generate_cognitive_system_prompt(self) -> str:
        """Generate structured Chain-of-Thought prompt enforcing REA decomposition."""
        return """You are the Lead Reasoning Engine and Chief Investment Officer for an autonomous trading bot.
You operate under the strict REA (Reverse Engineer Anything) Evidence-Based Framework.

Every decision you make must strictly decompose reality into four distinct categories:
1. OBSERVATIONS: Pure factual numbers from live broker and market telemetry (Cash, Equity, Positions, Price, RSI, ATR). Never hallucinate or guess a number.
2. MATHEMATICAL DERIVATIONS: Verifiable deterministic calculations (50% Protected Cash Reserve, Risk/Reward ratio, Kelly sizing).
3. TRADING INFERENCES: Your analytical thesis. You MUST explicitly define:
   - The thesis statement
   - The exact NUMERICAL INVALIDATION LEVEL (Stop-Loss) that proves the thesis wrong
   - Minimum 2.5:1 Risk/Reward ratio
4. RISK LIMITATIONS & UNKNOWNS: Known headwinds, macro volatility regimes, or lessons from past mistakes.

CRITICAL INSTITUTIONAL INVARIANTS:
- If market research status is IMPRUDENT_ADVERSE_VETO, you MUST return HOLD/SELL. Aggressive buys are strictly forbidden.
- The 50% Profit Quarantine Vault is untouchable. Only unreserved free cash can be allocated.
- Never confuse an inference (hypothesis) with an observation (fact).
"""


# Alias for backward and forward compatibility
ReasoningEngine = AgentBrain
