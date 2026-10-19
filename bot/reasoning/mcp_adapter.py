"""MCP Adapter for Trading Agents: Tool Calling, Function Schema, and stdio MCP Server.

Provides strict JSON Schema tool definitions compatible with:
- Model Context Protocol (MCP) clients (Claude Code, Cursor, Gemini CLI, Windsurf).
- Direct LLM function calling (OpenAI, Anthropic, Gemini, DeepSeek).
- Native in-process integration with AgentBrain, RiskEngine, InstitutionalExecutionGuard, and ExecutionSimulator.

Implements REA permission isolation:
- Observational tools: Read-only live telemetry (safe, zero mutation).
- Deliberative tools: Mutates session ledger with SHA-256 fingerprint audit.
- Mutative tools: State mutation gated by ExternalMutationAdmission (strict fail-closed checks).
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional

from bot.config import BOT_CONFIG, BotConfig
from bot.models import OrderSide, TradeSignal
from bot.reasoning.agent_brain import AgentBrain
from bot.reasoning.contracts import (
    AdmissionTicket,
    CognitivePhase,
    EvidenceLedger,
    ExecuteOrderArgs,
    ExecuteOrderResult,
    InspectMarketArgs,
    InspectMarketResult,
    InspectPortfolioArgs,
    InspectPortfolioResult,
    PermissionTier,
    SimulateOrderArgs,
    SimulateOrderResult,
    TOOL_EFFECTS_REGISTRY,
    ValidateRiskArgs,
    ValidateRiskResult,
)
from bot.risk.institutional_guard import InstitutionalExecutionGuard
from bot.risk.order_simulator import ExecutionSimulator

logger = logging.getLogger(__name__)


class MCPTradingAdapter:
    """Tool calling router and protocol adapter with REA permission isolation."""

    def __init__(
        self,
        brain: AgentBrain | None = None,
        config: BotConfig | None = None,
        institutional_guard: InstitutionalExecutionGuard | None = None,
        execution_simulator: ExecutionSimulator | None = None,
    ):
        self.config = config or BOT_CONFIG
        self.brain = brain or AgentBrain(config=self.config)
        self.institutional_guard = institutional_guard or InstitutionalExecutionGuard(
            cooldown_seconds=self.config.risk.cooldown_period_seconds,
        )
        self.execution_simulator = execution_simulator or ExecutionSimulator(
            min_risk_reward_ratio=2.0,
        )
        self.active_ledgers: dict[str, EvidenceLedger] = {}
        self.admission_tickets: dict[str, AdmissionTicket] = {}

    # --------------------------------------------------------------------------
    # REA PERMISSION ISOLATION & ADMISSION CONTROLLER
    # --------------------------------------------------------------------------

    def with_admitted_external_mutation(
        self,
        tool_name: str,
        ledger: EvidenceLedger | None,
        operation: Callable[[], dict[str, Any]],
        symbol: str,
        current_price: float,
        suggested_stop_loss: float | None = None,
        suggested_take_profit: float | None = None,
        quantity: float = 1.0,
        side: OrderSide = OrderSide.BUY,
    ) -> dict[str, Any]:
        """Admit a mutative action only if cryptographic and institutional checks pass."""
        tool_effects = TOOL_EFFECTS_REGISTRY.get(tool_name)
        tier = tool_effects.permission_tier if tool_effects else PermissionTier.OBSERVATIONAL

        # Observational and deliberative tiers don't require broker mutation admission
        if tier != PermissionTier.MUTATIVE:
            return operation()

        # Phase validation: must have deliberated ledger
        if not ledger:
            logger.warning("[MCP_ADMISSION] 🛑 Rejected %s: No active EvidenceLedger bound to session.", tool_name)
            ticket = AdmissionTicket(
                ledger_id="none",
                tool_name=tool_name,
                is_admitted=False,
                rejection_reason="No active EvidenceLedger bound to session. Decompile and deliberate first.",
                fingerprint="",
            )
            return {
                "isError": True,
                "admission": ticket.model_dump(mode="json"),
                "content": [{"type": "text", "text": ticket.rejection_reason}],
            }

        if ledger.phase not in (CognitivePhase.UNDERSTAND, CognitivePhase.RECREATE):
            reason = f"Invalid ledger phase '{ledger.phase}'. Must be in UNDERSTAND phase with synthesized thesis."
            logger.warning("[MCP_ADMISSION] 🛑 Rejected %s: %s", tool_name, reason)
            ticket = AdmissionTicket(
                ledger_id=ledger.ledger_id,
                tool_name=tool_name,
                is_admitted=False,
                rejection_reason=reason,
                fingerprint=ledger.compute_fingerprint(),
            )
            return {
                "isError": True,
                "admission": ticket.model_dump(mode="json"),
                "content": [{"type": "text", "text": reason}],
            }

        # 1. Institutional Pre-Flight Guard (Hummingbot & Freqtrade)
        preflight = self.institutional_guard.evaluate_preflight(
            symbol=symbol,
            side=side,
            current_price=current_price,
            screen_price=current_price,
        )
        if not preflight.is_approved:
            reason = "; ".join(preflight.veto_reasons)
            logger.warning("[MCP_ADMISSION] 🛑 Institutional Guard VETO for %s: %s", symbol, reason)
            ticket = AdmissionTicket(
                ledger_id=ledger.ledger_id,
                tool_name=tool_name,
                is_admitted=False,
                rejection_reason=reason,
                fingerprint=ledger.compute_fingerprint(),
            )
            return {
                "isError": True,
                "admission": ticket.model_dump(mode="json"),
                "content": [{"type": "text", "text": reason}],
            }

        # 2. Backtrader / VectorBT Expectancy & Convexity Check
        if suggested_stop_loss and suggested_take_profit and suggested_stop_loss > 0 and suggested_take_profit > 0:
            sim = self.execution_simulator.simulate_trade_execution(
                nominal_price=current_price,
                suggested_stop_loss=suggested_stop_loss,
                suggested_take_profit=suggested_take_profit,
                quantity=quantity,
                side=side,
                asset_class=self.brain.identify_asset_class(symbol),
            )
            if not sim.is_positive_expectancy or not sim.is_convex_asymmetric:
                reason = sim.rejection_reason or "Negative mathematical expectancy or non-convex R:R."
                logger.warning("[MCP_ADMISSION] 🛑 Simulation Edge VETO for %s: %s", symbol, reason)
                ticket = AdmissionTicket(
                    ledger_id=ledger.ledger_id,
                    tool_name=tool_name,
                    is_admitted=False,
                    rejection_reason=reason,
                    fingerprint=ledger.compute_fingerprint(),
                )
                return {
                    "isError": True,
                    "admission": ticket.model_dump(mode="json"),
                    "content": [{"type": "text", "text": reason}],
                }

        # Admission Granted
        ticket = AdmissionTicket(
            ledger_id=ledger.ledger_id,
            tool_name=tool_name,
            is_admitted=True,
            fingerprint=ledger.compute_fingerprint(),
        )
        self.admission_tickets[ticket.ticket_id] = ticket
        logger.info("[MCP_ADMISSION] ✅ Mutation admitted for %s (Ticket: %s)", symbol, ticket.ticket_id)
        result = operation()
        if isinstance(result, dict) and "structuredContent" in result:
            result["admissionTicket"] = ticket.model_dump(mode="json")
        return result

    # --------------------------------------------------------------------------
    # TOOL DEFINITIONS (MCP JSON SCHEMA)
    # --------------------------------------------------------------------------

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return canonical tool schemas for MCP and LLM function calling."""
        return [
            {
                "name": "inspect_portfolio",
                "description": (
                    "Reads unadulterated live portfolio telemetry from AlpacaBroker: "
                    "cash, equity, buying power, open positions, and the 50% Profit Quarantine Vault floor."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_positions": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to include detailed open positions list.",
                        }
                    },
                },
            },
            {
                "name": "inspect_market",
                "description": (
                    "Inspects deterministic technical indicators (RSI, ATR, Bollinger), "
                    "macroeconomic market regime, web sentiment, and historical mistake patterns for a symbol."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Ticker symbol (e.g., 'BTC/USD', 'AAPL', 'ETH/USD').",
                        }
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "simulate_order_execution",
                "description": (
                    "Simulates realistic fill prices with Backtrader directional slippage, "
                    "VectorBT commission friction, and verifies mathematical expectancy and convexity."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "action": {"type": "string", "enum": ["BUY", "SELL"], "default": "BUY"},
                        "nominal_price": {"type": "number"},
                        "stop_loss": {"type": "number"},
                        "take_profit": {"type": "number"},
                        "quantity": {"type": "number", "default": 1.0},
                        "historical_win_rate": {"type": "number", "default": 0.55},
                    },
                    "required": ["symbol", "nominal_price", "stop_loss", "take_profit"],
                },
            },
            {
                "name": "deliberate_evidence_ledger",
                "description": (
                    "Submits the agent's chain-of-thought breakdown following the REA architecture: "
                    "records observations, derivations, probabilistic inferences with explicit invalidation levels, "
                    "and limitations before executing a decision."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "rating": {
                            "type": "string",
                            "enum": ["Buy", "Overweight", "Hold", "Underweight", "Sell"],
                            "description": "Authoritative rating decision.",
                        },
                        "conviction": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Conviction score between 0.0 and 1.0.",
                        },
                        "invalidation_price": {
                            "type": "number",
                            "description": "Exact price level where the thesis is proven false.",
                        },
                        "target_price": {
                            "type": "number",
                            "description": "Take-profit price target.",
                        },
                        "thesis_rationale": {
                            "type": "string",
                            "description": "Detailed reasoning referencing observed data and mathematical derivations.",
                        },
                    },
                    "required": ["symbol", "rating", "conviction", "thesis_rationale"],
                },
            },
            {
                "name": "validate_risk",
                "description": (
                    "Runs the proposed trade through the quantitative RiskEngine: "
                    "enforces the 50% Profit Quarantine Vault, Kelly position sizing, "
                    "maximum drawdowns, and pattern memory vetoes."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["BUY", "SELL", "HOLD"],
                        },
                        "current_price": {"type": "number"},
                        "stop_loss": {"type": "number"},
                        "target_price": {"type": "number"},
                        "confidence": {"type": "number", "default": 0.5},
                    },
                    "required": ["symbol", "action", "current_price"],
                },
            },
            {
                "name": "execute_order",
                "description": (
                    "Executes an approved order on AlpacaBroker with attached Evidence Ledger ID "
                    "and persists full audit telemetry to Neon PostgreSQL."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "action": {"type": "string", "enum": ["BUY", "SELL"]},
                        "quantity": {"type": "number"},
                        "stop_loss": {"type": "number"},
                        "take_profit": {"type": "number"},
                        "ledger_id": {
                            "type": "string",
                            "description": "ID of the deliberated Evidence Ledger.",
                        },
                    },
                    "required": ["symbol", "action", "quantity", "ledger_id"],
                },
            },
        ]

    # --------------------------------------------------------------------------
    # EXECUTE TOOL ROUTER
    # --------------------------------------------------------------------------

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name with REA permission checks and return canonical JSON result."""
        try:
            if name == "inspect_portfolio":
                return self._tool_inspect_portfolio(arguments)
            elif name == "inspect_market":
                return self._tool_inspect_market(arguments)
            elif name == "simulate_order_execution":
                return self._tool_simulate_order_execution(arguments)
            elif name == "deliberate_evidence_ledger":
                return self._tool_deliberate_evidence(arguments)
            elif name == "validate_risk":
                symbol = arguments.get("symbol", "")
                price = float(arguments.get("current_price", 0.0))
                sl = float(arguments["stop_loss"]) if arguments.get("stop_loss") else None
                tp = float(arguments["target_profit"]) if arguments.get("target_profit") else (float(arguments["target_price"]) if arguments.get("target_price") else None)
                side = OrderSide.BUY if arguments.get("action", "").upper() == "BUY" else OrderSide.SELL
                ledger = self.active_ledgers.get(symbol)
                return self.with_admitted_external_mutation(
                    tool_name="validate_risk",
                    ledger=ledger,
                    operation=lambda: self._tool_validate_risk(arguments),
                    symbol=symbol,
                    current_price=price,
                    suggested_stop_loss=sl,
                    suggested_take_profit=tp,
                    side=side,
                )
            elif name == "execute_order":
                symbol = arguments.get("symbol", "")
                ledger_id = arguments.get("ledger_id", "")
                ledger = self.active_ledgers.get(ledger_id) or self.active_ledgers.get(symbol)
                qty = float(arguments.get("quantity", 0.0))
                sl = float(arguments["stop_loss"]) if arguments.get("stop_loss") else None
                tp = float(arguments["take_profit"]) if arguments.get("take_profit") else None
                side = OrderSide.BUY if arguments.get("action", "").upper() == "BUY" else OrderSide.SELL
                price = ledger.decision.entry_price if ledger and ledger.decision else 0.0
                return self.with_admitted_external_mutation(
                    tool_name="execute_order",
                    ledger=ledger,
                    operation=lambda: self._tool_execute_order(arguments),
                    symbol=symbol,
                    current_price=price,
                    suggested_stop_loss=sl,
                    suggested_take_profit=tp,
                    quantity=qty,
                    side=side,
                )
            else:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Unknown tool: '{name}'"}],
                }
        except Exception as e:
            logger.error("[MCP_ADAPTER] Tool execution error for %s: %s", name, e, exc_info=True)
            return {
                "isError": True,
                "content": [{"type": "text", "text": json.dumps({"error": str(e), "tool": name})}],
            }

    # --------------------------------------------------------------------------
    # TOOL HANDLERS
    # --------------------------------------------------------------------------

    def _tool_inspect_portfolio(self, args: dict[str, Any]) -> dict[str, Any]:
        bal = self.brain.broker.get_account_balance()
        positions = self.brain.broker.get_positions()
        protected_floor, _, quarantined_profit = self.brain.risk_engine.calculate_protected_cash_reserve(bal)
        free_cash = max(0.0, bal.cash - protected_floor)

        result = InspectPortfolioResult(
            cash=bal.cash,
            equity=bal.equity,
            buying_power=bal.buying_power,
            protected_cash_reserve=protected_floor,
            unrestricted_free_cash=free_cash,
            drawdown_pct=bal.daily_drawdown_pct,
            open_positions=[
                {
                    "symbol": p.symbol,
                    "qty": p.quantity,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "pnl": p.unrealized_pnl,
                    "pnl_pct": p.unrealized_pnl_pct,
                }
                for p in positions
            ] if args.get("include_positions", True) else [],
        )
        return {
            "structuredContent": result.model_dump(mode="json"),
            "content": [{"type": "text", "text": json.dumps(result.model_dump(mode="json"), indent=2)}],
        }

    def _tool_inspect_market(self, args: dict[str, Any]) -> dict[str, Any]:
        symbol = args["symbol"]
        norm_sym = symbol.replace("/", "-")

        # Decompile into internal state ledger
        ledger = self.brain.decompile_state(symbol)
        self.active_ledgers[symbol] = ledger
        self.active_ledgers[ledger.ledger_id] = ledger

        # Extract telemetry
        curr_price = 0.0
        rsi = None
        atr = None
        setup = None
        score = None
        regime = "STABLE"
        prudence = "PRUDENT"
        directive = "Normal operation"
        volatility = 1.0
        web_sent = None
        fng = None
        lessons: list[str] = []

        for obs in ledger.observations:
            if obs.facet == "quant_metrics":
                curr_price = float(obs.data.get("current_price", 0.0))
                rsi = obs.data.get("rsi")
                atr = obs.data.get("atr")
                setup = obs.data.get("setup_name")
                score = obs.data.get("score")
            elif obs.facet == "macro_regime":
                regime = obs.data.get("regime", "STABLE")
                prudence = obs.data.get("prudence_status", "PRUDENT")
                directive = obs.data.get("prudence_directive", "")
                volatility = obs.data.get("volatility_score", 1.0)
                web_sent = obs.data.get("web_sentiment")
                fng = obs.data.get("fear_greed_value")

        for lim in ledger.limitations:
            if lim.factor == "RecurringMistakePattern":
                lessons.append(f"Caution: {lim.source_warning}. Lesson: {lim.mitigation}")

        result = InspectMarketResult(
            symbol=symbol,
            current_price=curr_price,
            regime=regime,
            prudence_status=prudence,
            prudence_directive=directive,
            volatility_score=volatility,
            web_sentiment_score=None,
            fear_greed_index=fng,
            quant_setup=setup,
            quant_score=score,
            rsi=rsi,
            atr=atr,
            historical_lessons=lessons,
        )
        return {
            "structuredContent": result.model_dump(mode="json"),
            "content": [{"type": "text", "text": json.dumps(result.model_dump(mode="json"), indent=2)}],
        }

    def _tool_simulate_order_execution(self, args: dict[str, Any]) -> dict[str, Any]:
        symbol = args["symbol"]
        action = OrderSide.BUY if args.get("action", "BUY").upper() == "BUY" else OrderSide.SELL
        nominal = float(args["nominal_price"])
        sl = float(args["stop_loss"])
        tp = float(args["take_profit"])
        qty = float(args.get("quantity", 1.0))
        win_rate = float(args.get("historical_win_rate", 0.55))

        sim = self.execution_simulator.simulate_trade_execution(
            nominal_price=nominal,
            suggested_stop_loss=sl,
            suggested_take_profit=tp,
            quantity=qty,
            side=action,
            asset_class=self.brain.identify_asset_class(symbol),
            historical_win_rate=win_rate,
        )

        res = SimulateOrderResult(
            simulated_fill_price=sim.simulated_fill_price,
            effective_stop_loss=sim.effective_stop_loss,
            effective_take_profit=sim.effective_take_profit,
            estimated_slippage_dollars=sim.estimated_slippage_dollars,
            estimated_commission_dollars=sim.estimated_commission_dollars,
            net_reward_dollars=sim.net_reward_dollars,
            net_risk_dollars=sim.net_risk_dollars,
            net_risk_reward_ratio=sim.net_risk_reward_ratio,
            statistical_expectancy=sim.statistical_expectancy,
            is_positive_expectancy=sim.is_positive_expectancy,
            is_convex_asymmetric=sim.is_convex_asymmetric,
            rejection_reason=sim.rejection_reason,
        )
        return {
            "structuredContent": res.model_dump(mode="json"),
            "content": [{"type": "text", "text": json.dumps(res.model_dump(mode="json"), indent=2)}],
        }

    def _tool_deliberate_evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        symbol = args["symbol"]
        ledger = self.active_ledgers.get(symbol)
        if not ledger:
            ledger = self.brain.decompile_state(symbol)
            self.active_ledgers[symbol] = ledger

        rating = args["rating"]
        conviction = float(args["conviction"])
        invalidation = args.get("invalidation_price")
        target = args.get("target_price")
        rationale = args.get("thesis_rationale", "")

        deliberated = self.brain.deliberate_and_synthesize(
            ledger=ledger,
            candidate_rating=rating,
            confidence=conviction,
            target_price=target,
            stop_loss=invalidation,
            rationale=rationale,
        )

        self.active_ledgers[symbol] = deliberated
        self.active_ledgers[deliberated.ledger_id] = deliberated

        return {
            "structuredContent": {
                "ledger_id": deliberated.ledger_id,
                "fingerprint": deliberated.compute_fingerprint(),
                "action": deliberated.decision.action if deliberated.decision else "HOLD",
                "conviction": deliberated.decision.conviction if deliberated.decision else 0.0,
                "invalidation_price": invalidation,
            },
            "content": [{
                "type": "text",
                "text": f"Evidence Ledger {deliberated.ledger_id} deliberated successfully (Action: {deliberated.decision.action if deliberated.decision else 'HOLD'}, Fingerprint: {deliberated.compute_fingerprint()[:16]}...)",
            }],
        }

    def _tool_validate_risk(self, args: dict[str, Any]) -> dict[str, Any]:
        symbol = args["symbol"]
        action_str = args["action"].upper()
        action = OrderSide.BUY if action_str == "BUY" else (OrderSide.SELL if action_str == "SELL" else OrderSide.HOLD)
        price = float(args["current_price"])
        sl = args.get("stop_loss")
        tp = args.get("target_price")
        conf = float(args.get("confidence", 0.5))

        signal = TradeSignal(
            symbol=symbol,
            asset_class=self.brain.identify_asset_class(symbol),
            trade_date="today",
            rating=action_str.capitalize(),
            action=action,
            confidence=conf,
            current_price=price,
            stop_loss=sl,
            target_price=tp,
        )

        bal = self.brain.broker.get_account_balance()
        pos = self.brain.broker.get_positions()
        eval_res = self.brain.risk_engine.evaluate(signal, bal, pos)

        res = ValidateRiskResult(
            is_approved=eval_res.is_approved,
            adjusted_quantity=eval_res.adjusted_quantity,
            risk_multiplier=eval_res.risk_multiplier,
            stop_loss_price=eval_res.stop_loss_price,
            take_profit_price=eval_res.take_profit_price,
            rejection_reasons=eval_res.rejection_reasons,
            warning_notes=eval_res.warning_notes,
        )
        return {
            "structuredContent": res.model_dump(mode="json"),
            "content": [{"type": "text", "text": json.dumps(res.model_dump(mode="json"), indent=2)}],
        }

    def _tool_execute_order(self, args: dict[str, Any]) -> dict[str, Any]:
        ledger_id = args["ledger_id"]
        ledger = self.active_ledgers.get(ledger_id)

        if not ledger:
            symbol = args["symbol"]
            ledger = self.active_ledgers.get(symbol)

        if not ledger:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"No active Evidence Ledger found with ID {ledger_id}"}],
            }

        order, executed_ledger = self.brain.recreate_and_execute(ledger)
        if order:
            result = ExecuteOrderResult(
                success=True,
                order_id=order.order_id,
                symbol=order.symbol,
                action=order.side.value,
                executed_quantity=order.quantity,
                filled_price=order.filled_price or (ledger.decision.entry_price if ledger.decision else None),
                status=order.status.value,
            )
        else:
            rejection = "Order not filled or vetoed by RiskEngine"
            for lim in executed_ledger.limitations:
                if lim.factor == "RiskEngineVeto":
                    rejection = lim.source_warning or rejection
            result = ExecuteOrderResult(
                success=False,
                order_id=None,
                symbol=args["symbol"],
                action=args["action"],
                executed_quantity=0.0,
                status="VETOED",
                error_message=rejection,
            )

        return {
            "structuredContent": result.model_dump(mode="json"),
            "content": [{"type": "text", "text": json.dumps(result.model_dump(mode="json"), indent=2)}],
        }

    # --------------------------------------------------------------------------
    # STDIO MCP PROTOCOL LOOP
    # --------------------------------------------------------------------------

    def run_stdio_server(self) -> None:
        """Run an interactive stdio JSON-RPC server adhering to the Model Context Protocol."""
        sys.stderr.write("[MCP_SERVER] Trading Agents MCP Server initialized on stdio.\n")
        sys.stderr.flush()

        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                req_id = request.get("id")
                method = request.get("method")
                params = request.get("params", {})

                if method == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"tools": self.get_tool_definitions()},
                    }
                elif method == "tools/call":
                    tool_name = params.get("name")
                    tool_args = params.get("arguments", {})
                    call_res = self.execute_tool(tool_name, tool_args)
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": call_res,
                    }
                elif method == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": "tradingagents-mcp", "version": "1.0.0"},
                            "capabilities": {"tools": {}},
                        },
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Method not found: {method}"},
                    }

                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": f"Internal RPC error: {str(e)}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def main():
    """CLI entrypoint for running the MCP server."""
    adapter = MCPTradingAdapter()
    adapter.run_stdio_server()


if __name__ == "__main__":
    main()
