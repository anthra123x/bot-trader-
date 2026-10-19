"""MCP Adapter for Trading Agents: Tool Calling, Function Schema, and stdio MCP Server.

Provides strict JSON Schema tool definitions compatible with:
- Model Context Protocol (MCP) clients (Claude Code, Cursor, Gemini CLI, Windsurf).
- Direct LLM function calling (OpenAI, Anthropic, Gemini, DeepSeek).
- Native in-process integration with AgentBrain and RiskEngine.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable, Dict, List

from bot.config import BOT_CONFIG, BotConfig
from bot.reasoning.agent_brain import AgentBrain
from bot.reasoning.contracts import (
    CognitivePhase,
    EvidenceLedger,
    ExecuteOrderArgs,
    ExecuteOrderResult,
    InspectMarketArgs,
    InspectMarketResult,
    InspectPortfolioArgs,
    InspectPortfolioResult,
    ValidateRiskArgs,
    ValidateRiskResult,
)

logger = logging.getLogger(__name__)


class MCPTradingAdapter:
    """Tool calling router and protocol adapter for quantitative trading agents."""

    def __init__(self, brain: AgentBrain | None = None, config: BotConfig | None = None):
        self.config = config or BOT_CONFIG
        self.brain = brain or AgentBrain(config=self.config)
        self.active_ledgers: dict[str, EvidenceLedger] = {}

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

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name and return canonical result or error diagnostic."""
        try:
            if name == "inspect_portfolio":
                return self._tool_inspect_portfolio(arguments)
            elif name == "inspect_market":
                return self._tool_inspect_market(arguments)
            elif name == "deliberate_evidence_ledger":
                return self._tool_deliberate_evidence(arguments)
            elif name == "validate_risk":
                return self._tool_validate_risk(arguments)
            elif name == "execute_order":
                return self._tool_execute_order(arguments)
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
        protected_floor, _, _ = self.brain.risk_engine.calculate_protected_cash_reserve(bal)
        free_cash = max(0.0, bal.cash - protected_floor)

        result = InspectPortfolioResult(
            cash=bal.cash,
            equity=bal.equity,
            buying_power=bal.buying_power,
            protected_cash_reserve=protected_floor,
            unrestricted_free_cash=free_cash,
            open_positions=[p.model_dump(mode="json") for p in positions] if args.get("include_positions", True) else [],
            drawdown_pct=bal.daily_drawdown_pct,
        )
        return {
            "structuredContent": result.model_dump(mode="json"),
            "content": [{"type": "text", "text": json.dumps(result.model_dump(mode="json"), indent=2)}],
        }

    def _tool_inspect_market(self, args: dict[str, Any]) -> dict[str, Any]:
        symbol = args["symbol"]
        ledger = self.brain.decompile_state(symbol)
        self.active_ledgers[symbol] = ledger

        price = 0.0
        rsi = None
        atr = None
        setup = None
        score = None
        regime = "NORMAL"
        prudence = "PRUDENT_FAVORABLE"
        directive = "Normal execution"
        vol_score = 50.0
        fgi = 50
        sentiment = "Neutral"

        for obs in ledger.observations:
            if obs.facet == "quant_metrics":
                price = float(obs.data.get("current_price", 0.0))
                rsi = obs.data.get("rsi")
                atr = obs.data.get("atr")
                setup = obs.data.get("setup_name")
                score = obs.data.get("score")
            elif obs.facet == "macro_regime":
                regime = obs.data.get("regime", "NORMAL")
                prudence = obs.data.get("prudence_status", "PRUDENT_FAVORABLE")
                directive = obs.data.get("prudence_directive", "")
                vol_score = float(obs.data.get("volatility_score", 50.0))
                fgi = obs.data.get("fear_greed_value")
                sentiment = obs.data.get("web_sentiment")

        lessons = [lim.mitigation for lim in ledger.limitations if lim.factor == "RecurringMistakePattern"]

        result = InspectMarketResult(
            symbol=symbol,
            current_price=price,
            regime=regime,
            prudence_status=prudence,
            prudence_directive=directive,
            volatility_score=vol_score,
            web_sentiment_score=float(fgi) if fgi is not None else None,
            fear_greed_index=int(fgi) if fgi is not None else None,
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

    def _tool_deliberate_evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        symbol = args["symbol"]
        rating = args["rating"]
        conviction = float(args["conviction"])
        inv_price = args.get("invalidation_price")
        target_price = args.get("target_price")
        rationale = args["thesis_rationale"]

        # Retrieve existing decompiled ledger or create one
        ledger = self.active_ledgers.get(symbol) or self.brain.decompile_state(symbol)
        deliberated = self.brain.deliberate_and_synthesize(
            ledger=ledger,
            candidate_rating=rating,
            confidence=conviction,
            target_price=target_price,
            stop_loss=inv_price,
            rationale=rationale,
        )
        self.active_ledgers[symbol] = deliberated
        self.active_ledgers[deliberated.ledger_id] = deliberated

        return {
            "structuredContent": deliberated.model_dump(mode="json"),
            "content": [{"type": "text", "text": json.dumps(deliberated.model_dump(mode="json"), indent=2)}],
        }

    def _tool_validate_risk(self, args: dict[str, Any]) -> dict[str, Any]:
        from bot.models import AssetClass, OrderSide, TradeSignal
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
            # Fallback by symbol
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
                filled_price=order.filled_price or ledger.decision.entry_price if ledger.decision else None,
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
