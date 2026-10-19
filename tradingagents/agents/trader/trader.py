"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()
        portfolio_context = get_portfolio_context_from_state(state)

        if market_report:
            grounding = (
                "Apply Reverse-Engineering Analysis (Decompile -> Understand Invalidation -> Reconstruct Proposal): "
                "1. Decompile: Ground concrete price levels in the technical market report's real price structure -- "
                "current price, support/resistance, ATR, and volatility -- rather than ungrounded estimates. "
                "2. Understand Failure (Invalidation Condition): Explicitly define the invalidation condition (stop-loss) "
                "where this trade is proven false before calculating any profit target. "
                "3. Mathematical Asymmetry: Enforce mathematical risk/reward convexity: only propose Buy when expected profit "
                "target offers at least 2.0x the distance to stop-loss ((Target - Entry) / (Entry - StopLoss) >= 2.0). "
                "Place stop-loss beyond normal volatility noise using ATR and local swing support. "
                "If favorable 2:1 asymmetry is absent or risk is unquantifiable, recommend Hold. "
                "4. Market Prudence: Respect the Real-Time Market Research Specialist directive in portfolio context: under CAUTION "
                "status, scale sizing down to 50%; under IMPRUDENT status, recommend Hold. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = (
                "Apply Reverse-Engineering Analysis: Enforce convex risk/reward: only propose Buy when upside potential "
                "is at least twice the downside risk and has an explicit invalidation condition. Otherwise recommend Hold. "
            )
            report_section = ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an elite quantitative trading agent formulating institutional transaction proposals. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    + grounding
                    # Entry/stop are numeric price fields. Asking for concrete
                    # levels invites a percentage ("15%"), which is not a price
                    # and fails the structured parse (#1288).
                    + "State entry price and stop-loss as absolute price levels in the "
                    "instrument's quote currency (for example 189.5), never a percentage "
                    "or a range; convert a percentage distance to the price level it "
                    "implies, or omit the field if you cannot state a number. "
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"{report_section}"
                    f"{portfolio_context}\n\n"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    "Make an informed, strategic trading decision.\n\n"
                    "## Output\n\n"
                    "Write these sections, in this order, starting with the action "
                    "on its own line:\n\n"
                    "- **Action**: exactly one of Buy / Hold / Sell. A research "
                    "recommendation of Overweight is a Buy and Underweight is a Sell, "
                    "sized by how strong the case is; conflict alone is not a Hold.\n"
                    "- **Reasoning**: why, against the plan and the price structure\n"
                    "- **Entry Price**, **Stop Loss**, **Position Sizing**: when you can state them"
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
        }

    return trader_node
