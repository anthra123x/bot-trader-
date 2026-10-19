"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call. Its rating is the run's
``final_rating``, and the decision is rendered to markdown as
``final_trade_decision`` for the memory log, CLI display and saved reports.
When a provider does not expose structured output, the agent falls back to
free-text generation and the rating is read from that text.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.rating import parse_rating
from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager and Chief Investment Officer, synthesize the risk analysts' debate and deliver the authoritative final trading decision.

{instrument_context}

{portfolio_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong institutional conviction; convex risk/reward justified by empirical confluence
- **Overweight**: Constructive view; gradually increase exposure with disciplined risk limits
- **Hold**: Maintain current position, no action needed, or insufficient edge to risk capital. Conflicting arguments alone are not a reason to Hold: commit to the stronger side, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after weighing, or too thin to support a call.
- **Underweight**: Reduce exposure, harvest partial profits, or de-risk ahead of adverse catalysts
- **Sell**: Exit position completely or avoid entry due to structural downside

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

### Executive Institutional Decision Protocol:
1. **Market Prudence & Macro Alignment**: You MUST align with the Real-Time Market Research Specialist directive in the portfolio context. If the market is marked as IMPRUDENT_ADVERSE_VETO, veto Buy/Overweight calls and enforce Hold/Sell to preserve 100% cash liquidity. Under CAUTION status, cap sizing and enforce defensive risk buffers.
2. **Asymmetric Risk/Reward Verification**: Audit the Trader's proposal. Only approve a Buy if the setup demonstrates an asymmetric reward-to-risk ratio (minimum 2:1 R:R) with a verifiable target price and an explicit numerical invalidation level.
3. **Past Mistake & Pattern Immunity**: Cross-examine the proposed trade against the lessons from prior trades ({lessons_line}). Actively reject setups that mimic known failure modes (e.g. overbought bull traps, choppy range breakdowns, catching falling knives).
4. **Capital Preservation Mandate**: When the risk analysts present valid unhedged downside or conflicting macro evidence, prioritize capital safety over speculative FOMO. A decision to Hold with zero capital loss is superior to an impulsive loss.

## Output

Write these sections, in this order, starting with the rating on its own line:

- **Rating**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Executive Summary**: the definitive call, execution parameters, invalidation level, and sizing guidance
- **Investment Thesis**: the core quantitative and risk evidence that decided it, incorporating prior lessons and what conditions would invalidate the thesis

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        # The typed rating is the decision; the rendered text only carries it.
        # Read back from text, a rating the thesis quotes could replace it.
        decision = invoke_structured(structured_llm, prompt, "Portfolio Manager")
        if decision is not None:
            final_trade_decision = render_pm_decision(decision)
            final_rating = decision.rating.value
        else:
            final_trade_decision = llm.invoke(prompt).content
            final_rating = parse_rating(final_trade_decision)

        new_risk_debate_state = {
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "final_rating": final_rating,
        }

    return portfolio_manager_node
