"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
    report_or_absent,
)
from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)
        market_report = report_or_absent(state.get("market_report"), "market")
        sentiment_report = report_or_absent(state.get("sentiment_report"), "sentiment")
        news_report = report_or_absent(state.get("news_report"), "news")
        fundamentals_report = report_or_absent(state.get("fundamentals_report"), "fundamentals")
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and Chief Investment Strategist, your role is to critically synthesize the bull/bear debate and deliver a rigorous, evidence-based investment plan for the trader.

{instrument_context}

{portfolio_context}

### Core Intelligence Reports:
- Technical Market Structure: {market_report}
- Global News & Catalysts: {news_report}
- Social Sentiment Dynamics: {sentiment_report}
- Fundamentals Analysis: {fundamentals_report}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis backed by empirical confluence; recommend taking or growing position
- **Overweight**: Constructive view; recommend gradually increasing exposure with prudent sizing
- **Hold**: Balanced view or lack of directional edge. Conflicting arguments alone are not a reason to Hold: commit to the stronger side, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after weighing, or too thin to support a call.
- **Underweight**: Cautious view, fading momentum, or emerging risks; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

### Strategic Evaluation Protocols (Reverse Engineering Discipline):
1. **Real-Time Market Prudence**: Respect the Real-Time Market Research Specialist directive provided in the portfolio context. If the market is in CAUTION or IMPRUDENT status, do not issue an unhedged Buy; moderate to Hold or Overweight with defensive sizing.
2. **Empirical Edge Over Rhetoric (Decompile vs Narrative)**: The debate contains opposing arguments. Strip speculative rhetoric and weigh evidence based on verifiable support/resistance levels, volume, real liquidity, and macro trend.
3. **Asymmetric Risk/Reward Focus & Invalidation (Reverse Engineer Failure)**: Demand that strategic actions target an asymmetric reward-to-risk profile (minimum 2:1 R:R) and specify clear invalidation conditions where the trade thesis is mathematically disproven.

---

**Debate History:**
{history}

## Output

Write these sections, in this order, starting with the recommendation on its own line:

- **Recommendation**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Rationale**: which arguments and empirical evidence decided it
- **Strategic Actions**: concrete steps for the trader, sized against risk conditions (entry zone, R:R target, invalidation level)

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
