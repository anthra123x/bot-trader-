"""Reasoning and Cognitive Decomposition Module inspired by morluto/rea.

Provides:
- AgentBrain (ReasoningEngine): 3-phase cognitive flow (Decompile -> Understand -> Execute).
- Contracts & Pydantic models: EvidenceLedger, Observation, MathematicalDerivation, TradingInference, RiskLimitation.
- MCPTradingAdapter: Model Context Protocol (MCP) tool server and function-calling schemas.
"""

from bot.reasoning.contracts import (
    CognitivePhase,
    EvidenceKind,
    EvidenceLedger,
    ExecuteOrderArgs,
    ExecuteOrderResult,
    InspectMarketArgs,
    InspectMarketResult,
    InspectPortfolioArgs,
    InspectPortfolioResult,
    MathematicalDerivation,
    Observation,
    RiskLimitation,
    TradingDecision,
    TradingInference,
    ValidateRiskArgs,
    ValidateRiskResult,
)
from bot.reasoning.agent_brain import AgentBrain, ReasoningEngine
from bot.reasoning.mcp_adapter import MCPTradingAdapter

__all__ = [
    "AgentBrain",
    "ReasoningEngine",
    "CognitivePhase",
    "EvidenceKind",
    "EvidenceLedger",
    "Observation",
    "MathematicalDerivation",
    "TradingInference",
    "RiskLimitation",
    "TradingDecision",
    "InspectPortfolioArgs",
    "InspectPortfolioResult",
    "InspectMarketArgs",
    "InspectMarketResult",
    "ValidateRiskArgs",
    "ValidateRiskResult",
    "ExecuteOrderArgs",
    "ExecuteOrderResult",
    "MCPTradingAdapter",
]
