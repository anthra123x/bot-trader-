"""Contracts, Pydantic Schemas, and Evidence Data Models for REA-inspired Cognitive Trading.

Adapts the reverse-engineering evidence ledger architecture from `morluto/rea`
to quantitative autonomous trading with strict distinction between:
- Observations: Directly inspected, immutable live market and portfolio facts.
- Mathematical Derivations: Deterministic calculations (Kelly sizing, 50% quarantine reserve, R:R).
- Trading Inferences: Probabilistic hypotheses with explicit invalidation levels.
- Risk Limitations & Unknowns: Operational caveats, liquidity bounds, and adverse catalysts.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class EvidenceKind(StrEnum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class PermissionTier(StrEnum):
    """REA-inspired permission tiers for tool isolation."""
    OBSERVATIONAL = "OBSERVATIONAL"  # Read-only telemetry (safe, zero mutation)
    DELIBERATIVE = "DELIBERATIVE"    # Session/Ledger mutation (recorded audit trail)
    MUTATIVE = "MUTATIVE"            # System mutation (order execution, risk updates)


class ToolEffects(BaseModel):
    """Capabilities and side-effects profile reverse-engineered from REA."""
    permission_tier: PermissionTier = PermissionTier.OBSERVATIONAL
    mutates_target: bool = False
    mutates_session: bool = False
    writes_filesystem: bool = False
    may_discard_data: bool = False
    idempotent: bool = True


class CognitivePhase(StrEnum):
    DECOMPILE = "DECOMPILE"      # Phase 1: Read raw account telemetry and market data without hallucination
    UNDERSTAND = "UNDERSTAND"    # Phase 2: Derive math and infer market thesis with invalidation bounds
    RECREATE = "RECREATE"        # Phase 3: Controlled risk check and broker order execution


class Observation(BaseModel):
    """Immutable factual observation read directly from live broker or market data."""
    id: str = Field(default_factory=lambda: f"obs_{uuid.uuid4().hex[:8]}")
    source: str  # e.g., "AlpacaBroker", "QuantScreener", "RegimeSpecialist", "PatternMemory"
    facet: str   # e.g., "account_balance", "open_positions", "quant_metrics", "macro_regime"
    data: dict[str, Any]
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MathematicalDerivation(BaseModel):
    """Deterministic mathematical computation derived from observations."""
    id: str = Field(default_factory=lambda: f"der_{uuid.uuid4().hex[:8]}")
    metric_name: str  # e.g., "protected_cash_reserve", "kelly_position_size", "risk_reward_ratio"
    formula: str      # e.g., "PeakEquity - (TotalProfit * 0.50)", "(TakeProfit - Entry) / (Entry - StopLoss)"
    inputs: dict[str, float]
    value: float
    unit: str = "USD"


class TradingInference(BaseModel):
    """Probabilistic hypothesis inferred from observations and mathematical derivations."""
    id: str = Field(default_factory=lambda: f"inf_{uuid.uuid4().hex[:8]}")
    thesis: str
    invalidation_condition: str  # Explicit condition that proves the thesis wrong
    invalidation_price: Optional[float] = None
    target_price: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)
    time_horizon_bars: int = 5
    supporting_observation_ids: list[str] = Field(default_factory=list)


class RiskLimitation(BaseModel):
    """Identified uncertainty, operational blindspot, or negative pattern warning."""
    id: str = Field(default_factory=lambda: f"lim_{uuid.uuid4().hex[:8]}")
    factor: str
    severity: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH", "CRITICAL_VETO"
    mitigation: str
    source_warning: Optional[str] = None


class TradingDecision(BaseModel):
    """Authoritative decision synthesized from the Evidence Ledger."""
    rating: str               # "Buy", "Overweight", "Hold", "Underweight", "Sell"
    action: str               # "BUY", "SELL", "HOLD"
    conviction: float = Field(ge=0.0, le=1.0)
    entry_price: float
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    suggested_quantity: float = 0.0
    rationale: str


class EvidenceLedger(BaseModel):
    """Canonical Evidence Ledger representing an agent's complete audit trail.
    
    Adheres to REA principles:
    - Structured distinction between what is observed, derived, inferred, and unknown.
    - Immutable SHA-256 fingerprint for tamper-proof verification.
    """
    ledger_id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}")
    symbol: str
    phase: CognitivePhase = CognitivePhase.DECOMPILE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    observations: list[Observation] = Field(default_factory=list)
    derivations: list[MathematicalDerivation] = Field(default_factory=list)
    inferences: list[TradingInference] = Field(default_factory=list)
    limitations: list[RiskLimitation] = Field(default_factory=list)
    decision: Optional[TradingDecision] = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    def compute_fingerprint(self) -> str:
        """Compute SHA-256 digest of normalized observations and derivations."""
        raw_payload = {
            "symbol": self.symbol,
            "observations": [o.model_dump(mode="json") for o in self.observations],
            "derivations": [d.model_dump(mode="json") for d in self.derivations],
            "decision": self.decision.model_dump(mode="json") if self.decision else None,
        }
        serialized = json.dumps(raw_payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------
# MCP Tool Calling Schema Definitions
# ------------------------------------------------------------------------------

class InspectPortfolioArgs(BaseModel):
    include_positions: bool = True


class InspectPortfolioResult(BaseModel):
    cash: float
    equity: float
    buying_power: float
    protected_cash_reserve: float
    unrestricted_free_cash: float
    open_positions: list[dict[str, Any]]
    drawdown_pct: float


class InspectMarketArgs(BaseModel):
    symbol: str


class InspectMarketResult(BaseModel):
    symbol: str
    current_price: float
    regime: str
    prudence_status: str
    prudence_directive: str
    volatility_score: float
    web_sentiment_score: Optional[float] = None
    fear_greed_index: Optional[int] = None
    quant_setup: Optional[str] = None
    quant_score: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    historical_lessons: list[str] = Field(default_factory=list)


class ValidateRiskArgs(BaseModel):
    symbol: str
    action: str
    current_price: float
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    confidence: float = 0.5


class ValidateRiskResult(BaseModel):
    is_approved: bool
    adjusted_quantity: float
    risk_multiplier: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    rejection_reasons: list[str] = Field(default_factory=list)
    warning_notes: list[str] = Field(default_factory=list)


class ExecuteOrderArgs(BaseModel):
    symbol: str
    action: str  # "BUY", "SELL"
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    ledger_id: str


class ExecuteOrderResult(BaseModel):
    success: bool
    order_id: Optional[str] = None
    symbol: str
    action: str
    executed_quantity: float
    filled_price: Optional[float] = None
    status: str
    error_message: Optional[str] = None


class SimulateOrderArgs(BaseModel):
    symbol: str
    action: str = "BUY"
    nominal_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    historical_win_rate: float = 0.55


class SimulateOrderResult(BaseModel):
    simulated_fill_price: float
    effective_stop_loss: float
    effective_take_profit: float
    estimated_slippage_dollars: float
    estimated_commission_dollars: float
    net_reward_dollars: float
    net_risk_dollars: float
    net_risk_reward_ratio: float
    statistical_expectancy: float
    is_positive_expectancy: bool
    is_convex_asymmetric: bool
    rejection_reason: Optional[str] = None


class AdmissionTicket(BaseModel):
    """REA-inspired admission certificate required to execute state mutations."""
    ticket_id: str = Field(default_factory=lambda: f"adm_{uuid.uuid4().hex[:10]}")
    ledger_id: str
    tool_name: str
    is_admitted: bool
    rejection_reason: Optional[str] = None
    fingerprint: str
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Explicit capability audit for MCP tools reverse-engineered from REA
TOOL_EFFECTS_REGISTRY: dict[str, ToolEffects] = {
    "inspect_portfolio": ToolEffects(
        permission_tier=PermissionTier.OBSERVATIONAL,
        mutates_target=False,
        mutates_session=False,
        idempotent=True,
    ),
    "inspect_market": ToolEffects(
        permission_tier=PermissionTier.OBSERVATIONAL,
        mutates_target=False,
        mutates_session=False,
        idempotent=True,
    ),
    "simulate_order_execution": ToolEffects(
        permission_tier=PermissionTier.OBSERVATIONAL,
        mutates_target=False,
        mutates_session=False,
        idempotent=True,
    ),
    "query_evidence_ledger": ToolEffects(
        permission_tier=PermissionTier.OBSERVATIONAL,
        mutates_target=False,
        mutates_session=False,
        idempotent=True,
    ),
    "deliberate_evidence_ledger": ToolEffects(
        permission_tier=PermissionTier.DELIBERATIVE,
        mutates_target=False,
        mutates_session=True,  # Mutates session ledger
        idempotent=False,
    ),
    "validate_risk": ToolEffects(
        permission_tier=PermissionTier.MUTATIVE,
        mutates_target=False,
        mutates_session=True,
        idempotent=True,
    ),
    "execute_order": ToolEffects(
        permission_tier=PermissionTier.MUTATIVE,
        mutates_target=True,  # Mutates account positions/equity on broker
        mutates_session=True,
        may_discard_data=False,
        idempotent=False,
    ),
}

