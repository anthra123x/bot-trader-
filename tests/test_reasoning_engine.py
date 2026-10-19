"""Comprehensive Unit & Integration Tests for REA-inspired ReasoningEngine & MCP Adapter."""

import json
from unittest.mock import MagicMock, patch
import pytest

from bot.brokers.base import BaseBroker
from bot.models import AccountBalance, AssetClass, Order, OrderSide, OrderStatus, OrderType, Position, RiskValidationResult
from bot.reasoning.agent_brain import AgentBrain, ReasoningEngine
from bot.reasoning.contracts import (
    CognitivePhase,
    EvidenceKind,
    EvidenceLedger,
    MathematicalDerivation,
    Observation,
    RiskLimitation,
    TradingDecision,
    TradingInference,
)
from bot.reasoning.mcp_adapter import MCPTradingAdapter
from bot.risk.regime_specialist import MarketPrudence, MarketRegime, RegimeAnalysis


class MockBroker(BaseBroker):
    """Deterministic mock broker for reasoning tests."""

    def __init__(self, cash: float = 25000.0, equity: float = 50000.0):
        self._balance = AccountBalance(
            cash=cash,
            equity=equity,
            buying_power=cash * 2.0,
            initial_day_equity=equity,
        )
        self._positions: list[Position] = []
        self.submitted_orders: list[Order] = []

    def get_account_balance(self) -> AccountBalance:
        return self._balance

    def get_positions(self) -> list[Position]:
        return self._positions

    def get_current_price(self, symbol: str) -> float:
        return 65000.0 if "BTC" in symbol else 150.0

    def get_position(self, symbol: str) -> Position | None:
        for p in self._positions:
            if p.symbol == symbol:
                return p
        return None

    def cancel_order(self, order_id: str) -> bool:
        return True

    def submit_order(self, order: Order) -> Order:
        order.status = OrderStatus.FILLED
        order.filled_price = self.get_current_price(order.symbol)
        self.submitted_orders.append(order)
        return order


@pytest.fixture
def mock_broker():
    return MockBroker()


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.save_debate.return_value = 1
    db.save_order.return_value = True
    return db


@pytest.fixture
def brain(mock_broker, mock_db):
    return AgentBrain(broker=mock_broker, db=mock_db)


@pytest.fixture
def mcp_adapter(brain):
    return MCPTradingAdapter(brain=brain)


# ==============================================================================
# 1. CONTRACTS & EVIDENCE LEDGER TESTS
# ==============================================================================

def test_evidence_ledger_fingerprint_and_immutability():
    """Verify that Evidence Ledger produces a deterministic SHA-256 fingerprint."""
    ledger = EvidenceLedger(symbol="BTC/USD")
    ledger.observations.append(
        Observation(source="AlpacaBroker", facet="account_balance", data={"cash": 10000.0, "equity": 20000.0})
    )
    ledger.derivations.append(
        MathematicalDerivation(
            metric_name="protected_cash_reserve",
            formula="PeakEquity - (TotalProfit * 0.50)",
            inputs={"equity": 20000.0},
            value=5000.0,
        )
    )

    fingerprint1 = ledger.compute_fingerprint()
    fingerprint2 = ledger.compute_fingerprint()

    assert fingerprint1 == fingerprint2
    assert len(fingerprint1) == 64  # SHA-256 hex string

    # Modifying content must change fingerprint
    ledger.observations.append(
        Observation(source="QuantScreener", facet="price", data={"price": 65000.0})
    )
    fingerprint3 = ledger.compute_fingerprint()
    assert fingerprint3 != fingerprint1


def test_contracts_json_serialization():
    """Verify clean Pydantic serialization adhering to REA evidence schema."""
    ledger = EvidenceLedger(symbol="ETH/USD")
    ledger.inferences.append(
        TradingInference(
            thesis="Bullish breakout continuation above 200 EMA",
            invalidation_condition="Price breaks below $3,400",
            invalidation_price=3400.0,
            confidence=0.85,
        )
    )
    raw = ledger.model_dump(mode="json")
    assert raw["symbol"] == "ETH/USD"
    assert raw["inferences"][0]["confidence"] == 0.85
    assert raw["inferences"][0]["invalidation_price"] == 3400.0


# ==============================================================================
# 2. AGENT BRAIN 3-PHASE COGNITIVE FLOW TESTS
# ==============================================================================

def test_brain_phase_1_decompile_state(brain):
    """Phase 1: Verify raw portfolio and market state are decompiled without hallucination."""
    ledger = brain.decompile_state("BTC/USD")

    assert ledger.symbol == "BTC/USD"
    assert ledger.phase == CognitivePhase.DECOMPILE
    assert len(ledger.observations) >= 2  # account_balance, open_positions

    # Check that 50% quarantine reserve derivation exists
    reserve_derivations = [d for d in ledger.derivations if d.metric_name == "protected_cash_reserve"]
    assert len(reserve_derivations) == 1
    assert reserve_derivations[0].value >= 0.0

    # Free cash derivation
    free_cash = [d for d in ledger.derivations if d.metric_name == "unrestricted_free_cash"]
    assert len(free_cash) == 1
    assert free_cash[0].value <= 25000.0


def test_brain_phase_2_deliberate_normal_buy(brain):
    """Phase 2: High-conviction thesis generates valid TradingDecision with invalidation price."""
    ledger = brain.decompile_state("BTC/USD")
    deliberated = brain.deliberate_and_synthesize(
        ledger=ledger,
        candidate_rating="Buy",
        confidence=0.88,
        target_price=70000.0,
        stop_loss=62000.0,
        rationale="Confluence of quantitative breakout and macro stability.",
    )

    assert deliberated.phase == CognitivePhase.UNDERSTAND
    assert deliberated.decision is not None
    assert deliberated.decision.action == "BUY"
    assert deliberated.decision.rating == "Buy"
    assert deliberated.decision.conviction == 0.88
    assert deliberated.decision.suggested_stop_loss == 62000.0

    # Ensure inference was created with explicit invalidation level
    assert len(deliberated.inferences) == 1
    assert deliberated.inferences[0].invalidation_price == 62000.0
    assert "invalidation" in deliberated.inferences[0].invalidation_condition.lower()


def test_brain_phase_2_deliberate_critical_veto(brain):
    """Phase 2: Critical Veto (e.g. Market Research Veto) enforces capital preservation to 100% Cash."""
    ledger = brain.decompile_state("BTC/USD")
    # Inject critical veto
    ledger.limitations.append(
        RiskLimitation(
            factor="MarketPrudenceVeto",
            severity="CRITICAL_VETO",
            mitigation="Veto any Long; 100% Cash",
            source_warning="Adverse market conditions",
        )
    )

    deliberated = brain.deliberate_and_synthesize(
        ledger=ledger,
        candidate_rating="Buy",
        confidence=0.90,
        target_price=75000.0,
        stop_loss=60000.0,
    )

    assert deliberated.decision is not None
    assert deliberated.decision.action == "HOLD"
    assert deliberated.decision.rating == "Hold"
    assert deliberated.decision.suggested_quantity == 0.0
    assert "VETO_PREVENTIVO" in deliberated.decision.rationale


def test_brain_phase_2_deliberate_low_conviction_gate(brain):
    """Phase 2: Low conviction (<0.65) is automatically converted to HOLD."""
    ledger = brain.decompile_state("BTC/USD")
    deliberated = brain.deliberate_and_synthesize(
        ledger=ledger,
        candidate_rating="Buy",
        confidence=0.55,
        target_price=68000.0,
        stop_loss=64000.0,
    )

    assert deliberated.decision.action == "HOLD"
    assert deliberated.decision.rating == "Hold"
    assert "insuficiente" in deliberated.decision.rationale.lower()


def test_brain_phase_3_recreate_hold_no_order(brain, mock_broker):
    """Phase 3: Recreate/Execute on HOLD decision dispatches 0 orders to broker."""
    ledger = brain.decompile_state("BTC/USD")
    deliberated = brain.deliberate_and_synthesize(ledger, "Hold", 0.50)

    order, executed_ledger = brain.recreate_and_execute(deliberated)

    assert order is None
    assert len(mock_broker.submitted_orders) == 0
    assert executed_ledger.phase == CognitivePhase.RECREATE


def test_brain_phase_3_recreate_buy_order_executed(brain, mock_broker, mock_db):
    """Phase 3: Approved BUY order is validated and executed on broker with audit log."""
    ledger = brain.decompile_state("BTC/USD")
    # Ensure current price is populated in observations
    ledger.observations.append(
        Observation(source="QuantScreener", facet="quant_metrics", data={"current_price": 65000.0})
    )
    deliberated = brain.deliberate_and_synthesize(
        ledger=ledger,
        candidate_rating="Buy",
        confidence=0.85,
        target_price=70000.0,
        stop_loss=63000.0,
        rationale="Strong institutional edge.",
    )

    with patch.object(
        brain.risk_engine,
        "evaluate",
        return_value=RiskValidationResult(
            is_approved=True,
            adjusted_quantity=0.15,
            risk_multiplier=1.0,
            stop_loss_price=63000.0,
            take_profit_price=70000.0,
        ),
    ):
        order, executed_ledger = brain.recreate_and_execute(deliberated)

        assert order is not None
        assert order.symbol == "BTC/USD"
        assert order.side == OrderSide.BUY
        assert order.quantity == 0.15
        assert len(mock_broker.submitted_orders) == 1
        assert mock_db.save_debate.called
        assert mock_db.save_order.called


# ==============================================================================
# 3. MCP ADAPTER & TOOL CALLING TESTS
# ==============================================================================

def test_mcp_adapter_tool_definitions(mcp_adapter):
    """Verify tool schemas are valid and contain all required REA trading tools."""
    tools = mcp_adapter.get_tool_definitions()
    tool_names = [t["name"] for t in tools]

    assert "inspect_portfolio" in tool_names
    assert "inspect_market" in tool_names
    assert "deliberate_evidence_ledger" in tool_names
    assert "validate_risk" in tool_names
    assert "execute_order" in tool_names

    for t in tools:
        assert "inputSchema" in t
        assert t["inputSchema"]["type"] == "object"


def test_mcp_execute_tool_inspect_portfolio(mcp_adapter):
    """Verify inspect_portfolio tool executes and returns structured telemetry."""
    res = mcp_adapter.execute_tool("inspect_portfolio", {"include_positions": True})

    assert "structuredContent" in res
    assert res["structuredContent"]["cash"] == 25000.0
    assert res["structuredContent"]["equity"] == 50000.0
    assert "protected_cash_reserve" in res["structuredContent"]
    assert "unrestricted_free_cash" in res["structuredContent"]


def test_mcp_execute_tool_inspect_market(mcp_adapter):
    """Verify inspect_market tool executes and returns market telemetry."""
    res = mcp_adapter.execute_tool("inspect_market", {"symbol": "BTC/USD"})

    assert "structuredContent" in res
    assert res["structuredContent"]["symbol"] == "BTC/USD"
    assert "regime" in res["structuredContent"]
    assert "prudence_status" in res["structuredContent"]


def test_mcp_execute_tool_deliberate_and_execute(mcp_adapter, mock_broker):
    """Verify full MCP round-trip: deliberate_evidence_ledger -> execute_order."""
    # 1. Deliberate
    delib_res = mcp_adapter.execute_tool(
        "deliberate_evidence_ledger",
        {
            "symbol": "BTC/USD",
            "rating": "Buy",
            "conviction": 0.85,
            "invalidation_price": 63000.0,
            "target_price": 70000.0,
            "thesis_rationale": "Empirical trend confluence across indicators.",
        },
    )
    ledger_id = delib_res["structuredContent"]["ledger_id"]
    assert ledger_id.startswith("ev_")

    # 2. Execute via MCP
    with patch.object(
        mcp_adapter.brain.risk_engine,
        "evaluate",
        return_value=RiskValidationResult(
            is_approved=True,
            adjusted_quantity=0.1,
            risk_multiplier=1.0,
            stop_loss_price=63000.0,
            take_profit_price=70000.0,
        ),
    ):
        exec_res = mcp_adapter.execute_tool(
            "execute_order",
            {
                "symbol": "BTC/USD",
                "action": "BUY",
                "quantity": 0.1,
                "ledger_id": ledger_id,
            },
        )
        assert exec_res["structuredContent"]["success"] is True
        assert exec_res["structuredContent"]["executed_quantity"] == 0.1
        assert len(mock_broker.submitted_orders) == 1


def test_mcp_unknown_tool_handling(mcp_adapter):
    """Verify unknown tool gracefully returns standard error diagnostic."""
    res = mcp_adapter.execute_tool("non_existent_tool", {})
    assert res["isError"] is True
    assert "Unknown tool" in res["content"][0]["text"]


# ==============================================================================
# 4. BOT RUNNER INTEGRATION TEST
# ==============================================================================

def test_bot_runner_has_agent_brain(mock_broker, mock_db):
    """Verify BotRunner seamlessly instantiates AgentBrain and MCPTradingAdapter."""
    from bot.runner import BotRunner
    runner = BotRunner(broker=mock_broker, db=mock_db)

    assert hasattr(runner, "brain")
    assert isinstance(runner.brain, AgentBrain)
    assert hasattr(runner, "mcp_adapter")
    assert isinstance(runner.mcp_adapter, MCPTradingAdapter)
