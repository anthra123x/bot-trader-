"""Unit & Integration Tests for Reverse-Engineered Cognition Architecture.

Validates the 4 core pillars:
1. Elite CLI Agentic Loops (Claude Code / Aider):
   - Scratchpad reasoning phases, deterministic algorithmic fallback, invalidation bounds.
2. Institutional Execution Guards (Hummingbot & Freqtrade):
   - SpreadGuard, PriceDriftGuard, ConsecutiveLossCooldownGuard, OrderLifecycleTracker.
3. Backtesting & Execution Realism (Backtrader & VectorBT):
   - Directional slippage, exchange commission friction, mathematical expectancy, net convexity.
4. REA MCP Permission Isolation & External Mutation Admission:
   - Tool capability matrix (TOOL_EFFECTS_REGISTRY), admission tickets, fail-closed gatekeeping.
"""

from unittest.mock import MagicMock, patch
import pytest

from bot.models import AccountBalance, AssetClass, Order, OrderSide, OrderStatus, OrderType, Position, RiskValidationResult
from bot.reasoning.agent_brain import AgentBrain
from bot.reasoning.contracts import (
    CognitivePhase,
    EvidenceLedger,
    Observation,
    PermissionTier,
    TOOL_EFFECTS_REGISTRY,
    TradingDecision,
)
from bot.reasoning.mcp_adapter import MCPTradingAdapter
from bot.risk.institutional_guard import GuardEvaluationResult, InstitutionalExecutionGuard, OrderState
from bot.risk.order_simulator import ExecutionSimulator, OrderSimulationResult
from bot.risk.quant_screener import QuantScreenResult
from tests.test_reasoning_engine import MockBroker


@pytest.fixture
def broker():
    return MockBroker(cash=50000.0, equity=100000.0)


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.save_debate.return_value = 1
    db.save_order.return_value = True
    return db


@pytest.fixture
def institutional_guard():
    return InstitutionalExecutionGuard(
        max_spread_pct=0.50,
        max_price_drift_pct=1.25,
        cooldown_seconds=1800,
        max_consecutive_losses=2,
    )


@pytest.fixture
def execution_simulator():
    return ExecutionSimulator(min_risk_reward_ratio=2.0)


@pytest.fixture
def brain(broker, mock_db, institutional_guard, execution_simulator):
    return AgentBrain(
        broker=broker,
        db=mock_db,
        institutional_guard=institutional_guard,
        execution_simulator=execution_simulator,
    )


@pytest.fixture
def mcp_adapter(brain, institutional_guard, execution_simulator):
    return MCPTradingAdapter(
        brain=brain,
        institutional_guard=institutional_guard,
        execution_simulator=execution_simulator,
    )


# ==============================================================================
# 1. ELITE CLI AGENTIC LOOPS (Claude Code / Aider)
# ==============================================================================

def test_cognitive_loop_generates_invalidation_and_expectancy(brain):
    """Verify that cognitive deliberation produces explicit invalidation bounds and expectancy derivations."""
    screen_res = QuantScreenResult(
        symbol="BTC/USD",
        bias="BUY",
        score=85.0,
        setup_name="Bullish Momentum Breakout",
        rationale="Strong RSI divergence and volume spike",
        current_price=65000.0,
        suggested_stop_loss=62500.0,
        suggested_take_profit=71000.0,
        risk_reward_ratio=2.40,
        is_actionable=True,
        metrics={"rsi": 54.0, "atr": 1200.0},
    )

    deliberated = brain.deliberate_with_cognitive_loop(
        symbol="BTC/USD",
        screen_res=screen_res,
    )

    assert deliberated.phase == CognitivePhase.UNDERSTAND
    assert deliberated.decision is not None
    assert deliberated.decision.action == "BUY"
    assert deliberated.decision.suggested_stop_loss == 62500.0
    assert deliberated.decision.suggested_take_profit == 71000.0

    # Verify mathematical expectancy derivation exists
    exp_derivations = [d for d in deliberated.derivations if d.metric_name == "statistical_expectancy"]
    assert len(exp_derivations) == 1
    assert exp_derivations[0].value > 0.0

    # Verify invalidation condition exists in inferences
    assert len(deliberated.inferences) >= 1
    assert deliberated.inferences[0].invalidation_price == 62500.0
    assert "invalidation" in deliberated.inferences[0].invalidation_condition.lower()


def test_cognitive_loop_downgrades_on_negative_expectancy(brain):
    """Verify that a candidate trade with negative expectancy or poor net R:R is downgraded to HOLD."""
    # Bad setup: high risk, tiny profit target
    screen_res = QuantScreenResult(
        symbol="ETH/USD",
        bias="BUY",
        score=75.0,
        setup_name="Weak Bounce",
        rationale="Questionable momentum",
        current_price=3000.0,
        suggested_stop_loss=2900.0,  # Risk: $100
        suggested_take_profit=3050.0, # Reward: $50 -> Net R:R ~0.50x (severely non-convex)
        risk_reward_ratio=0.50,
        is_actionable=True,
        metrics={"rsi": 48.0, "atr": 50.0},
    )

    deliberated = brain.deliberate_with_cognitive_loop(
        symbol="ETH/USD",
        screen_res=screen_res,
    )

    assert deliberated.decision.action == "HOLD"
    assert any(lim.factor == "SimulationExpectancyVeto" for lim in deliberated.limitations)


# ==============================================================================
# 2. INSTITUTIONAL EXECUTION GUARDS (Hummingbot & Freqtrade)
# ==============================================================================

def test_spread_guard_blocks_toxic_order_flow(institutional_guard):
    """Hummingbot: Verify SpreadGuard vetoes wide bid-ask spreads (>0.50%)."""
    # Acceptable spread (0.10%)
    ok, spread_pct, err = institutional_guard.evaluate_spread("BTC/USD", bid_price=64950.0, ask_price=65015.0)
    assert ok is True
    assert err is None
    assert spread_pct < 0.50

    # Toxic illiquid spread (1.23%)
    ok_bad, spread_pct_bad, err_bad = institutional_guard.evaluate_spread("ILLIQUID/USD", bid_price=10.0, ask_price=10.15)
    assert ok_bad is False
    assert "SpreadGuard VETO" in err_bad


def test_price_drift_guard_blocks_slippage(institutional_guard):
    """Hummingbot: Verify PriceDriftGuard blocks execution when price drifted adversely (>1.25%)."""
    # Acceptable drift (0.3%)
    ok, drift, err = institutional_guard.evaluate_price_drift("SOL/USD", screen_price=150.0, current_price=150.45, side=OrderSide.BUY)
    assert ok is True

    # Adverse drift (2.0%)
    ok_bad, drift_bad, err_bad = institutional_guard.evaluate_price_drift("SOL/USD", screen_price=150.0, current_price=153.0, side=OrderSide.BUY)
    assert ok_bad is False
    assert "PriceDriftGuard VETO" in err_bad


def test_consecutive_loss_cooldown_guard(institutional_guard):
    """Freqtrade: Verify consecutive losses trigger dynamic lockout period."""
    symbol = "DOGE/USD"
    # First loss: standard 30-min cooldown
    institutional_guard.register_trade_outcome(symbol, is_loss=True)
    in_cd, rem = institutional_guard.is_in_cooldown(symbol)
    assert in_cd is True
    assert 0 < rem <= 1800

    # Win resets cooldown
    institutional_guard.register_trade_outcome(symbol, is_loss=False)
    in_cd2, rem2 = institutional_guard.is_in_cooldown(symbol)
    assert in_cd2 is False
    assert rem2 == 0.0

    # Two consecutive losses: extended 1-hour lockout
    institutional_guard.register_trade_outcome(symbol, is_loss=True)
    institutional_guard.register_trade_outcome(symbol, is_loss=True)
    in_cd3, rem3 = institutional_guard.is_in_cooldown(symbol)
    assert in_cd3 is True
    assert rem3 > 1800  # Extended lockout


def test_order_lifecycle_state_machine(institutional_guard):
    """Hummingbot: Verify order transitions adhere to state machine invariants."""
    order_id = "ord_test_123"
    assert institutional_guard.get_order_state(order_id) == OrderState.PENDING_CREATE

    assert institutional_guard.transition_order_state(order_id, OrderState.SUBMITTED) is True
    assert institutional_guard.get_order_state(order_id) == OrderState.SUBMITTED

    assert institutional_guard.transition_order_state(order_id, OrderState.FILLED) is True
    assert institutional_guard.get_order_state(order_id) == OrderState.FILLED

    assert institutional_guard.transition_order_state(order_id, OrderState.EXITED) is True
    assert institutional_guard.get_order_state(order_id) == OrderState.EXITED

    # Invalid backward transition
    assert institutional_guard.transition_order_state(order_id, OrderState.OPEN) is False


# ==============================================================================
# 3. BACKTESTING & SIMULATION REALISM (Backtrader & VectorBT)
# ==============================================================================

def test_execution_simulator_applies_slippage_and_commission(execution_simulator):
    """Backtrader / VectorBT: Verify realistic directional slippage and commission deduction."""
    nominal_price = 100.0
    sl = 96.0   # Risk: $4.00
    tp = 110.0  # Reward: $10.00
    qty = 50.0

    sim = execution_simulator.simulate_trade_execution(
        nominal_price=nominal_price,
        suggested_stop_loss=sl,
        suggested_take_profit=tp,
        quantity=qty,
        side=OrderSide.BUY,
        asset_class=AssetClass.EQUITY,
        historical_win_rate=0.60,
    )

    # Slippage increases BUY fill price above nominal
    assert sim.simulated_fill_price > nominal_price
    # Commission fees estimated
    assert sim.estimated_commission_dollars > 0.0
    # Net R:R is positive and convex
    assert sim.net_risk_reward_ratio >= 2.0
    assert sim.is_positive_expectancy is True
    assert sim.is_convex_asymmetric is True


# ==============================================================================
# 4. REA MCP PERMISSION ISOLATION & ADMISSION CONTROLLER
# ==============================================================================

def test_mcp_tool_effects_registry_capabilities():
    """REA: Verify explicit capability profiling across all tools."""
    assert TOOL_EFFECTS_REGISTRY["inspect_portfolio"].permission_tier == PermissionTier.OBSERVATIONAL
    assert TOOL_EFFECTS_REGISTRY["inspect_portfolio"].mutates_target is False

    assert TOOL_EFFECTS_REGISTRY["deliberate_evidence_ledger"].permission_tier == PermissionTier.DELIBERATIVE
    assert TOOL_EFFECTS_REGISTRY["deliberate_evidence_ledger"].mutates_session is True

    assert TOOL_EFFECTS_REGISTRY["execute_order"].permission_tier == PermissionTier.MUTATIVE
    assert TOOL_EFFECTS_REGISTRY["execute_order"].mutates_target is True


def test_mcp_admission_controller_blocks_unauthorized_mutation(mcp_adapter):
    """REA: Verify mutative execution is blocked fail-closed when no valid EvidenceLedger exists."""
    res = mcp_adapter.execute_tool(
        "execute_order",
        {
            "symbol": "BTC/USD",
            "action": "BUY",
            "quantity": 0.1,
            "ledger_id": "non_existent_id",
        },
    )
    assert res.get("isError") is True
    assert "admission" in res
    assert res["admission"]["is_admitted"] is False
    assert "No active EvidenceLedger" in res["admission"]["rejection_reason"]


def test_mcp_admission_controller_admits_verified_mutation(mcp_adapter, brain):
    """REA: Verify mutative execution is admitted when ledger is deliberated and guards pass."""
    symbol = "BTC/USD"
    ledger = brain.decompile_state(symbol)
    ledger.observations.append(
        Observation(source="QuantScreener", facet="quant_metrics", data={"current_price": 65000.0})
    )
    deliberated = brain.deliberate_and_synthesize(
        ledger=ledger,
        candidate_rating="Buy",
        confidence=0.85,
        target_price=71000.0,
        stop_loss=62500.0,
        rationale="Verified institutional setup",
    )
    mcp_adapter.active_ledgers[deliberated.ledger_id] = deliberated
    mcp_adapter.active_ledgers[symbol] = deliberated

    with patch.object(
        brain.risk_engine,
        "evaluate",
        return_value=RiskValidationResult(
            is_approved=True,
            adjusted_quantity=0.1,
            risk_multiplier=1.0,
            stop_loss_price=62500.0,
            take_profit_price=71000.0,
        ),
    ):
        res = mcp_adapter.execute_tool(
            "execute_order",
            {
                "symbol": symbol,
                "action": "BUY",
                "quantity": 0.1,
                "ledger_id": deliberated.ledger_id,
            },
        )
        assert res.get("isError") is not True
        assert "admissionTicket" in res
        assert res["admissionTicket"]["is_admitted"] is True
