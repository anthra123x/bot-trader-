"""Unit and integration tests for Phase 4: Speed (Parallel Pipeline & Batching).

Tests:
- Parallel execution of QuantScreener across multiple tickers
- Asynchronous wrapper screen_universe_async
- Circuit Breaker HALT short-circuit in run_portfolio_cycle
- Capacity limit early short-circuit (no tokens/compute spent when portfolio is full)
- Alpha-prioritization ranking of actionable setups
- Re-use of pre-screened results in run_symbol
"""

import asyncio
from unittest.mock import MagicMock, patch
import pytest

from bot.config import BotConfig, BotRiskConfig
from bot.brokers.paper import PaperBroker
from bot.models import (
    AccountBalance,
    AssetClass,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)
from bot.risk.quant_screener import QuantScreener, QuantScreenResult
from bot.runner import BotRunner
from bot.learning.portfolio_intelligence import PORTFOLIO_INTELLIGENCE


@pytest.fixture
def sample_screen_results() -> dict[str, QuantScreenResult]:
    return {
        "BTC-USD": QuantScreenResult(
            symbol="BTC-USD",
            is_actionable=True,
            bias="BUY",
            score=70.0,
            setup_name="TREND_PULLBACK",
            current_price=90_000.0,
            suggested_stop_loss=87_000.0,
            suggested_take_profit=96_000.0,
            risk_reward_ratio=2.0,
            metrics={"rsi": 48.0, "atr_pct": 2.1},
            rationale="Valid trend pullback setup",
        ),
        "ETH-USD": QuantScreenResult(
            symbol="ETH-USD",
            is_actionable=True,
            bias="BUY",
            score=65.0,
            setup_name="MOMENTUM_BREAKOUT",
            current_price=3_200.0,
            suggested_stop_loss=3_050.0,
            suggested_take_profit=3_500.0,
            risk_reward_ratio=2.0,
            metrics={"rsi": 54.0, "atr_pct": 2.5},
            rationale="Breakout above consolidation",
        ),
        "SOL-USD": QuantScreenResult(
            symbol="SOL-USD",
            is_actionable=False,
            bias="NEUTRAL",
            score=45.0,
            setup_name="CHOPPY_RANGE",
            current_price=180.0,
            suggested_stop_loss=170.0,
            suggested_take_profit=200.0,
            risk_reward_ratio=2.0,
            metrics={"rsi": 51.0, "atr_pct": 3.0},
            rationale="Range bound chop",
        ),
    }


def test_quant_screener_parallel_screening(sample_screen_results):
    screener = QuantScreener()
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "INVALID-TICKER"]

    def mock_screen(sym: str, date: str) -> QuantScreenResult:
        if sym in sample_screen_results:
            return sample_screen_results[sym]
        raise ValueError(f"Unknown symbol: {sym}")

    with patch.object(screener, "screen", side_effect=mock_screen):
        with patch.object(screener, "get_btc_macro_health", return_value={}):
            results = screener.screen_universe_parallel(symbols, "2026-10-20", max_workers=4)

    assert len(results) == 4
    result_map = dict(results)
    assert result_map["BTC-USD"].is_actionable is True
    assert result_map["ETH-USD"].is_actionable is True
    assert result_map["SOL-USD"].is_actionable is False
    # Graceful fallback on exception
    assert result_map["INVALID-TICKER"].is_actionable is False
    assert result_map["INVALID-TICKER"].setup_name == "Error / Fallback"


@pytest.mark.anyio
async def test_quant_screener_async_screening(sample_screen_results):
    screener = QuantScreener()
    symbols = ["BTC-USD", "ETH-USD"]

    with patch.object(screener, "screen", side_effect=lambda s, d: sample_screen_results[s]):
        with patch.object(screener, "get_btc_macro_health", return_value={}):
            results = await screener.screen_universe_async(symbols, "2026-10-20", max_workers=2)

    assert len(results) == 2
    res_dict = dict(results)
    assert res_dict["BTC-USD"].score == 70.0
    assert res_dict["ETH-USD"].score == 65.0


def test_run_portfolio_cycle_circuit_breaker_halt():
    config = BotConfig(risk=BotRiskConfig())
    broker = PaperBroker()
    runner = BotRunner(config=config, broker=broker, db=MagicMock())

    with patch.object(PORTFOLIO_INTELLIGENCE, "should_halt_trading", return_value=True):
        with patch.object(runner.quant_screener, "screen_universe_parallel") as mock_screen:
            orders = runner.run_portfolio_cycle("2026-10-20")

    assert orders == []
    # Screener should NOT even be invoked
    mock_screen.assert_not_called()


def test_run_portfolio_cycle_capacity_short_circuit():
    config = BotConfig(
        risk=BotRiskConfig(max_concurrent_crypto_positions=2),
        watchlist_crypto=["BTC-USD", "ETH-USD", "SOL-USD"],
        watchlist_equities=["AAPL"],
    )
    broker = PaperBroker()
    runner = BotRunner(config=config, broker=broker, db=MagicMock())

    # Simulate 2 active crypto positions filling the limit
    pos1 = Position(symbol="BTC-USD", asset_class=AssetClass.CRYPTO, quantity=1.0, entry_price=90_000.0, current_price=91_000.0)
    pos2 = Position(symbol="ETH-USD", asset_class=AssetClass.CRYPTO, quantity=10.0, entry_price=3_200.0, current_price=3_250.0)
    broker.get_positions = MagicMock(return_value=[pos1, pos2])

    # Simulate equity market closed (e.g. weekend or overnight)
    with patch.object(runner, "is_equity_market_open", return_value=False):
        with patch.object(runner, "manage_open_positions", return_value=[]):
            with patch.object(PORTFOLIO_INTELLIGENCE, "should_halt_trading", return_value=False):
                with patch.object(runner.quant_screener, "screen_universe_parallel") as mock_screen:
                    orders = runner.run_portfolio_cycle("2026-10-20")

    # Short-circuited before screening because capacity is 100% full
    mock_screen.assert_not_called()
    assert orders == []


def test_dynamic_scan_order_prioritization_and_prescreen(sample_screen_results):
    config = BotConfig(
        risk=BotRiskConfig(max_concurrent_crypto_positions=1),  # Only 1 slot available!
        watchlist_crypto=["BTC-USD", "ETH-USD"],
        watchlist_equities=[],
    )
    broker = PaperBroker()
    runner = BotRunner(config=config, broker=broker, db=MagicMock())

    # Empty positions, 1 slot available
    broker.get_positions = MagicMock(return_value=[])

    # Pre-configure PortfolioIntelligence:
    # BTC-USD has Alpha Edge multiplier (e.g. 1.35x), ETH-USD has lower (1.0x)
    records = [
        {"symbol": "BTC-USD", "pnl_pct": 5.0, "outcome": "WIN"},
        {"symbol": "BTC-USD", "pnl_pct": 4.0, "outcome": "WIN"},
        {"symbol": "ETH-USD", "pnl_pct": 1.0, "outcome": "WIN"},
    ]
    PORTFOLIO_INTELLIGENCE.compute_metrics(records)

    screened_list = [
        ("ETH-USD", sample_screen_results["ETH-USD"]),  # score 65 * 1.0 = 65
        ("BTC-USD", sample_screen_results["BTC-USD"]),  # score 70 * 1.35 = 94.5 (Top Priority!)
    ]

    executed_symbols = []

    def mock_run_symbol(sym, trade_date=None, pre_screen_res=None):
        executed_symbols.append(sym)
        assert pre_screen_res is not None
        assert pre_screen_res.symbol == sym
        # Simulate that executing BTC-USD fills the only available crypto slot
        broker.get_positions = MagicMock(return_value=[
            Position(symbol=sym, asset_class=AssetClass.CRYPTO, quantity=1.0, entry_price=90_000.0, current_price=90_000.0)
        ])
        return Order(
            order_id="test1234",
            symbol=sym,
            asset_class=AssetClass.CRYPTO,
            side=OrderSide.BUY,
            quantity=1.0,
            status=OrderStatus.FILLED,
        )

    with patch.object(runner, "is_equity_market_open", return_value=False):
        with patch.object(runner, "manage_open_positions", return_value=[]):
            with patch.object(PORTFOLIO_INTELLIGENCE, "should_halt_trading", return_value=False):
                with patch.object(runner.quant_screener, "screen_universe_parallel", return_value=screened_list):
                    with patch.object(runner, "run_symbol", side_effect=mock_run_symbol):
                        orders = runner.run_portfolio_cycle("2026-10-20")

    # BTC-USD should have been prioritized FIRST due to its higher alpha multiplier and score
    assert len(executed_symbols) == 1
    assert executed_symbols[0] == "BTC-USD"
    # ETH-USD should have been short-circuited because BTC filled the max_concurrent_crypto_positions=1 slot!
    assert "ETH-USD" not in executed_symbols
    assert len(orders) == 1
    assert orders[0].symbol == "BTC-USD"
