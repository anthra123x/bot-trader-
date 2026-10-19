"""Data models and value objects for the trading bot."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class AssetClass(StrEnum):
    EQUITY = "EQUITY"
    CRYPTO = "CRYPTO"


class AccountBalance(BaseModel):
    """Account balance and buying power representation."""
    currency: str = "USD"
    cash: float
    equity: float
    buying_power: float
    initial_day_equity: float = 0.0

    @property
    def daily_pnl(self) -> float:
        if self.initial_day_equity <= 0:
            return 0.0
        return self.equity - self.initial_day_equity

    @property
    def daily_drawdown_pct(self) -> float:
        if self.initial_day_equity <= 0:
            return 0.0
        drawdown = (self.initial_day_equity - self.equity) / self.initial_day_equity
        return max(0.0, drawdown * 100.0)


class Position(BaseModel):
    """Open position in an asset."""
    symbol: str
    asset_class: AssetClass
    quantity: float
    entry_price: float
    current_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    stop_loss_price: float | None = None
    take_profit_price: float | None = None

    def model_post_init(self, __context: Any) -> None:
        self.market_value = self.quantity * self.current_price
        cost_basis = self.quantity * self.entry_price
        self.unrealized_pnl = self.market_value - cost_basis
        if cost_basis > 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl / cost_basis) * 100.0


class TradeSignal(BaseModel):
    """Structured signal emitted by the multi-agent framework."""
    symbol: str
    asset_class: AssetClass
    trade_date: str
    rating: str  # Buy, Overweight, Hold, Underweight, Sell
    action: OrderSide
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    current_price: float
    target_price: float | None = None
    stop_loss: float | None = None
    reasoning: str = ""
    time_horizon_days: int = 5
    metrics: dict[str, float] = Field(default_factory=dict)


class Order(BaseModel):
    """Execution order representation."""
    order_id: str
    symbol: str
    asset_class: AssetClass
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float
    limit_price: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float | None = None
    filled_at: datetime | None = None
    rejection_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RiskValidationResult(BaseModel):
    """Result of validating a trade signal through the RiskEngine."""
    is_approved: bool
    adjusted_quantity: float = 0.0
    allocated_capital: float = 0.0
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    rejection_reasons: list[str] = Field(default_factory=list)
    risk_warnings: list[str] = Field(default_factory=list)


class TradeOutcome(BaseModel):
    """Result and features of a closed trade for post-mortem feedback and pattern learning."""
    trade_id: str
    symbol: str
    asset_class: AssetClass
    side: OrderSide
    entry_price: float
    exit_price: float
    quantity: float
    pnl_dollars: float
    pnl_pct: float
    outcome: str  # "WIN" or "LOSS"
    duration_hours: float = 0.0
    rsi_entry: float = 50.0
    ema_ratio: float = 1.0  # entry_price / EMA50
    volatility_atr_pct: float = 2.0
    market_regime: str = "NEUTRAL"
    mistake_signature: str = ""
    lesson_learned: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PatternCheckResult(BaseModel):
    """Result of comparing a prospective trade against known historical mistake patterns."""
    is_safe: bool = True
    risk_multiplier: float = 1.0  # 1.0 = normal, 0.5 = downsize, 0.0 = veto
    matched_mistake_tag: str | None = None
    similarity_score: float = 0.0
    warning_message: str | None = None
    learned_lesson: str | None = None
