"""Configuration settings for the trading bot."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()


class BotRiskConfig(BaseModel):
    """Hard-coded defensive risk limits with asymmetric profit expectation."""
    max_risk_per_trade_pct: float = Field(default=1.0, description="Max portfolio loss risk per trade (e.g. 1.0%)")
    max_position_size_pct: float = Field(default=6.0, description="Max capital allocation to single position (6.0%)")
    max_daily_drawdown_pct: float = Field(default=3.0, description="Kill switch threshold for daily loss (3%)")
    default_stop_loss_pct: float = Field(default=2.5, description="Fallback stop loss if LLM provides none (2.5%)")
    max_stop_loss_pct: float = Field(default=4.0, description="Maximum allowable stop loss distance (4.0%)")
    min_risk_reward_ratio: float = Field(default=2.5, description="Min reward/risk ratio (strictly >= 2.5:1 for positive mathematical expectancy)")
    max_overbought_rsi_buy: float = Field(default=64.0, description="Max RSI allowable for BUY entry (anti-bull-trap ceiling)")
    enable_macro_sma200_filter: bool = Field(default=True, description="Enforce macro 200 SMA trend filter for long positions")
    enable_defensive_regime_veto: bool = Field(default=True, description="Strictly veto aggressive BUYs during DEFENSIVE_BEAR_BREAKDOWN")
    # QuantConnect LEAN & Freqtrade Precision Enhancements
    enable_trailing_stop: bool = Field(default=True, description="Enable high-water mark trailing stop (QuantConnect LEAN)")
    trailing_stop_activation_pct: float = Field(default=1.2, description="Min profit % before trailing stop activates")
    trailing_stop_distance_pct: float = Field(default=1.0, description="Distance % behind peak price to lock in profits")
    cooldown_period_seconds: int = Field(default=1800, description="Cooldown seconds before re-entering an asset after loss (Freqtrade CooldownPeriod)")
    max_consecutive_stoplosses: int = Field(default=3, description="Consecutive stoplosses before triggering temporary halt (Freqtrade StoplossGuard)")
    # Tiered Partial Take-Profit (Scale-Out Profit Harvesting)
    enable_partial_take_profit: bool = Field(default=True, description="Enable partial scale-out profit harvesting")
    partial_take_profit_threshold_pct: float = Field(default=1.2, description="Profit % trigger for partial scale-out stage 1 (+1.2%)")
    partial_take_profit_ratio: float = Field(default=0.35, description="Fraction of position to sell on partial take profit (35%)")
    enable_multi_stage_tp: bool = Field(default=True, description="Enable secondary take profit tier (+3.0%)")
    stage2_take_profit_threshold_pct: float = Field(default=3.0, description="Profit % trigger for stage 2 take profit (+3.0%)")
    stage2_take_profit_ratio: float = Field(default=0.45, description="Fraction of remaining position to harvest at stage 2")
    # High-Conviction Institutional Quality Gate
    min_signal_confidence: float = Field(default=0.0, description="Minimum multi-agent conviction required for entry (0.0=disabled by default, 0.68 in runner)")
    # Correlated Exposure Limit (Systemic Risk Control)
    max_concurrent_crypto_positions: int = Field(default=4, description="Max simultaneous crypto positions to prevent systemic portfolio risk")
    # Dynamic Portfolio Balancing & Concentration Trim (Capital Rotation)
    enable_dynamic_rebalancing: bool = Field(default=True, description="Enable capital rebalancing when single asset exceeds concentration cap")
    max_concentration_threshold_pct: float = Field(default=20.0, description="Concentration cap where position trim/harvest is recommended")
    target_concentration_pct: float = Field(default=12.0, description="Target concentration after rebalancing trim")


class BotConfig(BaseModel):
    """Global Bot Configuration."""
    watchlist_equities: list[str] = Field(
        default_factory=lambda: ["NVDA", "AAPL", "MSFT", "SPY", "TSLA"]
    )
    watchlist_crypto: list[str] = Field(
        default_factory=lambda: [
            "BTC-USD",
            "ETH-USD",
            "SOL-USD",
            "AVAX-USD",
            "LINK-USD",
            "DOGE-USD",
            "NEAR-USD",
            "UNI-USD",
            "AAVE-USD",
            "LTC-USD",
        ]
    )
    risk: BotRiskConfig = Field(default_factory=BotRiskConfig)
    
    # Storage and paths
    data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TRADINGAGENTS_DATA_DIR", "data"))
    )
    
    # Notification webhooks (optional)
    telegram_bot_token: str | None = Field(default=os.getenv("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: str | None = Field(default=os.getenv("TELEGRAM_CHAT_ID"))
    discord_webhook_url: str | None = Field(default=os.getenv("DISCORD_WEBHOOK_URL"))
    
    # Broker selection: 'paper', 'alpaca', 'ccxt'
    broker_type: str = Field(default=os.getenv("BOT_BROKER_TYPE", "paper"))
    
    # TradingAgents base config override
    tradingagents_config: dict = Field(default_factory=lambda: DEFAULT_CONFIG.copy())

    def model_post_init(self, __context: Any) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


BOT_CONFIG = BotConfig()
