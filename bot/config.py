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
    max_risk_per_trade_pct: float = Field(default=1.5, description="Max portfolio loss risk per trade (1.5%)")
    max_position_size_pct: float = Field(default=25.0, description="Max capital allocation to single position (25.0%)")
    max_daily_drawdown_pct: float = Field(default=3.0, description="Kill switch threshold for daily loss (3%)")
    default_stop_loss_pct: float = Field(default=2.5, description="Fallback stop loss if LLM provides none (2.5%)")
    max_stop_loss_pct: float = Field(default=4.0, description="Maximum allowable stop loss distance (4.0%)")
    min_risk_reward_ratio: float = Field(default=2.5, description="Min reward/risk ratio (strictly >= 2.5:1 for positive mathematical expectancy)")
    max_overbought_rsi_buy: float = Field(default=64.0, description="Max RSI allowable for BUY entry (anti-bull-trap ceiling)")
    enable_macro_sma200_filter: bool = Field(default=True, description="Enforce macro 200 SMA trend filter for long positions")
    enable_defensive_regime_veto: bool = Field(default=True, description="Strictly veto aggressive BUYs during DEFENSIVE_BEAR_BREAKDOWN")
    screener_conviction_threshold: float = Field(default=55.0, description="Minimum algorithmic screener score to awaken LLM committee (55.0)")
    # QuantConnect LEAN & Freqtrade Precision Enhancements
    enable_trailing_stop: bool = Field(default=True, description="Enable high-water mark trailing stop (QuantConnect LEAN)")
    trailing_stop_activation_pct: float = Field(default=1.2, description="Min profit % before trailing stop activates")
    trailing_stop_distance_pct: float = Field(default=1.0, description="Distance % behind peak price to lock in profits")
    cooldown_period_seconds: int = Field(default=1800, description="Cooldown seconds before re-entering an asset after loss (Freqtrade CooldownPeriod)")
    max_consecutive_stoplosses: int = Field(default=3, description="Consecutive stoplosses before triggering temporary halt (Freqtrade StoplossGuard)")
    # Tiered Partial Take-Profit (Scale-Out Profit Harvesting)
    enable_partial_take_profit: bool = Field(default=True, description="Enable partial scale-out profit harvesting")
    partial_take_profit_threshold_pct: float = Field(default=2.2, description="Profit % trigger for partial scale-out stage 1 (+2.2%)")
    partial_take_profit_ratio: float = Field(default=0.35, description="Fraction of position to sell on partial take profit (35%)")
    enable_multi_stage_tp: bool = Field(default=True, description="Enable secondary take profit tier (+4.5%)")
    stage2_take_profit_threshold_pct: float = Field(default=4.5, description="Profit % trigger for stage 2 take profit (+4.5%)")
    stage2_take_profit_ratio: float = Field(default=0.45, description="Fraction of remaining position to harvest at stage 2")
    enable_stage3_tp: bool = Field(default=True, description="Enable stage 3 take profit tier (+8.0%)")
    stage3_take_profit_threshold_pct: float = Field(default=8.0, description="Profit % trigger for stage 3 take profit (+8.0%)")
    stage3_take_profit_ratio: float = Field(default=0.50, description="Fraction of remaining position to harvest at stage 3 (50%)")

    # Multi-Tier Profit Vault Ratchet (Peak Profit Conservation)
    enable_profit_vault_ratchet: bool = Field(default=True, description="Enable multi-tier profit vault ratchet floor locks")
    vault_tier1_activation_pct: float = Field(default=1.5, description="Peak profit % to lock Tier 1 floor (+0.4%)")
    vault_tier1_floor_pct: float = Field(default=0.4, description="Guaranteed locked profit % at Tier 1 (+0.4%)")
    vault_tier2_activation_pct: float = Field(default=2.5, description="Peak profit % to lock Tier 2 floor (+1.2%)")
    vault_tier2_floor_pct: float = Field(default=1.2, description="Guaranteed locked profit % at Tier 2 (+1.2%)")
    vault_tier3_activation_pct: float = Field(default=4.0, description="Peak profit % to lock Tier 3 floor (+2.4%)")
    vault_tier3_floor_pct: float = Field(default=2.4, description="Guaranteed locked profit % at Tier 3 (+2.4%)")
    vault_tier4_activation_pct: float = Field(default=6.0, description="Peak profit % to lock Tier 4 floor (+4.2%)")
    vault_tier4_floor_pct: float = Field(default=4.2, description="Guaranteed locked profit % at Tier 4 (+4.2%)")
    vault_tier5_activation_pct: float = Field(default=8.5, description="Peak profit % to lock Tier 5 floor (+6.5%)")
    vault_tier5_floor_pct: float = Field(default=6.5, description="Guaranteed locked profit % at Tier 5 (+6.5%)")
    vault_tier6_activation_pct: float = Field(default=12.0, description="Peak profit % to lock Tier 6 floor (+9.5%)")
    vault_tier6_floor_pct: float = Field(default=9.5, description="Guaranteed locked profit % at Tier 6 (+9.5%)")

    # Dynamic Runner Target Extension (Trend Riding & Amplification)
    enable_runner_target_extension: bool = Field(default=True, description="Extend target for remaining runner on explosive momentum breakouts")
    runner_extension_harvest_ratio: float = Field(default=0.60, description="Harvest 60% at initial TP target and let 40% runner ride")
    runner_tp_multiplier: float = Field(default=1.15, description="Extend runner target TP by +15% above current level")
    runner_tight_stop_distance_pct: float = Field(default=1.2, description="Tight trailing stop distance % behind runner peak (1.2%)")

    # Free-Roll Winner Pyramiding (Compounding Winners Safely)
    enable_winner_pyramiding: bool = Field(default=True, description="Enable scaling into confirmed free-roll winners with profit locked")
    min_profit_pct_for_pyramiding: float = Field(default=2.8, description="Minimum unrealized profit % on existing position to permit pyramiding (+2.8%)")
    pyramiding_max_allocation_boost: float = Field(default=1.35, description="Multiplier on max position size allocation for free-roll winners (16.2%)")

    # High-Conviction Institutional Quality Gate
    min_signal_confidence: float = Field(default=0.0, description="Minimum multi-agent conviction required for entry (0.0=disabled by default, 0.68 in runner)")
    # Correlated Exposure Limit (Systemic Risk Control)
    max_concurrent_crypto_positions: int = Field(default=5, description="Max simultaneous crypto positions to prevent systemic portfolio risk")

    # Dynamic Portfolio Balancing & Concentration Trim (Capital Rotation)
    enable_dynamic_rebalancing: bool = Field(default=True, description="Enable capital rebalancing when single asset exceeds concentration cap")
    max_concentration_threshold_pct: float = Field(default=32.0, description="Concentration cap where position trim/harvest is recommended (32.0%)")
    target_concentration_pct: float = Field(default=25.0, description="Target concentration after rebalancing trim (25.0%)")

    # Capital Liquidity & Minimum Cash Reserve Floor
    min_cash_reserve_pct: float = Field(default=20.0, description="Minimum percentage of portfolio kept in unencumbered cash (e.g. 20%)")
    # Profit Quarantine Vault (Untouchable Retained Profit Lockbox - 40%)
    enable_profit_quarantine_vault: bool = Field(default=True, description="Enable institutional profit quarantine lockbox vault")
    profit_quarantine_ratio: float = Field(default=0.40, description="Ratio of cumulative net profit quarantined into untouchable cash vault (40%)")
    profit_quarantine_milestone_step: float = Field(default=200.0, description="Milestone profit step in USD to harvest and lock into untouchable vault (e.g. $200)")
    # Portfolio Aggregate Crypto Exposure Limit (Systemic Risk Control)
    max_total_crypto_allocation_pct: float = Field(default=30.0, description="Max total portfolio allocation across all crypto assets combined (30%)")
    # Portfolio-Level High-Water Mark Trailing Drawdown Protection (Peak Profit Lock)
    enable_portfolio_trailing_drawdown_guard: bool = Field(default=True, description="Enable portfolio peak high-water mark trailing drawdown guard")
    max_trailing_drawdown_from_peak_pct: float = Field(default=2.5, description="Max drawdown permitted from intraday peak equity (2.5%)")

    # --- 4 PALANCAS CUANTITATIVAS DE MÁXIMO RENDIMIENTO ---
    # Palanca A: Dynamic Hot Momentum Pool
    enable_dynamic_hot_scanner: bool = Field(default=True, description="Dynamically scan and rank Top N crypto/equity momentum movers")
    hot_scanner_top_n: int = Field(default=12, description="Number of top momentum assets to dynamically inject into universe")

    # Palanca B: Multi-Timeframe Confluence (MTF)
    enable_mtf_confluence: bool = Field(default=True, description="Multi-timeframe confluence filter (Macro trend + pullback trigger)")
    mtf_bonus_score: float = Field(default=20.0, description="Conviction bonus points when multi-timeframe trend aligns (+20)")
    mtf_penalty_score: float = Field(default=25.0, description="Conviction penalty when trading against macro trend (-25)")

    # Palanca C: Adaptive Volatility ATR Elastic Take-Profits & Trailing Stops
    enable_adaptive_atr_targets: bool = Field(default=True, description="Adaptive ATR-scaled take profit and trailing stop targets for explosive volatility")
    atr_tp_multiplier_stage1: float = Field(default=1.5, description="ATR multiplier for Stage 1 partial take profit floor (1.5x)")
    atr_tp_multiplier_stage2: float = Field(default=3.0, description="ATR multiplier for Stage 2 partial take profit floor (3.0x)")
    atr_tp_multiplier_stage3: float = Field(default=5.0, description="ATR multiplier for Stage 3 partial take profit floor (5.0x)")
    atr_tp_multiplier_runner: float = Field(default=8.0, description="ATR multiplier for dynamic runner extension (8.0x)")

    # Palanca D: Order Book Imbalance / L2 Depth Analysis
    enable_order_book_imbalance: bool = Field(default=True, description="Level 2 order book bid/ask depth imbalance analysis")
    min_order_book_imbalance_ratio: float = Field(default=1.25, description="Minimum bid/ask depth ratio for bullish institutional absorption confirmation (1.25x)")
    max_seller_overhang_ratio: float = Field(default=0.60, description="Maximum bid/ask ratio before triggering seller overhang size reduction (0.60x)")

    # Intelligent Time-Stop & Capital Rotation (Opportunity Cost Time-Decay)
    enable_time_stop_rotation: bool = Field(default=True, description="Enable time-based exit for stagnant crypto positions to rotate capital into hot momentum runners")
    max_crypto_stagnant_hours: float = Field(default=4.0, description="Max holding hours for crypto position in narrow stagnation range (4.0 hours)")
    stagnant_range_pct: float = Field(default=1.2, description="Max absolute profit/loss % boundary to consider a position stagnant (+/- 1.2%)")


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
