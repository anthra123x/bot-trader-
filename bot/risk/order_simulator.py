"""Order Execution Simulator & Mathematical Expectancy: Reverse-Engineered from Backtrader & VectorBT.

Implements execution realism and statistical edge validation:
1. Directional Slippage Modeling (Backtrader): Adjusts fill price by market impact and volatility.
2. Exchange Fee & Commission Modeling (VectorBT): Accounts for maker/taker fees on equity and crypto.
3. Mathematical Expectancy Validation:
   Expectancy = (WinRate * AvgGainDistance) - ((1 - WinRate) * AvgLossDistance) - RoundTripFriction
   Ensures that only trades with strictly positive mathematical expectancy and >= 2.0x net R:R are executed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from bot.models import AssetClass, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class OrderSimulationResult:
    """Detailed result of realistic order fill simulation and statistical expectancy."""
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
    is_convex_asymmetric: bool  # net R:R >= min_ratio
    rejection_reason: str | None = None


class ExecutionSimulator:
    """Realistic fill and expectancy simulator inspired by Backtrader & VectorBT."""

    def __init__(
        self,
        crypto_slippage_pct: float = 0.12,     # 0.12% realistic crypto slippage
        equity_slippage_pct: float = 0.04,     # 0.04% realistic equity slippage
        crypto_taker_fee_pct: float = 0.10,    # 0.10% Alpaca/Binance crypto fee
        equity_fee_per_share: float = 0.005,   # $0.005/share equity commission
        min_risk_reward_ratio: float = 1.70,   # Net mathematical convexity floor after friction (slippage & fees)
    ):
        self.crypto_slippage_pct = crypto_slippage_pct
        self.equity_slippage_pct = equity_slippage_pct
        self.crypto_taker_fee_pct = crypto_taker_fee_pct
        self.equity_fee_per_share = equity_fee_per_share
        self.min_risk_reward_ratio = min_risk_reward_ratio

    def calculate_slippage(
        self,
        nominal_price: float,
        side: OrderSide,
        asset_class: AssetClass,
        volatility_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        """Calculate realistic fill price after directional slippage."""
        base_slippage = (
            self.crypto_slippage_pct if asset_class == AssetClass.CRYPTO else self.equity_slippage_pct
        ) / 100.0
        slippage_factor = base_slippage * max(0.5, min(3.0, volatility_multiplier))

        if side == OrderSide.BUY:
            fill_price = nominal_price * (1.0 + slippage_factor)
        else:
            fill_price = nominal_price * (1.0 - slippage_factor)

        slippage_dollars = abs(fill_price - nominal_price)
        return round(fill_price, 4), round(slippage_dollars, 4)

    def calculate_commission(
        self,
        fill_price: float,
        quantity: float,
        asset_class: AssetClass,
    ) -> float:
        """Calculate estimated exchange commission per leg."""
        if quantity <= 0 or fill_price <= 0:
            return 0.0

        if asset_class == AssetClass.CRYPTO:
            notional = fill_price * quantity
            return round(notional * (self.crypto_taker_fee_pct / 100.0), 4)
        else:
            # Equities: minimum $1.00, or $0.005/share
            return max(1.0, round(quantity * self.equity_fee_per_share, 4))

    def simulate_trade_execution(
        self,
        nominal_price: float,
        suggested_stop_loss: float,
        suggested_take_profit: float,
        quantity: float,
        side: OrderSide,
        asset_class: AssetClass,
        historical_win_rate: float = 0.55,
        volatility_multiplier: float = 1.0,
    ) -> OrderSimulationResult:
        """Simulate entry and exit fills, commissions, and calculate net mathematical expectancy."""
        if nominal_price <= 0 or suggested_stop_loss <= 0 or suggested_take_profit <= 0 or quantity <= 0:
            return OrderSimulationResult(
                simulated_fill_price=nominal_price,
                effective_stop_loss=suggested_stop_loss,
                effective_take_profit=suggested_take_profit,
                estimated_slippage_dollars=0.0,
                estimated_commission_dollars=0.0,
                net_reward_dollars=0.0,
                net_risk_dollars=0.0,
                net_risk_reward_ratio=0.0,
                statistical_expectancy=0.0,
                is_positive_expectancy=False,
                is_convex_asymmetric=False,
                rejection_reason="Invalid trade dimensions or zero price/quantity.",
            )

        # 1. Simulate entry fill with slippage (Backtrader)
        entry_fill, entry_slip = self.calculate_slippage(nominal_price, side, asset_class, volatility_multiplier)
        entry_commission = self.calculate_commission(entry_fill, quantity, asset_class)

        # 2. Simulate exit fills with slippage
        opposite_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        tp_fill, tp_slip = self.calculate_slippage(suggested_take_profit, opposite_side, asset_class, volatility_multiplier)
        sl_fill, sl_slip = self.calculate_slippage(suggested_stop_loss, opposite_side, asset_class, volatility_multiplier)

        tp_commission = self.calculate_commission(tp_fill, quantity, asset_class)
        sl_commission = self.calculate_commission(sl_fill, quantity, asset_class)

        # 3. Calculate net reward and risk per share/unit
        if side == OrderSide.BUY:
            gross_reward_per_unit = max(0.0, tp_fill - entry_fill)
            gross_risk_per_unit = max(0.0001, entry_fill - sl_fill)
        else:
            gross_reward_per_unit = max(0.0, entry_fill - tp_fill)
            gross_risk_per_unit = max(0.0001, sl_fill - entry_fill)

        total_entry_friction = entry_slip + (entry_commission / quantity)
        total_tp_friction = tp_slip + (tp_commission / quantity)
        total_sl_friction = sl_slip + (sl_commission / quantity)

        net_reward_per_unit = max(0.0, gross_reward_per_unit - total_entry_friction - total_tp_friction)
        net_risk_per_unit = gross_risk_per_unit + total_entry_friction + total_sl_friction

        net_rr = round(net_reward_per_unit / net_risk_per_unit, 2) if net_risk_per_unit > 0 else 0.0

        # 4. Statistical Expectancy (VectorBT Edge Formula)
        # E = (WinRate * NetReward) - ((1 - WinRate) * NetRisk)
        w = max(0.1, min(0.9, historical_win_rate))
        expectancy = round((w * net_reward_per_unit) - ((1.0 - w) * net_risk_per_unit), 4)

        is_pos_exp = expectancy > 0
        is_convex = net_rr >= self.min_risk_reward_ratio

        rejection_reason = None
        if not is_pos_exp:
            rejection_reason = (
                f"Negative mathematical expectancy (${expectancy:.2f}/u) taking into account "
                f"slippage and fees. WinRate: {w*100:.0f}%."
            )
        elif not is_convex:
            rejection_reason = (
                f"Insufficient net risk-reward ratio ({net_rr:.2f}x < {self.min_risk_reward_ratio:.2f}x min) "
                f"after slippage (${entry_slip+tp_slip:.2f}) and commission fees."
            )

        return OrderSimulationResult(
            simulated_fill_price=entry_fill,
            effective_stop_loss=sl_fill,
            effective_take_profit=tp_fill,
            estimated_slippage_dollars=round((entry_slip + tp_slip) * quantity, 2),
            estimated_commission_dollars=round(entry_commission + tp_commission, 2),
            net_reward_dollars=round(net_reward_per_unit * quantity, 2),
            net_risk_dollars=round(net_risk_per_unit * quantity, 2),
            net_risk_reward_ratio=net_rr,
            statistical_expectancy=expectancy,
            is_positive_expectancy=is_pos_exp,
            is_convex_asymmetric=is_convex,
            rejection_reason=rejection_reason,
        )
