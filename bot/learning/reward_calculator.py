"""Quantitative Continuous Reward Calculator for Neural Feedback and Reinforcement.

Inspired by Marcos Lopez de Prado's 'Advances in Financial Machine Learning'
and Deep RL in quantitative trading:
Computes a dense, continuous scalar reward R in [-1.0, +1.0] for closed trades,
evaluating:
1. Volatility-normalized PnL: tanh(pnl_pct / max(atr_pct, 0.5))
2. Excursion quality (MAE vs Stop Loss risk distance): penalizes high drawdown during hold
3. Trade efficiency ratio (PnL vs MFE): measures how much of the peak profit was captured
4. Opportunity cost / stagnation penalty: penalizes capital tied up with zero or negative return
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class RewardBreakdown:
    pnl_component: float
    mae_component: float
    efficiency_component: float
    opportunity_penalty: float
    total_reward: float


class RewardCalculator:
    """Calculates continuous reward signal for closed trades."""

    def __init__(
        self,
        weight_pnl: float = 0.55,
        weight_mae: float = 0.20,
        weight_efficiency: float = 0.20,
        weight_opp_cost: float = 0.05,
    ):
        self.w_pnl = weight_pnl
        self.w_mae = weight_mae
        self.w_eff = weight_efficiency
        self.w_opp = weight_opp_cost

    def calculate(
        self,
        pnl_pct: float,
        volatility_atr_pct: float = 2.0,
        mae_pct: float = 0.0,
        mfe_pct: float = 0.0,
        stop_loss_pct: float = 2.5,
        duration_hours: float = 0.0,
    ) -> float:
        """Compute total continuous reward scalar in [-1.0, 1.0]."""
        breakdown = self.calculate_breakdown(
            pnl_pct=pnl_pct,
            volatility_atr_pct=volatility_atr_pct,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
            stop_loss_pct=stop_loss_pct,
            duration_hours=duration_hours,
        )
        return breakdown.total_reward

    def calculate_breakdown(
        self,
        pnl_pct: float,
        volatility_atr_pct: float = 2.0,
        mae_pct: float = 0.0,
        mfe_pct: float = 0.0,
        stop_loss_pct: float = 2.5,
        duration_hours: float = 0.0,
    ) -> RewardBreakdown:
        """Compute detailed components and final reward."""
        # 1. Volatility-normalized PnL via tanh
        safe_atr = max(volatility_atr_pct, 0.5)
        norm_return = pnl_pct / safe_atr
        pnl_comp = math.tanh(norm_return)  # strictly in (-1.0, 1.0)

        # 2. MAE component (Drawdown containment)
        # If MAE is 0, mae_comp = +1.0 (clean entry, zero adverse excursion)
        # If MAE equals or exceeds stop_loss_pct, mae_comp = -1.0
        safe_sl = max(abs(stop_loss_pct), 0.5)
        mae_ratio = min(max(mae_pct / safe_sl, 0.0), 2.0)
        mae_comp = 1.0 - (mae_ratio * 2.0)  # maps [0, 1] -> [1, -1]
        mae_comp = max(-1.0, min(1.0, mae_comp))

        # 3. Efficiency component (Exit timing quality)
        if pnl_pct > 0 and mfe_pct > 0:
            eff = min(pnl_pct / max(mfe_pct, pnl_pct), 1.0)
            eff_comp = (eff * 1.5) - 0.5
        elif pnl_pct <= 0:
            if mae_pct > 0:
                unrealized_worse = mae_pct - abs(pnl_pct)
                eff_comp = -1.0 + min(max(unrealized_worse / safe_sl, 0.0), 0.8)
            else:
                eff_comp = -0.5
        else:
            eff_comp = 0.0
        eff_comp = max(-1.0, min(1.0, eff_comp))

        # 4. Opportunity cost penalty (stagnant capital)
        opp_penalty = 0.0
        if abs(pnl_pct) < 0.3 and duration_hours > 3.0:
            opp_penalty = -min(0.5, (duration_hours - 3.0) / 24.0)

        # Weighted combination
        raw_total = (
            (self.w_pnl * pnl_comp)
            + (self.w_mae * mae_comp)
            + (self.w_eff * eff_comp)
            + (self.w_opp * opp_penalty)
        )
        total_reward = round(max(-1.0, min(1.0, raw_total)), 4)

        return RewardBreakdown(
            pnl_component=round(pnl_comp, 4),
            mae_component=round(mae_comp, 4),
            efficiency_component=round(eff_comp, 4),
            opportunity_penalty=round(opp_penalty, 4),
            total_reward=total_reward,
        )


REWARD_CALCULATOR = RewardCalculator()
