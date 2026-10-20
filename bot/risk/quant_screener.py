"""Deterministic Quantitative Screener and Pre-Filtering Engine.

Implements high-frequency multi-factor technical screening to evaluate market
conditions in under 50ms with ZERO token consumption. Dispatches signals to
the LLM multi-agent committee or executes directly when high-probability
2.0:1 risk-reward setups emerge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from stockstats import wrap

from tradingagents.dataflows.vendors.yahoo.snapshot import _verified_rows

logger = logging.getLogger(__name__)


@dataclass
class QuantScreenResult:
    """Outcome of algorithmic quantitative screening for an instrument."""
    symbol: str
    is_actionable: bool
    bias: str  # "BUY", "SELL", "NEUTRAL"
    score: float  # 0 to 100 conviction score
    setup_name: str
    current_price: float
    suggested_stop_loss: float
    suggested_take_profit: float
    risk_reward_ratio: float
    metrics: dict[str, float]
    rationale: str


class QuantScreener:
    """High-speed algorithmic screener evaluating Trend, Momentum, Volatility, and Mean Reversion."""

    def __init__(
        self,
        rsi_oversold: float = 42.0,
        rsi_overbought: float = 62.0,
        atr_multiplier_sl: float = 1.2,
        atr_multiplier_tp: float = 2.4,
        conviction_threshold: float = 55.0,
        enable_mtf_confluence: bool = True,
        mtf_bonus_score: float = 20.0,
        mtf_penalty_score: float = 25.0,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.conviction_threshold = conviction_threshold
        self.enable_mtf_confluence = enable_mtf_confluence
        self.mtf_bonus_score = mtf_bonus_score
        self.mtf_penalty_score = mtf_penalty_score
        self._btc_macro_cache: tuple[float, dict[str, Any]] | None = None

    def get_btc_macro_health(self, as_of_date: str) -> dict[str, Any]:
        """Check Bitcoin macro trend to act as market regime gatekeeper for altcoins (<20ms)."""
        import time
        now = time.time()
        if self._btc_macro_cache and (now - self._btc_macro_cache[0]) < 120.0:
            return self._btc_macro_cache[1]

        try:
            df = _verified_rows("BTC-USD", as_of_date)
            stock_df = wrap(df.copy())
            stock_df["rsi"]
            stock_df["close_50_sma"]
            stock_df["macdh"]
            close = float(stock_df["close"].iloc[-1])
            sma50 = float(stock_df["close_50_sma"].iloc[-1])
            rsi = float(stock_df["rsi"].iloc[-1])
            macdh = float(stock_df["macdh"].iloc[-1])

            # Severe bear regime: price significantly below SMA50 with negative momentum
            is_bearish = (close < sma50 * 0.97) and (macdh < 0 or rsi < 40)
            res = {
                "btc_price": close,
                "sma50": sma50,
                "rsi": rsi,
                "macdh": macdh,
                "is_bearish": is_bearish,
                "regime": "BEARISH_CORRECTION" if is_bearish else "BULLISH_OR_STABLE",
            }
            self._btc_macro_cache = (now, res)
            return res
        except Exception as e:
            logger.debug("Notice checking BTC macro health: %s", e)
            return {"is_bearish": False, "regime": "UNKNOWN"}

    @staticmethod
    def _detect_divergence(closes: list[float], rsis: list[float], lookback: int = 14) -> tuple[str | None, str]:
        """Detect regular Bullish or Bearish divergence over the lookback window with statistical edge."""
        if len(closes) < lookback + 2 or len(rsis) < lookback + 2:
            return None, ""
        curr_c = closes[-1]
        curr_rsi = rsis[-1]

        # Bullish Divergence: Lower Low in price with Higher Low in RSI (momentum exhaust)
        past_c_window = closes[-lookback:-2]
        past_rsi_window = rsis[-lookback:-2]
        min_c_idx = int(np.argmin(past_c_window))
        past_min_c = past_c_window[min_c_idx]
        past_min_rsi = past_rsi_window[min_c_idx]

        if curr_c <= past_min_c * 1.008 and curr_rsi > (past_min_rsi + 3.0) and curr_rsi < 55.0:
            return "BULLISH", f"Divergencia Alcista RSI ({curr_rsi:.1f} vs {past_min_rsi:.1f} anterior con precio en nuevo mínimo)"

        # Bearish Divergence: Higher High in price with Lower High in RSI
        max_c_idx = int(np.argmax(past_c_window))
        past_max_c = past_c_window[max_c_idx]
        past_max_rsi = past_rsi_window[max_c_idx]

        if curr_c >= past_max_c * 0.992 and curr_rsi < (past_max_rsi - 3.0) and curr_rsi > 45.0:
            return "BEARISH", f"Divergencia Bajista RSI ({curr_rsi:.1f} vs {past_max_rsi:.1f} anterior con precio en nuevo máximo)"

        return None, ""

    def screen(self, symbol: str, as_of_date: str) -> QuantScreenResult:
        """Evaluate instrument technicals deterministically in milliseconds."""
        try:
            df = _verified_rows(symbol, as_of_date)
            stock_df = wrap(df.copy())
            
            # Trigger calculation of core technical indicators
            stock_df["rsi"]
            stock_df["macdh"]
            stock_df["boll_lb"]
            stock_df["boll_ub"]
            stock_df["boll"]
            stock_df["atr"]
            stock_df["close_10_ema"]
            stock_df["close_50_sma"]
            stock_df["close_200_sma"]

            close = float(stock_df["close"].iloc[-1])
            prev_close = float(stock_df["close"].iloc[-2]) if len(stock_df) > 1 else close
            rsi = float(stock_df["rsi"].iloc[-1])
            prev_rsi = float(stock_df["rsi"].iloc[-2]) if len(stock_df) > 1 else rsi
            macdh = float(stock_df["macdh"].iloc[-1])
            prev_macdh = float(stock_df["macdh"].iloc[-2]) if len(stock_df) > 1 else macdh
            boll_lb = float(stock_df["boll_lb"].iloc[-1])
            boll_ub = float(stock_df["boll_ub"].iloc[-1])
            boll_mid = float(stock_df["boll"].iloc[-1])
            atr = float(stock_df["atr"].iloc[-1])
            ema10 = float(stock_df["close_10_ema"].iloc[-1])
            sma50 = float(stock_df["close_50_sma"].iloc[-1])
            sma200 = float(stock_df["close_200_sma"].iloc[-1]) if "close_200_sma" in stock_df else sma50

            # RVOL (Relative Volume vs 20-period moving average)
            rvol = 1.0
            if "volume" in stock_df and len(stock_df["volume"]) >= 20:
                try:
                    vol_series = stock_df["volume"]
                    curr_vol = float(vol_series.iloc[-1])
                    avg_vol = float(vol_series.iloc[-20:].mean())
                    if avg_vol > 0:
                        rvol = round(curr_vol / avg_vol, 2)
                except Exception:
                    pass

            # Detect Statistical Divergences (RSI vs Price)
            div_type, div_desc = self._detect_divergence(
                stock_df["close"].tolist(),
                stock_df["rsi"].tolist()
            )

            # Channel percentile: 0.0 = lower band, 1.0 = upper band
            channel_width = boll_ub - boll_lb
            band_pct = (close - boll_lb) / channel_width if channel_width > 0 else 0.5
            atr_pct = round((atr / close) * 100.0, 2) if close > 0 else 0.0

            metrics = {
                "close": close,
                "rsi": rsi,
                "macdh": macdh,
                "boll_lb": boll_lb,
                "boll_ub": boll_ub,
                "boll_mid": boll_mid,
                "atr": atr,
                "atr_pct": atr_pct,
                "band_pct": round(band_pct, 4),
                "ema10": ema10,
                "sma50": sma50,
                "sma200": sma200,
                "rvol": rvol,
                "divergence": div_type or "NONE",
                "mtf_confluence": "NEUTRAL",
            }

            bull_points = 0
            bear_points = 0
            bull_reasons: list[str] = []
            bear_reasons: list[str] = []

            # 1. Momentum & Mean Reversion Factor
            if rsi <= self.rsi_oversold:
                bull_points += 30
                bull_reasons.append(f"RSI en sobreventa ({rsi:.1f})")
                if rsi > prev_rsi:
                    bull_points += 15
                    bull_reasons.append("RSI curvando al alza")
            elif rsi >= self.rsi_overbought:
                bear_points += 30
                bear_reasons.append(f"RSI en sobrecompra ({rsi:.1f})")
                if rsi < prev_rsi:
                    bear_points += 15
                    bear_reasons.append("RSI perdiendo fuerza")

            # 2. Bollinger Band Channel Position Factor
            if band_pct <= 0.25:
                bull_points += 25
                bull_reasons.append(f"Precio en cuartil inferior Bollinger ({band_pct*100:.1f}%)")
            elif band_pct >= 0.75:
                bear_points += 25
                bear_reasons.append(f"Precio en cuartil superior Bollinger ({band_pct*100:.1f}%)")

            # 3. MACD Momentum Acceleration
            if macdh > 0:
                bull_points += 10
                if macdh > prev_macdh:
                    bull_points += 10
                    bull_reasons.append("MACD acelerando positivo")
            else:
                bear_points += 10
                if macdh < prev_macdh:
                    bear_points += 10
                    bear_reasons.append("MACD acelerando negativo")

            # 4. Moving Average Trend Alignment (Intermediate SMA-50 & Macro SMA-200)
            if close > sma50 and close > sma200:
                bull_points += 20
                bull_reasons.append("Precio en tendencia alcista macro (sobre SMA-50 y SMA-200)")
            elif close > sma50:
                bull_points += 15
                bull_reasons.append("Precio sobre SMA-50 (soporte alcista)")
            elif close < sma50 and rsi <= self.rsi_oversold:
                # Oversold dip discount relative to intermediate average
                bull_points += 10
                bull_reasons.append("Descuento significativo bajo SMA-50")
            elif close < sma50:
                bear_points += 15
                bear_reasons.append("Precio bajo SMA-50 (resistencia bajista)")

            # Palanca B: Multi-Timeframe Confluence (MTF) Alignment
            if self.enable_mtf_confluence:
                macro_bullish = (close > sma50 and (ema10 >= sma50 or macdh > 0))
                macro_bearish = (close < sma50 and (ema10 <= sma50 or macdh < 0))
                is_pullback_dip = (rsi <= 48.0 or band_pct <= 0.35 or div_type == "BULLISH")
                is_bear_rally = (rsi >= 58.0 or band_pct >= 0.70 or div_type == "BEARISH")

                if bull_points > bear_points:
                    if macro_bullish and is_pullback_dip:
                        bull_points += int(self.mtf_bonus_score)
                        bull_reasons.append(f"Confluencia MTF Alcista: Pullback a soporte en macro alcista (+{int(self.mtf_bonus_score)} pts)")
                        metrics["mtf_confluence"] = "BULLISH_CONFLUENCE"
                    elif close < sma50 and close < sma200 and macdh < 0 and div_type != "BULLISH":
                        bull_points = max(0, bull_points - int(self.mtf_penalty_score))
                        bull_reasons.append(f"Aviso MTF: Intento contra tendencia macro bajista (-{int(self.mtf_penalty_score)} pts)")
                        metrics["mtf_confluence"] = "BEARISH_DIVERGENT"
                elif bear_points > bull_points:
                    if macro_bearish and is_bear_rally:
                        bear_points += int(self.mtf_bonus_score)
                        bear_reasons.append(f"Confluencia MTF Bajista: Rebote a resistencia en macro bajista (+{int(self.mtf_bonus_score)} pts)")
                        metrics["mtf_confluence"] = "BEARISH_CONFLUENCE"
                    elif close > sma50 and close > sma200 and macdh > 0 and div_type != "BEARISH":
                        bear_points = max(0, bear_points - int(self.mtf_penalty_score))
                        bear_reasons.append(f"Aviso MTF: Venta contra tendencia macro alcista fuerte (-{int(self.mtf_penalty_score)} pts)")
                        metrics["mtf_confluence"] = "BULLISH_DIVERGENT"

            # Palanca B: Macro SMA-200 Filter (Paul Tudor Jones Rule)
            if close < sma200 and div_type != "BULLISH":
                if bull_points > bear_points:
                    bull_points = max(0, bull_points - 25)
                    bull_reasons.append(f"Aviso Macro: Precio bajo SMA-200 (${sma200:.2f}) sin divergencia alcista")

            # Palanca C: Techo Anti-Trampas (Anti-Bull Trap Ceiling: RSI >= 64.0)
            if rsi >= 64.0:
                if bull_points > 0:
                    bull_points = 0
                    bear_points += 20
                    bear_reasons.append(f"Techo Anti-Trampas: RSI ({rsi:.1f} >= 64.0) en sobrecompra; compras anuladas")

            # 5. Statistical Divergence Booster (Win Rate Multiplier)
            if div_type == "BULLISH":
                bull_points += 35
                bull_reasons.append(div_desc)
            elif div_type == "BEARISH":
                bear_points += 35
                bear_reasons.append(div_desc)

            # 6. Relative Volume (RVOL) Institutional Confirmation
            if rvol >= 1.25:
                if bull_points > bear_points:
                    bull_points += 15
                    bull_reasons.append(f"Volumen institucional confirmativo (RVOL: {rvol:.2f}x)")
                elif bear_points > bull_points:
                    bear_points += 15
                    bear_reasons.append(f"Volumen vendedor elevado (RVOL: {rvol:.2f}x)")
            elif rvol < 0.60:
                # Penalize low volume breakouts/breakdowns
                if bull_points > bear_points:
                    bull_points = max(0, bull_points - 10)
                    bull_reasons.append(f"Aviso: Volumen reducido (RVOL: {rvol:.2f}x)")
                elif bear_points > bull_points:
                    bear_points = max(0, bear_points - 10)
                    bear_reasons.append(f"Aviso: Volumen reducido (RVOL: {rvol:.2f}x)")

            # Determine setup and actionable conviction
            is_actionable = False
            bias = "NEUTRAL"
            score = max(bull_points, bear_points)
            setup_name = "Consolidación Neutral"
            rationale = "Mercado sin desequilibrio de oferta/demanda claro; capital preservado."

            # 7. Macro BTC Regime Gatekeeper for Altcoins
            is_altcoin = symbol not in ("BTC-USD", "BTCUSD", "BTC/USD") and ("-USD" in symbol or "USD" in symbol)
            if is_altcoin and bull_points > bear_points:
                btc_macro = self.get_btc_macro_health(as_of_date)
                if btc_macro.get("is_bearish"):
                    logger.info(
                        "[MACRO_GATEKEEPER] 🛡️ BTC en régimen correctivo (%s, Precio: $%.2f < SMA50: $%.2f). Veto a compras en %s.",
                        btc_macro.get("regime"), btc_macro.get("btc_price", 0), btc_macro.get("sma50", 0), symbol
                    )
                    return QuantScreenResult(
                        symbol=symbol,
                        is_actionable=False,
                        bias="NEUTRAL",
                        score=score,
                        setup_name="Veto Macro Gatekeeper (BTC)",
                        current_price=close,
                        suggested_stop_loss=round(close * 0.96, 2),
                        suggested_take_profit=round(close * 1.06, 2),
                        risk_reward_ratio=2.0,
                        metrics=metrics,
                        rationale=f"Altcoin pausada: BTC bajo SMA-50 (${btc_macro.get('btc_price', 0):.2f}) en corrección.",
                    )

            from bot.risk.regime_specialist import REGIME_SPECIALIST
            regime = REGIME_SPECIALIST.analyze_market_regime("BTC-USD", as_of_date)
            sl_mult = regime.sl_multiplier
            tp_mult = regime.tp_multiplier

            if bull_points >= self.conviction_threshold and bull_points > bear_points:
                is_actionable = True
                bias = "BUY"
                if div_type == "BULLISH":
                    setup_name = f"Divergencia Alcista (Alta Probabilidad - {regime.regime_display})"
                else:
                    setup_name = f"Oportunidad Alcista ({regime.regime_display})"
                rationale = " + ".join(bull_reasons)
                stop_loss = round(close - (atr * sl_mult), 2)
                take_profit = round(close + (atr * tp_mult), 2)
            elif bear_points >= self.conviction_threshold and bear_points > bull_points:
                is_actionable = True
                bias = "SELL"
                if div_type == "BEARISH":
                    setup_name = f"Divergencia Bajista (Reversión - {regime.regime_display})"
                else:
                    setup_name = f"Reversión Bajista ({regime.regime_display})"
                rationale = " + ".join(bear_reasons)
                stop_loss = round(close + (atr * sl_mult), 2)
                take_profit = round(close - (atr * tp_mult), 2)
            else:
                stop_loss = round(close * 0.96, 2)
                take_profit = round(close * 1.06, 2)

            rr_ratio = round(tp_mult / sl_mult, 2)

            return QuantScreenResult(
                symbol=symbol,
                is_actionable=is_actionable,
                bias=bias,
                score=score,
                setup_name=setup_name,
                current_price=close,
                suggested_stop_loss=stop_loss,
                suggested_take_profit=take_profit,
                risk_reward_ratio=rr_ratio,
                metrics=metrics,
                rationale=rationale,
            )

        except Exception as e:
            logger.warning("[QUANT_SCREENER] Notice evaluating %s: %s; falling back to neutral", symbol, e)
            return QuantScreenResult(
                symbol=symbol,
                is_actionable=False,
                bias="NEUTRAL",
                score=0.0,
                setup_name="Error / Fallback",
                current_price=0.0,
                suggested_stop_loss=0.0,
                suggested_take_profit=0.0,
                risk_reward_ratio=2.0,
                metrics={},
                rationale=str(e),
            )
