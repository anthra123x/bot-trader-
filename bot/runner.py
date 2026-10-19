"""Main trading bot orchestrator connecting TradingAgents, RiskEngine, and Brokers."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.graph.trading_graph import TradingAgentsGraph

from bot.brokers import get_broker
from bot.brokers.base import BaseBroker
from bot.config import BOT_CONFIG, BotConfig
from bot.database.neon_repo import NeonRepository, get_neon_repo
from bot.models import (
    AssetClass,
    EnhancedPortfolioContext,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskValidationResult,
    TradeSignal,
)
from bot.notifications.notifier import Notifier
from bot.risk.quant_screener import QuantScreener, QuantScreenResult
from bot.risk.risk_engine import RiskEngine

logger = logging.getLogger(__name__)


class BotRunner:
    """Coordinates multi-agent market analysis, risk evaluation, and order execution."""

    def __init__(
        self,
        config: BotConfig | None = None,
        broker: BaseBroker | None = None,
        db: NeonRepository | None = None,
    ):
        self.config = config or BOT_CONFIG
        self.broker = broker or get_broker(self.config.broker_type)
        self.db = db or get_neon_repo()
        self.risk_engine = RiskEngine(self.config.risk)
        self.quant_screener = QuantScreener()
        self.notifier = Notifier(
            telegram_token=self.config.telegram_bot_token,
            telegram_chat_id=self.config.telegram_chat_id,
            discord_webhook=self.config.discord_webhook_url,
        )

        # Specialized graphs: crypto runs technicals + social sentiment (Reddit/StockTwits); equities add fundamentals
        debug_mode = os.getenv("TRADINGAGENTS_DEBUG", "false").lower() in ("true", "1")
        self.crypto_graph = TradingAgentsGraph(
            selected_analysts=["market", "social"],
            debug=debug_mode,
            config=self.config.tradingagents_config,
        )
        self.equity_graph = TradingAgentsGraph(
            selected_analysts=["market", "fundamentals", "social"],
            debug=debug_mode,
            config=self.config.tradingagents_config,
        )
        self.graph = self.equity_graph

    def identify_asset_class(self, symbol: str) -> AssetClass:
        """Categorize instrument into Equity or Crypto."""
        normalized = symbol.upper().strip()
        if (
            "-USD" in normalized
            or normalized in [s.upper() for s in self.config.watchlist_crypto]
            or normalized in ("BTC", "ETH", "SOL", "USDT")
        ):
            return AssetClass.CRYPTO
        return AssetClass.EQUITY

    def _map_rating_to_side(self, rating: str) -> OrderSide:
        """Map 5-tier rating into actionable order side."""
        norm = (rating or "").strip().lower()
        if norm in ("buy", "overweight"):
            return OrderSide.BUY
        elif norm in ("sell", "underweight"):
            return OrderSide.SELL
        return OrderSide.HOLD

    def process_signal_and_execute(
        self,
        symbol: str,
        rating: str,
        final_state: dict,
        current_price: float,
        trade_date: str,
        quant_result: QuantScreenResult | None = None,
    ) -> Order | None:
        """Validate signal through risk engine and dispatch order to broker."""
        asset_class = self.identify_asset_class(symbol)
        action = self._map_rating_to_side(rating)

        # Volatility-calibrated ATR stop-loss and target price from Quant Screener
        if quant_result and quant_result.is_actionable:
            stop_loss = quant_result.suggested_stop_loss
            target_price = quant_result.suggested_take_profit
        else:
            stop_loss = None
            target_price = None

        # Fallback heuristic parsing if structured values are embedded in text
        balance = self.broker.get_account_balance()
        positions = self.broker.get_positions()

        signal = TradeSignal(
            symbol=symbol,
            asset_class=asset_class,
            trade_date=trade_date,
            rating=rating,
            action=action,
            current_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            confidence=round(quant_result.score / 100.0, 2) if quant_result else (0.75 if rating in ("Buy", "Sell", "BUY", "SELL") else 0.5),
            reasoning=str(final_state.get("final_trade_decision") or "")[:300],
            metrics=quant_result.metrics if quant_result else {},
        )

        logger.info(
            "[CONSENSUS] ⚖️ Consenso para %s: Calificación='%s', Acción='%s' (Precio: $%.2f)",
            symbol, rating, action.value if action else "HOLD", current_price
        )

        # Quality Gate Check: High-Conviction Institutional Filter
        min_conf = getattr(self.config.risk, "min_signal_confidence", 0.0) or 0.68
        if signal.action == OrderSide.BUY and signal.confidence < min_conf:
            logger.info(
                "[QUALITY_GATE] 🛡️ %s: Convicción (%.0f%%) inferior al umbral mínimo institucional (%.0f%%). Entrada pausada para maximizar Win Rate (0 capital arriesgado).",
                symbol, signal.confidence * 100.0, min_conf * 100.0
            )
            return None

        # Risk check
        risk_result: RiskValidationResult = self.risk_engine.evaluate(signal, balance, positions)
        self.notifier.notify_signal_and_risk(signal, risk_result)

        if not risk_result.is_approved or risk_result.adjusted_quantity <= 0:
            reasons = ", ".join(risk_result.rejection_reasons) if risk_result.rejection_reasons else "Decisión Neutral / Hold"
            logger.info("[DECISION] 🛡️ %s: Sin orden ejecutada (%s) - Capital resguardado", symbol, reasons)
            return None

        # Create Order
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            asset_class=asset_class,
            side=action,
            order_type=OrderType.MARKET,
            quantity=risk_result.adjusted_quantity,
            stop_loss_price=risk_result.stop_loss_price,
            take_profit_price=risk_result.take_profit_price,
            status=OrderStatus.PENDING,
        )

        executed_order = self.broker.submit_order(order)
        logger.info("[ORDER_FILLED] 🚀 ¡Orden ejecutada en Alpaca para %s! %s %s u @ $%.2f", symbol, action.value, risk_result.adjusted_quantity, current_price)
        self.notifier.notify_order_execution(executed_order)
        try:
            self.db.save_order(executed_order)
        except Exception as e:
            logger.warning("[NEON] Could not save order to Neon: %s", e)
        return executed_order

    def run_symbol(self, symbol: str, trade_date: str | None = None) -> Order | None:
        """Run full cycle: Algorithmic Quant Screener -> Multi-Agent Validation -> Risk -> Execution."""
        date = trade_date or get_current_date()

        is_crypto = self.identify_asset_class(symbol) == AssetClass.CRYPTO
        if not is_crypto and not self.is_equity_market_open():
            logger.info(
                "[MARKET_CLOSED] 🌙 Mercado bursátil (NYSE/NASDAQ) cerrado para %s. Órdenes spot requieren sesión regular (09:30-16:00 ET). Operativa enfocada en Cripto 24/7.",
                symbol
            )
            return None

        # Step 1: Deterministic Algorithmic Quant Screening (< 40ms, 0 tokens)
        screen_res = self.quant_screener.screen(symbol, date)
        if not screen_res.is_actionable:
            logger.info(
                "[QUANT_SCREENER] ⏸️ %s en %s (Puntaje: %.0f/100, RSI: %.1f). Mercado neutral; capital resguardado (0 tokens gastados).",
                symbol, screen_res.setup_name, screen_res.score, screen_res.metrics.get("rsi", 50.0)
            )
            return None

        # Step 2: High-Probability Opportunity detected -> Dispatch to Multi-Agent Committee
        logger.info(
            "[QUANT_SCREENER] 🎯 ¡Oportunidad en %s! Setup: %s (Bias: %s, Score: %.0f, SL: $%.2f, TP: $%.2f, R:R: %.2f). Despertando agentes...",
            symbol, screen_res.setup_name, screen_res.bias, screen_res.score, screen_res.suggested_stop_loss, screen_res.suggested_take_profit, screen_res.risk_reward_ratio
        )

        try:
            current_price = self.broker.get_current_price(symbol)
        except Exception:
            current_price = screen_res.current_price

        is_crypto = self.identify_asset_class(symbol) == AssetClass.CRYPTO
        asset_type = "crypto" if is_crypto else "stock"
        # Build live EnhancedPortfolioContext so subagents reason with real account positions, cash, quant metrics & lessons
        portfolio_ctx = None
        try:
            from tradingagents.portfolio import Position as PortfolioPosition
            bal = self.broker.get_account_balance()
            active_pos = self.broker.get_positions()

            def _norm_ticker(t: str) -> str:
                s = t.upper().replace("/", "-")
                if s.endswith("USD") and "-" not in s and len(s) > 3:
                    return f"{s[:-3]}-USD"
                return s

            port_positions = [
                PortfolioPosition(
                    ticker=_norm_ticker(p.symbol),
                    quantity=p.quantity,
                    average_price=p.entry_price,
                )
                for p in active_pos
            ]

            # Gather Macroeconomic Regime & Web Sentiment
            regime_text = ""
            reg_obj = None
            try:
                from bot.risk.regime_specialist import REGIME_SPECIALIST
                reg_obj = REGIME_SPECIALIST.analyze_market_regime(symbol)
                regime_text = f"- Regime: {reg_obj.regime_display} (Vol Score: {reg_obj.volatility_score:.1f}/100, ATR Ratio: {reg_obj.atr_ratio_90d:.2f}x)"
                if reg_obj.web_sentiment:
                    regime_text += f"\n- Web Sentiment: {reg_obj.web_sentiment.sentiment_label} (Fear & Greed Index: {reg_obj.web_sentiment.fear_greed_value}/100, Trend Bias: {reg_obj.trend_bias})"
            except Exception:
                pass

            # Gather Quantitative Screener Telemetry
            quant_text = (
                f"- Quantitative Setup: {screen_res.setup_name} (Bias: {screen_res.bias}, Score: {screen_res.score:.0f}/100)\n"
                f"- Technical Indicators: RSI(14)={screen_res.metrics.get('rsi')}, ATR%={screen_res.metrics.get('atr')}, Bollinger Bandwidth={screen_res.metrics.get('band_pct')}\n"
                f"- Mathematical Risk/Reward: {screen_res.risk_reward_ratio:.2f}:1 (Suggested SL: ${screen_res.suggested_stop_loss:.2f}, TP: ${screen_res.suggested_take_profit:.2f})\n"
                f"- Screener Consensus Rationale: {screen_res.rationale}"
            )

            # Gather Continuous Learning Lessons from Pattern Memory
            lessons_text = ""
            try:
                pat_check = self.risk_engine.pattern_memory.query_matching_patterns(
                    symbol=symbol,
                    side=OrderSide.BUY if screen_res.bias == "BUY" else OrderSide.SELL,
                    rsi=float(screen_res.metrics.get("rsi") or 50.0),
                    ema_ratio=1.0,
                    atr_pct=float(screen_res.metrics.get("atr") or 2.0),
                    market_regime=reg_obj.regime.value if reg_obj else "STABLE",
                )
                if pat_check.warning_message:
                    lessons_text = f"- ⚠️ Caution (Pattern Match): {pat_check.warning_message}. Lesson: {pat_check.learned_lesson}"
                else:
                    lessons_text = "- No recent recurring loss patterns detected for this setup. Maintain strict Stop-Loss discipline and capital preservation."
            except Exception:
                lessons_text = "- Maintain strict risk discipline."

            portfolio_ctx = EnhancedPortfolioContext(
                cash=bal.cash,
                currency=bal.currency or "USD",
                positions=port_positions,
                regime_context=regime_text,
                quant_context=quant_text,
                lessons_context=lessons_text,
            )
        except Exception as pe:
            logger.debug("Could not construct EnhancedPortfolioContext for graph: %s", pe)

        try:
            final_state, rating = graph.propagate(symbol, date, asset_type=asset_type, portfolio=portfolio_ctx)
        except Exception as err:
            logger.warning(
                "[QUANT_FAILOVER] ⚡ LLM comité no disponible o cuota agotada (%s). Activando ejecución algorítmica cuantitativa directa (Bias: %s, R:R %.2f)",
                err, screen_res.bias, screen_res.risk_reward_ratio
            )
            rating = screen_res.bias
            final_state = {
                "final_trade_decision": (
                    f"[ALGO_DIRECT_EXECUTION] Setup {screen_res.setup_name} confirmado por análisis "
                    f"cuantitativo determinista (Score: {screen_res.score:.0f}). Razón: {screen_res.rationale}. "
                    f"SL: ${screen_res.suggested_stop_loss:.2f}, TP: ${screen_res.suggested_take_profit:.2f}."
                ),
                "market_report": f"RSI: {screen_res.metrics.get('rsi')}, ATR: {screen_res.metrics.get('atr')}, Band Pct: {screen_res.metrics.get('band_pct')}",
                "trader_investment_plan": f"Ejecución algorítmica con R:R {screen_res.risk_reward_ratio:.2f}:1 para maximizar margen de ganancia."
            }

        # Record multi-agent consensus, perspective notes, and signals in Neon PostgreSQL
        try:
            analyst_signals = {
                "quant_setup": f"{screen_res.setup_name} ({screen_res.rationale})",
                "market_report": str(final_state.get("market_report") or "")[:500],
                "fundamentals_report": str(final_state.get("fundamentals_report") or "")[:500],
                "bull_perspective": str(final_state.get("bull_perspective") or "")[:500],
                "bear_perspective": str(final_state.get("bear_perspective") or "")[:500],
                "trader_plan": str(final_state.get("trader_investment_plan") or "")[:500],
            }
            reasoning = str(final_state.get("final_trade_decision") or "")
            self.db.save_debate(
                symbol=symbol,
                trade_date=date,
                rating=rating,
                action=self._map_rating_to_side(rating).value,
                current_price=current_price,
                analyst_signals=analyst_signals,
                reasoning=reasoning,
            )
        except Exception as e:
            logger.warning("[NEON] No se pudo registrar debate en Neon: %s", e)

        return self.process_signal_and_execute(symbol, rating, final_state, current_price, date, quant_result=screen_res)

    def manage_open_positions(self) -> list[Order]:
        """QuantConnect LEAN Trailing Stop & Position Manager.
        
        Monitors existing open positions, calculates High-Water Mark trailing stops,
        and submits exit orders to lock in profit or cut losses if dynamic stops are crossed.
        """
        closed_orders: list[Order] = []
        try:
            positions = self.broker.get_positions()
        except Exception as e:
            logger.warning("Could not fetch positions for management: %s", e)
            return []

        total_equity = 0.0
        try:
            bal = self.broker.get_account_balance()
            total_equity = bal.equity
        except Exception as be:
            logger.debug("Could not fetch balance for position rebalancing: %s", be)

        for p in positions:
            if p.quantity <= 0:
                continue

            try:
                curr_price = self.broker.get_current_price(p.symbol)
            except Exception:
                curr_price = p.current_price or p.entry_price

            p.current_price = curr_price

            # Area A: Multi-Stage Partial Take-Profit (Scale-Out Profit Harvesting)
            should_partial_tp, partial_qty = self.risk_engine.evaluate_partial_take_profit(
                symbol=p.symbol,
                current_price=curr_price,
                entry_price=p.entry_price,
                current_qty=p.quantity,
            )
            if should_partial_tp and 0 < partial_qty < p.quantity:
                current_stage = self.risk_engine.get_tp_stage(p.symbol) + 1
                partial_pnl_dollars = (curr_price - p.entry_price) * partial_qty
                partial_pnl_pct = ((curr_price - p.entry_price) / p.entry_price) * 100.0

                if current_stage == 1:
                    break_even_price = round(p.entry_price * 1.0015, 4)
                    p.stop_loss_price = max(p.stop_loss_price or 0.0, break_even_price)
                    logger.info(
                        "[PARTIAL_TAKE_PROFIT - ETAPA 1 (+1.2%%)] 💰 Toma parcial ejecutada para %s: Vendiendo %.6f u a $%.2f (PnL: +$%.2f / +%.2f%%). Stop elevado a Break-Even ($%.2f).",
                        p.symbol, partial_qty, curr_price, partial_pnl_dollars, partial_pnl_pct, break_even_price
                    )
                else:
                    guaranteed_profit_price = round(p.entry_price * 1.015, 4)
                    p.stop_loss_price = max(p.stop_loss_price or 0.0, guaranteed_profit_price)
                    logger.info(
                        "[PARTIAL_TAKE_PROFIT - ETAPA 2 (+3.0%%)] 🚀 Cosecha de beneficios mayores para %s: Vendiendo %.6f u a $%.2f (PnL: +$%.2f / +%.2f%%). Stop asegurado a +1.50%% ($%.2f).",
                        p.symbol, partial_qty, curr_price, partial_pnl_dollars, partial_pnl_pct, guaranteed_profit_price
                    )

                partial_order = Order(
                    order_id=str(uuid.uuid4())[:8],
                    symbol=p.symbol,
                    asset_class=p.asset_class,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=partial_qty,
                    limit_price=curr_price,
                    status=OrderStatus.PENDING,
                )
                executed_partial = self.broker.submit_order(partial_order)
                self.risk_engine.register_partial_tp(p.symbol)
                p.quantity -= partial_qty

                if hasattr(self.risk_engine, "pattern_memory") and self.risk_engine.pattern_memory:
                    try:
                        from bot.models import TradeOutcome
                        from bot.risk.regime_specialist import REGIME_SPECIALIST
                        regime_name = REGIME_SPECIALIST.analyze_market_regime(p.symbol.replace("/", "-")).regime.value
                        outcome = TradeOutcome(
                            trade_id=executed_partial.order_id or str(uuid.uuid4())[:8],
                            symbol=p.symbol,
                            asset_class=p.asset_class,
                            side=OrderSide.BUY,
                            entry_price=p.entry_price,
                            exit_price=curr_price,
                            quantity=partial_qty,
                            pnl_dollars=partial_pnl_dollars,
                            pnl_pct=partial_pnl_pct,
                            outcome="WIN",
                            market_regime=regime_name,
                        )
                        self.risk_engine.pattern_memory.record_trade_outcome(outcome)
                    except Exception as pe:
                        logger.debug("Failed to record partial TP outcome: %s", pe)

                self.notifier.notify_order_execution(executed_partial)
                try:
                    self.db.save_order(executed_partial)
                except Exception as e:
                    logger.warning("[NEON] Could not save partial order to Neon: %s", e)
                closed_orders.append(executed_partial)

            # Area B: Dynamic Portfolio Balancing & Concentration Trim
            if total_equity > 0:
                should_rebal, trim_qty, rebal_reason = self.risk_engine.evaluate_portfolio_rebalancing(
                    position=p,
                    current_price=curr_price,
                    total_equity=total_equity,
                )
                if should_rebal and 0 < trim_qty < p.quantity:
                    rebal_pnl_dollars = (curr_price - p.entry_price) * trim_qty
                    rebal_pnl_pct = ((curr_price - p.entry_price) / p.entry_price) * 100.0
                    logger.info(
                        "[PORTFOLIO_REBALANCE] ⚖️ %s. Vendiendo %.6f u a $%.2f (PnL liberado: +$%.2f / +%.2f%%)",
                        rebal_reason, trim_qty, curr_price, rebal_pnl_dollars, rebal_pnl_pct
                    )
                    rebal_order = Order(
                        order_id=str(uuid.uuid4())[:8],
                        symbol=p.symbol,
                        asset_class=p.asset_class,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=trim_qty,
                        limit_price=curr_price,
                        status=OrderStatus.PENDING,
                    )
                    executed_rebal = self.broker.submit_order(rebal_order)
                    p.quantity -= trim_qty
                    self.notifier.notify_order_execution(executed_rebal)
                    try:
                        self.db.save_order(executed_rebal)
                    except Exception as e:
                        logger.warning("[NEON] Could not save rebalance order to Neon: %s", e)
                    closed_orders.append(executed_rebal)

            default_sl = p.entry_price * (1.0 - (self.config.risk.default_stop_loss_pct / 100.0))
            current_sl = p.stop_loss_price or default_sl

            # Calculate LEAN trailing stop with per-symbol volatility adaptation
            new_sl, ratcheted = self.risk_engine.calculate_trailing_stop(
                symbol=p.symbol,
                current_price=curr_price,
                entry_price=p.entry_price,
                current_stop_loss=current_sl,
            )

            if ratcheted:
                p.stop_loss_price = new_sl
                logger.info(
                    "Trailing Stop for %s ratcheted UP to $%.2f (peak profit protection)",
                    p.symbol, new_sl,
                )

            # Check if stop loss was crossed
            should_exit = curr_price <= new_sl
            # Check take profit if defined
            if p.take_profit_price and curr_price >= p.take_profit_price:
                should_exit = True

            if should_exit:
                pnl_dollars = (curr_price - p.entry_price) * p.quantity
                pnl_pct = ((curr_price - p.entry_price) / p.entry_price) * 100.0
                is_loss = pnl_dollars < 0

                logger.warning(
                    "Protective exit triggered for %s at $%.2f (Stop: $%.2f, PnL: $%.2f / %.2f%%)",
                    p.symbol, curr_price, new_sl, pnl_dollars, pnl_pct,
                )

                sell_order = Order(
                    order_id=str(uuid.uuid4())[:8],
                    symbol=p.symbol,
                    asset_class=p.asset_class,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=p.quantity,
                    limit_price=curr_price,
                    status=OrderStatus.PENDING,
                )
                executed = self.broker.submit_order(sell_order)
                self.risk_engine.register_trade_exit(p.symbol, is_loss=is_loss)

                # Continuous Learning Feedback Loop: Register trade outcome into PatternMemory
                if hasattr(self.risk_engine, "pattern_memory") and self.risk_engine.pattern_memory:
                    try:
                        from bot.models import TradeOutcome
                        from bot.risk.regime_specialist import REGIME_SPECIALIST
                        regime_name = REGIME_SPECIALIST.analyze_market_regime("BTC-USD").regime.value
                        
                        outcome = TradeOutcome(
                            trade_id=executed.order_id or str(uuid.uuid4())[:8],
                            symbol=p.symbol,
                            asset_class=p.asset_class,
                            side=OrderSide.BUY,
                            entry_price=p.entry_price,
                            exit_price=curr_price,
                            quantity=p.quantity,
                            pnl_dollars=pnl_dollars,
                            pnl_pct=pnl_pct,
                            outcome="WIN" if not is_loss else "LOSS",
                            market_regime=regime_name,
                        )
                        self.risk_engine.pattern_memory.record_trade_outcome(outcome)
                    except Exception as pe:
                        logger.debug("Failed to record trade outcome in pattern memory: %s", pe)

                self.notifier.notify_order_execution(executed)
                try:
                    self.db.save_order(executed)
                except Exception as e:
                    logger.warning("[NEON] Could not save exit order to Neon: %s", e)
                closed_orders.append(executed)

        return closed_orders

    @staticmethod
    def is_equity_market_open() -> bool:
        """Check if US equity markets are currently in regular trading session (9:30 - 16:00 ET, Mon-Fri)."""
        now_utc = datetime.now(timezone.utc)
        if now_utc.weekday() >= 5:
            return False
        current_minute_utc = now_utc.hour * 60 + now_utc.minute
        # Regular trading hours: 13:30 UTC (9:30 AM EDT) to 20:00 UTC (4:00 PM EDT)
        return (13 * 60 + 30) <= current_minute_utc < (20 * 60)

    def run_portfolio_cycle(self, trade_date: str | None = None) -> list[Order]:
        """Run analysis on all symbols in the watchlist (Equities + Crypto)."""
        date = trade_date or get_current_date()
        
        # When equity market is closed overnight or weekends, prioritize active 24/7 liquid crypto
        equity_open = self.is_equity_market_open()
        if equity_open:
            all_symbols = self.config.watchlist_equities + self.config.watchlist_crypto
        else:
            all_symbols = self.config.watchlist_crypto
            
        executed_orders: list[Order] = []

        # 1. Manage and protect open positions with LEAN Trailing Stops
        exit_orders = self.manage_open_positions()
        executed_orders.extend(exit_orders)

        logger.info(
            "Starting portfolio scan for %d instruments on %s (Equity Market Open: %s)",
            len(all_symbols), date, equity_open
        )
        for sym in all_symbols:
            try:
                order = self.run_symbol(sym, trade_date=date)
                if order:
                    executed_orders.append(order)
            except Exception as e:
                logger.exception("Error processing symbol %s: %s", sym, e)

        return executed_orders
