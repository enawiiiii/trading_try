from __future__ import annotations

from dataclasses import dataclass

from trading_agent.config import AgentConfig
from trading_agent.models import Action, MarketState
from trading_agent.strategies import StrategySignal


@dataclass(frozen=True)
class RiskDecision:
    action: Action
    confidence: int
    risk_level: str
    reason: str


@dataclass(frozen=True)
class RiskContext:
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0
    has_position: bool = False


def apply_risk_rules(
    signal: StrategySignal,
    state: MarketState,
    config: AgentConfig,
    context: RiskContext,
) -> RiskDecision:
    risk_level = _risk_level(state)

    if context.daily_pnl_pct <= -abs(config.daily_loss_limit_pct):
        return RiskDecision(Action.NO_TRADE, min(signal.confidence, 20), "high", "Daily loss limit reached.")

    if context.consecutive_losses >= config.max_consecutive_losses:
        return RiskDecision(Action.NO_TRADE, min(signal.confidence, 20), "high", "Consecutive loss limit reached.")

    if state.liquidity_score < config.min_liquidity_score:
        return RiskDecision(Action.NO_TRADE, min(signal.confidence, 35), "high", "Liquidity is too weak.")

    if state.fake_breakout_probability > config.max_fake_breakout_probability:
        return RiskDecision(Action.NO_TRADE, min(signal.confidence, 40), "high", "Fake breakout risk is too high.")

    if state.volatility == "high" and signal.action == Action.BUY:
        return RiskDecision(Action.NO_TRADE, min(signal.confidence, 45), "high", "Volatility is too high for a fresh spot entry.")

    if signal.action == Action.BUY and not _bullish_edge_is_clear(state):
        return RiskDecision(Action.NO_TRADE, min(signal.confidence, 55), risk_level, "Bullish scenario does not have enough probability edge.")

    if signal.action == Action.SELL and not _bearish_edge_is_clear(state):
        return RiskDecision(Action.NO_TRADE, min(signal.confidence, 55), risk_level, "Bearish scenario does not have enough probability edge.")

    if signal.action == Action.BUY and signal.confidence < config.min_buy_confidence:
        return RiskDecision(Action.NO_TRADE, signal.confidence, risk_level, "Buy confidence is below the required threshold.")

    if signal.action == Action.SELL and not context.has_position:
        return RiskDecision(Action.NO_TRADE, signal.confidence, risk_level, "Sell signal exists, but paper portfolio has no spot position.")

    if signal.action == Action.SELL and signal.confidence < config.min_sell_confidence:
        return RiskDecision(Action.NO_TRADE, signal.confidence, risk_level, "Sell confidence is below the required threshold.")

    return RiskDecision(signal.action, signal.confidence, risk_level, "Risk rules allow the signal.")


def position_size(cash: float, price: float, state: MarketState, config: AgentConfig) -> float:
    risk_cash = cash * (config.risk_per_trade_pct / 100)
    assumed_stop_distance_pct = max(state.atr_pct * 1.5, 1.0)
    notional = risk_cash / (assumed_stop_distance_pct / 100)
    capped_notional = min(notional, cash * 0.25)
    return max(0.0, capped_notional / price)


def _risk_level(state: MarketState) -> str:
    if state.volatility == "high" or state.fake_breakout_probability >= 65 or state.liquidity == "low":
        return "high"
    if state.volatility == "medium" or state.fake_breakout_probability >= 45:
        return "medium"
    return "low"


def _bullish_edge_is_clear(state: MarketState) -> bool:
    strongest_non_bullish = max(state.bearish_probability, state.sideways_probability)
    return state.bullish_probability >= 45 and state.bullish_probability >= strongest_non_bullish + 5


def _bearish_edge_is_clear(state: MarketState) -> bool:
    strongest_non_bearish = max(state.bullish_probability, state.sideways_probability)
    return state.bearish_probability >= 45 and state.bearish_probability >= strongest_non_bearish + 5
