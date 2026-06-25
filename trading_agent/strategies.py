from __future__ import annotations

from dataclasses import dataclass

from trading_agent.models import Action, MarketState, StrategyName


@dataclass(frozen=True)
class StrategySignal:
    strategy: StrategyName
    action: Action
    confidence: int
    reasoning: str


def select_strategy(state: MarketState, weights: dict[str, float] | None = None) -> StrategyName:
    return _select_by_suitability(state, weights)


def select_signal(state: MarketState, weights: dict[str, float] | None = None, has_position: bool = False) -> StrategySignal:
    weights = weights or {}
    scored: list[tuple[float, StrategySignal]] = []
    for strategy in StrategyName:
        signal = run_strategy(strategy, state)
        suitability = _strategy_suitability(strategy, state) * weights.get(strategy.value, 1.0)
        scored.append((_action_aware_score(signal, suitability, has_position), signal))
    return max(scored, key=lambda item: item[0])[1]


def run_strategy(strategy: StrategyName, state: MarketState) -> StrategySignal:
    if strategy == StrategyName.BREAKOUT:
        return _breakout(state)
    if strategy == StrategyName.TREND_FOLLOWING:
        return _trend_following(state)
    return _mean_reversion(state)


def _select_by_suitability(state: MarketState, weights: dict[str, float] | None = None) -> StrategyName:
    weights = weights or {}
    scored = [
        (_strategy_suitability(strategy, state) * weights.get(strategy.value, 1.0), strategy)
        for strategy in StrategyName
    ]
    return max(scored, key=lambda item: item[0])[1]


def _strategy_suitability(strategy: StrategyName, state: MarketState) -> float:
    if strategy == StrategyName.BREAKOUT:
        return _breakout_suitability(state)
    if strategy == StrategyName.TREND_FOLLOWING:
        return _trend_suitability(state)
    return _mean_reversion_suitability(state)


def _action_aware_score(signal: StrategySignal, suitability: float, has_position: bool) -> float:
    if signal.action == Action.BUY:
        return 1_000 + suitability + signal.confidence
    if signal.action == Action.SELL:
        if has_position:
            return 800 + suitability + signal.confidence
        return -100 + (suitability * 0.1)
    return suitability + (signal.confidence * 0.25)


def _trend_following(state: MarketState) -> StrategySignal:
    bullish_edge = state.bullish_probability - max(state.bearish_probability, state.sideways_probability)
    if (
        state.trend == "bullish"
        and state.structure == "trend continuation"
        and state.volatility != "high"
        and 42 <= state.rsi <= 68
        and state.liquidity_score >= 45
        and state.fake_breakout_probability <= 55
        and state.bullish_probability >= 60
        and bullish_edge >= 15
    ):
        confidence = 68 + _quality_bonus(state)
        return StrategySignal(
            strategy=StrategyName.TREND_FOLLOWING,
            action=Action.BUY,
            confidence=min(88, confidence),
            reasoning="Trend is bullish and price is holding above the short-term average, but entry still needs risk filters.",
        )
    if state.trend == "bearish":
        return StrategySignal(
            strategy=StrategyName.TREND_FOLLOWING,
            action=Action.SELL,
            confidence=72,
            reasoning="Trend is bearish, so the professional spot response is to reduce exposure rather than force a long.",
        )
    return StrategySignal(
        strategy=StrategyName.TREND_FOLLOWING,
        action=Action.NO_TRADE,
        confidence=45,
        reasoning="Trend following is selected, but the trend does not have enough clean continuation quality.",
    )


def _breakout(state: MarketState) -> StrategySignal:
    if state.fake_breakout_probability > 55:
        return StrategySignal(
            strategy=StrategyName.BREAKOUT,
            action=Action.NO_TRADE,
            confidence=50,
            reasoning="Breakout exists, but the fake breakout probability is too high for a disciplined entry.",
        )
    if (
        state.structure == "breakout"
        and state.current_price > state.resistance
        and state.liquidity_score >= 55
        and state.fake_breakout_probability <= 40
        and state.volatility != "high"
        and state.bullish_probability >= 55
    ):
        confidence = 70 + _quality_bonus(state)
        return StrategySignal(
            strategy=StrategyName.BREAKOUT,
            action=Action.BUY,
            confidence=min(90, confidence),
            reasoning="Price broke resistance with acceptable liquidity and controlled fake-breakout risk.",
        )
    if state.current_price < state.support:
        return StrategySignal(
            strategy=StrategyName.BREAKOUT,
            action=Action.SELL,
            confidence=70,
            reasoning="Price broke support; in spot trading this favors exiting or staying in cash.",
        )
    return StrategySignal(
        strategy=StrategyName.BREAKOUT,
        action=Action.NO_TRADE,
        confidence=48,
        reasoning="The breakout setup is not confirmed enough to justify action.",
    )


def _mean_reversion(state: MarketState) -> StrategySignal:
    if state.volatility == "high":
        return StrategySignal(
            strategy=StrategyName.MEAN_REVERSION,
            action=Action.NO_TRADE,
            confidence=42,
            reasoning="Mean reversion is dangerous when volatility is high, so patience is better than forcing a trade.",
        )
    if (
        state.trend == "sideways"
        and state.sideways_probability >= 40
        and state.volatility != "high"
        and state.liquidity_score >= 45
        and state.rsi < 32
        and state.current_price <= state.support * 1.008
        and state.fake_breakout_probability <= 55
    ):
        return StrategySignal(
            strategy=StrategyName.MEAN_REVERSION,
            action=Action.BUY,
            confidence=72 + _quality_bonus(state),
            reasoning="Price is near range support and RSI is stretched, giving a controlled mean-reversion setup.",
        )
    if state.rsi > 70 or state.current_price >= state.resistance * 0.992:
        return StrategySignal(
            strategy=StrategyName.MEAN_REVERSION,
            action=Action.SELL,
            confidence=67,
            reasoning="Price is near resistance or momentum is overheated, so reducing exposure is favored.",
        )
    return StrategySignal(
        strategy=StrategyName.MEAN_REVERSION,
        action=Action.NO_TRADE,
        confidence=40,
        reasoning="The range is not offering a strong enough edge at support or resistance.",
    )


def _quality_bonus(state: MarketState) -> int:
    bonus = 0
    if state.liquidity_score >= 70:
        bonus += 6
    elif state.liquidity_score >= 45:
        bonus += 3
    if state.volatility == "medium":
        bonus += 4
    if state.fake_breakout_probability <= 35:
        bonus += 4
    return bonus


def _breakout_suitability(state: MarketState) -> float:
    score = 30.0
    if state.structure == "breakout":
        score += 45
        if state.fake_breakout_probability <= 40:
            score += 15
        if state.fake_breakout_probability > 55:
            score -= 60
    if state.liquidity_score >= 70:
        score += 10
    if state.volatility == "high":
        score -= 25
    return score


def _trend_suitability(state: MarketState) -> float:
    score = 30.0
    if state.trend in {"bullish", "bearish"}:
        score += 35
    if state.structure == "trend continuation":
        score += 25
    if state.volatility == "high":
        score -= 20
    return score


def _mean_reversion_suitability(state: MarketState) -> float:
    score = 30.0
    if state.structure == "range":
        score += 35
    if state.trend == "sideways":
        score += 20
    if state.volatility == "low":
        score += 10
    if state.volatility == "high":
        score -= 35
    return score
