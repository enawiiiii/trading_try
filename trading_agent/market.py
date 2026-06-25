from __future__ import annotations

from statistics import mean

from trading_agent.indicators import atr, rsi, sma, volume_zscore
from trading_agent.models import Candle, MarketState
from trading_agent.probability import estimate_scenarios


def analyze_market(candles: list[Candle]) -> MarketState:
    if len(candles) < 60:
        raise ValueError("At least 60 candles are required for market analysis.")

    closes = [c.close for c in candles]
    current = candles[-1]
    current_price = current.close
    sma_20 = sma(closes, 20)
    sma_50 = sma(closes, 50)
    atr_value = atr(candles)
    atr_pct = (atr_value / current_price) * 100 if current_price else 0
    current_rsi = rsi(closes)

    recent_high = max(c.high for c in candles[-20:-1])
    recent_low = min(c.low for c in candles[-20:-1])
    volume_score = _liquidity_score(candles)
    vol_z = volume_zscore(candles)
    momentum_6_pct = _momentum_pct(closes, 6)
    momentum_12_pct = _momentum_pct(closes, 12)

    if sma_20 > sma_50 * 1.006 and current_price > sma_20:
        trend = "bullish"
    elif sma_20 < sma_50 * 0.994 and current_price < sma_20:
        trend = "bearish"
    else:
        trend = "sideways"

    if atr_pct < 1.2:
        volatility = "low"
    elif atr_pct < 3.5:
        volatility = "medium"
    else:
        volatility = "high"

    liquidity = "high" if volume_score >= 70 else "medium" if volume_score >= 40 else "low"

    broke_resistance = current.close > recent_high and current.volume >= mean(c.volume for c in candles[-20:-1]) * 1.15
    broke_support = current.close < recent_low and current.volume >= mean(c.volume for c in candles[-20:-1]) * 1.15
    near_support = abs(current_price - recent_low) / current_price < 0.012
    near_resistance = abs(current_price - recent_high) / current_price < 0.012

    if broke_resistance or broke_support:
        structure = "breakout"
    elif trend == "sideways" and (near_support or near_resistance):
        structure = "range"
    elif _possible_reversal(trend, current_rsi, candles):
        structure = "reversal"
    else:
        structure = "range" if trend == "sideways" else "trend continuation"

    fake_breakout_probability = _fake_breakout_probability(
        structure=structure,
        close=current.close,
        recent_high=recent_high,
        recent_low=recent_low,
        volume_z=vol_z,
        atr_pct=atr_pct,
        liquidity_score=volume_score,
    )
    scenarios = estimate_scenarios(
        trend=trend,
        volatility=volatility,
        liquidity_score=volume_score,
        structure=structure,
        fake_breakout_probability=fake_breakout_probability,
        rsi=current_rsi,
        current_price=current_price,
        support=recent_low,
        resistance=recent_high,
        momentum_6_pct=momentum_6_pct,
        momentum_12_pct=momentum_12_pct,
    )

    explanation = (
        f"Trend is {trend}; volatility is {volatility} at {atr_pct:.2f}% ATR; "
        f"liquidity is {liquidity}; structure looks like {structure}; "
        f"fake breakout probability is {fake_breakout_probability}%. "
        f"{scenarios.note}"
    )

    return MarketState(
        trend=trend,
        volatility=volatility,
        liquidity=liquidity,
        liquidity_score=volume_score,
        structure=structure,
        fake_breakout_probability=fake_breakout_probability,
        current_price=current_price,
        atr_pct=atr_pct,
        rsi=current_rsi,
        support=recent_low,
        resistance=recent_high,
        bullish_probability=scenarios.bullish,
        bearish_probability=scenarios.bearish,
        sideways_probability=scenarios.sideways,
        evidence_summary=" | ".join(scenarios.evidence),
        explanation=explanation,
    )


def _liquidity_score(candles: list[Candle]) -> int:
    quote_volumes = [c.quote_volume for c in candles[-30:]]
    trades = [c.trades for c in candles[-30:]]
    latest_volume = quote_volumes[-1]
    avg_volume = mean(quote_volumes)
    latest_trades = trades[-1]
    avg_trades = mean(trades)

    volume_ratio = latest_volume / avg_volume if avg_volume else 0
    if avg_trades:
        trades_ratio = latest_trades / avg_trades
        raw = (volume_ratio * 55) + (trades_ratio * 45)
    else:
        raw = volume_ratio * 100
    return max(0, min(100, round(raw)))


def _momentum_pct(closes: list[float], period: int) -> float:
    if len(closes) <= period or closes[-period - 1] == 0:
        return 0.0
    return ((closes[-1] - closes[-period - 1]) / closes[-period - 1]) * 100


def _possible_reversal(trend: str, current_rsi: float, candles: list[Candle]) -> bool:
    last_three = candles[-3:]
    bullish_push = all(c.close > c.open for c in last_three)
    bearish_push = all(c.close < c.open for c in last_three)
    return (trend == "bearish" and current_rsi < 35 and bullish_push) or (
        trend == "bullish" and current_rsi > 68 and bearish_push
    )


def _fake_breakout_probability(
    structure: str,
    close: float,
    recent_high: float,
    recent_low: float,
    volume_z: float,
    atr_pct: float,
    liquidity_score: int,
) -> int:
    if structure != "breakout":
        return 30 if liquidity_score >= 40 else 50

    distance_above = max(0.0, (close - recent_high) / close * 100)
    distance_below = max(0.0, (recent_low - close) / close * 100)
    breakout_distance = max(distance_above, distance_below)

    probability = 55
    if volume_z > 1.0:
        probability -= 20
    if breakout_distance > atr_pct * 0.35:
        probability -= 15
    if liquidity_score < 40:
        probability += 20
    if atr_pct > 4:
        probability += 15

    return max(5, min(90, round(probability)))
