from __future__ import annotations

from statistics import mean, pstdev

from trading_agent.models import Candle


def sma(values: list[float], period: int) -> float:
    if len(values) < period:
        return mean(values)
    return mean(values[-period:])


def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0

    gains: list[float] = []
    losses: list[float] = []
    for previous, current in zip(closes[-period - 1 : -1], closes[-period:]):
        change = current - previous
        if change >= 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))

    avg_gain = mean(gains)
    avg_loss = mean(losses)
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0

    true_ranges: list[float] = []
    recent = candles[-period:]
    previous_close = candles[-period - 1].close if len(candles) > period else candles[0].close

    for candle in recent:
        true_range = max(
            candle.high - candle.low,
            abs(candle.high - previous_close),
            abs(candle.low - previous_close),
        )
        true_ranges.append(true_range)
        previous_close = candle.close

    return mean(true_ranges)


def volume_zscore(candles: list[Candle], period: int = 30) -> float:
    volumes = [c.quote_volume for c in candles[-period:]]
    if len(volumes) < 5:
        return 0.0
    deviation = pstdev(volumes)
    if deviation == 0:
        return 0.0
    return (volumes[-1] - mean(volumes)) / deviation
