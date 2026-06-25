from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioProbabilities:
    bullish: int
    bearish: int
    sideways: int
    evidence: list[str]
    note: str


def estimate_scenarios(
    trend: str,
    volatility: str,
    liquidity_score: int,
    structure: str,
    fake_breakout_probability: int,
    rsi: float,
    current_price: float,
    support: float,
    resistance: float,
    momentum_6_pct: float = 0.0,
    momentum_12_pct: float = 0.0,
) -> ScenarioProbabilities:
    scores = {"bullish": 33.0, "bearish": 33.0, "sideways": 34.0}
    evidence: list[tuple[str, str, float]] = []

    if trend == "bullish":
        _add(scores, evidence, "bullish", 18, "Bullish trend: price structure and moving averages favor upside.")
        _add(scores, evidence, "bearish", -8, "Bullish trend reduces immediate downside evidence.")
        _add(scores, evidence, "sideways", -10, "Directional trend reduces range probability.")
    elif trend == "bearish":
        _add(scores, evidence, "bearish", 18, "Bearish trend: price structure and moving averages favor downside.")
        _add(scores, evidence, "bullish", -8, "Bearish trend reduces immediate upside evidence.")
        _add(scores, evidence, "sideways", -10, "Directional trend reduces range probability.")
    else:
        _add(scores, evidence, "sideways", 14, "Sideways trend: no clean directional control.")
        _add(scores, evidence, "bullish", -7, "Sideways trend weakens bullish follow-through.")
        _add(scores, evidence, "bearish", -7, "Sideways trend weakens bearish follow-through.")

    if structure == "breakout":
        if current_price > resistance and fake_breakout_probability <= 55:
            _add(scores, evidence, "bullish", 18, "Confirmed resistance breakout supports bullish continuation.")
            _add(scores, evidence, "sideways", -8, "Confirmed breakout reduces range expectation.")
            _add(scores, evidence, "bearish", -10, "Upside breakout weakens bearish case.")
        elif current_price < support and fake_breakout_probability <= 55:
            _add(scores, evidence, "bearish", 18, "Confirmed support breakdown supports downside continuation.")
            _add(scores, evidence, "sideways", -8, "Confirmed breakdown reduces range expectation.")
            _add(scores, evidence, "bullish", -10, "Downside break weakens bullish case.")
        else:
            _add(scores, evidence, "sideways", 10, "Breakout exists but confirmation is weak or fake-move risk is elevated.")
            _add(scores, evidence, "bullish", -5, "Weak breakout lowers bullish conviction.")
            _add(scores, evidence, "bearish", -5, "Weak breakout lowers bearish conviction.")
    elif structure == "range":
        _add(scores, evidence, "sideways", 18, "Range structure favors mean reversion and waiting.")
        if rsi < 35:
            _add(scores, evidence, "bullish", 8, "RSI is stretched low near range conditions, supporting bounce probability.")
            _add(scores, evidence, "bearish", -8, "Oversold RSI weakens fresh downside probability.")
        elif rsi > 70:
            _add(scores, evidence, "bearish", 8, "RSI is overheated near range conditions, supporting pullback probability.")
            _add(scores, evidence, "bullish", -8, "Overbought RSI weakens fresh upside probability.")
    elif structure == "reversal":
        if trend == "bearish":
            _add(scores, evidence, "bullish", 12, "Potential bullish reversal after bearish trend.")
            _add(scores, evidence, "bearish", -10, "Reversal evidence weakens trend-continuation downside.")
        elif trend == "bullish":
            _add(scores, evidence, "bearish", 12, "Potential bearish reversal after bullish trend.")
            _add(scores, evidence, "bullish", -10, "Reversal evidence weakens trend-continuation upside.")

    if liquidity_score >= 70:
        if scores["bullish"] > scores["bearish"]:
            _add(scores, evidence, "bullish", 5, "High liquidity confirms the stronger bullish scenario.")
        elif scores["bearish"] > scores["bullish"]:
            _add(scores, evidence, "bearish", 5, "High liquidity confirms the stronger bearish scenario.")
    elif liquidity_score < 35:
        _add(scores, evidence, "sideways", 12, "Weak liquidity increases uncertainty and favors waiting.")
        _add(scores, evidence, "bullish", -6, "Weak liquidity reduces bullish reliability.")
        _add(scores, evidence, "bearish", -6, "Weak liquidity reduces bearish reliability.")

    if volatility == "high":
        _add(scores, evidence, "sideways", 8, "High volatility raises chaos risk and lowers prediction quality.")
        _add(scores, evidence, "bullish", -4, "High volatility weakens bullish signal quality.")
        _add(scores, evidence, "bearish", -4, "High volatility weakens bearish signal quality.")
    elif volatility == "low" and structure == "range":
        _add(scores, evidence, "sideways", 6, "Low volatility inside a range supports sideways continuation.")

    if fake_breakout_probability >= 65:
        _add(scores, evidence, "sideways", 10, "High fake breakout probability favors no-trade/range outcome.")
        _add(scores, evidence, "bullish", -5, "High fake-move risk lowers bullish conviction.")
        _add(scores, evidence, "bearish", -5, "High fake-move risk lowers bearish conviction.")

    if momentum_6_pct >= 0.45 and momentum_12_pct >= 0.25:
        _add(scores, evidence, "bullish", 14, "Recent 6/12-candle momentum confirms upside pressure.")
        _add(scores, evidence, "sideways", -8, "Strong recent momentum reduces sideways probability.")
        _add(scores, evidence, "bearish", -6, "Strong upside momentum weakens bearish probability.")
    elif momentum_6_pct <= -0.45 and momentum_12_pct <= -0.25:
        _add(scores, evidence, "bearish", 14, "Recent 6/12-candle momentum confirms downside pressure.")
        _add(scores, evidence, "sideways", -8, "Strong recent downside momentum reduces sideways probability.")
        _add(scores, evidence, "bullish", -6, "Strong downside momentum weakens bullish probability.")
    elif abs(momentum_6_pct) < 0.2 and abs(momentum_12_pct) < 0.3:
        _add(scores, evidence, "sideways", 8, "Recent momentum is muted, supporting sideways probability.")

    bullish, bearish, sideways = _normalize([scores["bullish"], scores["bearish"], scores["sideways"]])
    top_evidence = _top_evidence(evidence)
    note = (
        f"Scenario probabilities: bullish {bullish}%, bearish {bearish}%, "
        f"sideways {sideways}%. Evidence: {'; '.join(top_evidence)}"
    )
    return ScenarioProbabilities(
        bullish=bullish,
        bearish=bearish,
        sideways=sideways,
        evidence=top_evidence,
        note=note,
    )


def strongest_scenario(probabilities: ScenarioProbabilities) -> str:
    values = {
        "bullish": probabilities.bullish,
        "bearish": probabilities.bearish,
        "sideways": probabilities.sideways,
    }
    return max(values.items(), key=lambda item: item[1])[0]


def _add(scores: dict[str, float], evidence: list[tuple[str, str, float]], scenario: str, weight: float, reason: str) -> None:
    scores[scenario] += weight
    if weight:
        evidence.append((scenario, reason, weight))


def _top_evidence(evidence: list[tuple[str, str, float]], limit: int = 4) -> list[str]:
    ranked = sorted(evidence, key=lambda item: abs(item[2]), reverse=True)
    return [f"{scenario} {weight:+.0f}: {reason}" for scenario, reason, weight in ranked[:limit]]


def _normalize(values: list[float]) -> tuple[int, int, int]:
    clipped = [max(5.0, value) for value in values]
    total = sum(clipped)
    raw = [value / total * 100 for value in clipped]
    rounded = [int(round(value)) for value in raw]
    delta = 100 - sum(rounded)
    rounded[rounded.index(max(rounded))] += delta
    return rounded[0], rounded[1], rounded[2]
