from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from trading_agent.market import analyze_market
from trading_agent.models import Candle


def backtest_probabilities(
    candles: list[Candle],
    lookback: int = 80,
    horizon: int = 6,
    move_threshold_pct: float = 0.35,
) -> dict[str, Any]:
    if len(candles) < lookback + horizon + 1:
        raise ValueError("Not enough candles for backtest.")

    results: list[dict[str, Any]] = []
    for end in range(lookback, len(candles) - horizon):
        window = candles[end - lookback : end]
        state = analyze_market(window)
        entry_price = window[-1].close
        future_price = candles[end + horizon].close
        move_pct = ((future_price - entry_price) / entry_price) * 100
        actual = _actual_scenario(move_pct, move_threshold_pct)
        predicted = _predicted_scenario(state)
        confidence = max(
            state.bullish_probability,
            state.bearish_probability,
            state.sideways_probability,
        )
        results.append(
            {
                "predicted": predicted,
                "actual": actual,
                "correct": predicted == actual,
                "confidence": confidence,
                "move_pct": move_pct,
                "bullish_probability": state.bullish_probability,
                "bearish_probability": state.bearish_probability,
                "sideways_probability": state.sideways_probability,
            }
        )

    correct = sum(1 for row in results if row["correct"])
    by_scenario = _scenario_stats(results)
    high_confidence = [row for row in results if row["confidence"] >= 45]
    high_confidence_correct = sum(1 for row in high_confidence if row["correct"])

    return {
        "samples": len(results),
        "accuracy": _pct(correct, len(results)),
        "high_confidence_samples": len(high_confidence),
        "high_confidence_accuracy": _pct(high_confidence_correct, len(high_confidence)),
        "prediction_distribution": dict(Counter(row["predicted"] for row in results)),
        "actual_distribution": dict(Counter(row["actual"] for row in results)),
        "by_scenario": by_scenario,
        "last_10": results[-10:],
        "calibration_note": _calibration_note(results),
    }


def _predicted_scenario(state) -> str:
    values = {
        "bullish": state.bullish_probability,
        "bearish": state.bearish_probability,
        "sideways": state.sideways_probability,
    }
    return max(values.items(), key=lambda item: item[1])[0]


def _actual_scenario(move_pct: float, threshold_pct: float) -> str:
    if move_pct >= threshold_pct:
        return "bullish"
    if move_pct <= -threshold_pct:
        return "bearish"
    return "sideways"


def _scenario_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for scenario in ("bullish", "bearish", "sideways"):
        rows = [row for row in results if row["predicted"] == scenario]
        correct = sum(1 for row in rows if row["correct"])
        output[scenario] = {
            "samples": len(rows),
            "accuracy": _pct(correct, len(rows)),
            "avg_confidence": round(sum(row["confidence"] for row in rows) / len(rows), 2) if rows else 0.0,
        }
    return output


def _calibration_note(results: list[dict[str, Any]]) -> str:
    high = [row for row in results if row["confidence"] >= 55]
    if not high:
        return "No >=55% confidence samples yet; use more history before trusting calibration."
    actual_accuracy = _pct(sum(1 for row in high if row["correct"]), len(high))
    avg_confidence = sum(row["confidence"] for row in high) / len(high)
    gap = actual_accuracy - avg_confidence
    return f"For >=55% confidence predictions, realized accuracy was {actual_accuracy:.2f}% vs average stated confidence {avg_confidence:.2f}% (gap {gap:.2f}%)."


def _pct(value: int, total: int) -> float:
    return round((value / total) * 100, 2) if total else 0.0
