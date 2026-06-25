from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig
from trading_agent.models import MarketState


@dataclass(frozen=True)
class CalibrationState:
    scenario_accuracy: dict[str, float]
    samples: int
    symbol: str
    interval: str
    updated_at: str


def load_calibration(config: AgentConfig) -> CalibrationState | None:
    path = Path(config.calibration_path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationState(
        scenario_accuracy={key: float(value) for key, value in raw.get("scenario_accuracy", {}).items()},
        samples=int(raw.get("samples", 0)),
        symbol=str(raw.get("symbol", "")),
        interval=str(raw.get("interval", "")),
        updated_at=str(raw.get("updated_at", "")),
    )


def save_calibration(config: AgentConfig, symbol: str, interval: str, report: dict[str, Any]) -> CalibrationState:
    by_scenario = report.get("by_scenario", {})
    scenario_accuracy = {
        scenario: float(by_scenario.get(scenario, {}).get("accuracy", 0.0))
        for scenario in ("bullish", "bearish", "sideways")
    }
    state = CalibrationState(
        scenario_accuracy=scenario_accuracy,
        samples=int(report.get("samples", 0)),
        symbol=symbol.upper(),
        interval=interval,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    path = Path(config.calibration_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.__dict__, indent=2, ensure_ascii=True), encoding="utf-8")
    return state


def apply_calibration(state: MarketState, calibration: CalibrationState | None, symbol: str, interval: str) -> MarketState:
    if not calibration or calibration.samples < 100:
        return state
    if calibration.symbol and calibration.symbol != symbol.upper():
        return state
    if calibration.interval and calibration.interval != interval:
        return state

    raw = {
        "bullish": state.bullish_probability,
        "bearish": state.bearish_probability,
        "sideways": state.sideways_probability,
    }
    adjusted = {}
    for scenario, probability in raw.items():
        accuracy = calibration.scenario_accuracy.get(scenario, 33.0)
        factor = max(0.25, accuracy / 50.0)
        adjusted[scenario] = probability * factor

    total = sum(adjusted.values())
    if total <= 0:
        return state

    bullish = round(adjusted["bullish"] / total * 100)
    bearish = round(adjusted["bearish"] / total * 100)
    sideways = 100 - bullish - bearish
    calibration_note = (
        f" Calibrated probabilities: bullish {bullish}%, bearish {bearish}%, sideways {sideways}%. "
        f"Calibration used {calibration.samples} historical samples: "
        f"bullish accuracy {calibration.scenario_accuracy.get('bullish', 0):.2f}%, "
        f"bearish accuracy {calibration.scenario_accuracy.get('bearish', 0):.2f}%, "
        f"sideways accuracy {calibration.scenario_accuracy.get('sideways', 0):.2f}%."
    )

    return replace(
        state,
        bullish_probability=bullish,
        bearish_probability=bearish,
        sideways_probability=sideways,
        explanation=state.explanation + calibration_note,
    )
