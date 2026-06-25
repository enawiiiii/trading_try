from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig
from trading_agent.journal import load_recent
from trading_agent.models import StrategyName


DEFAULT_WEIGHTS = {
    StrategyName.BREAKOUT.value: 1.0,
    StrategyName.TREND_FOLLOWING.value: 1.0,
    StrategyName.MEAN_REVERSION.value: 1.0,
}


@dataclass(frozen=True)
class LearningState:
    strategy_weights: dict[str, float]
    last_updated: str | None = None
    observations: int = 0


def load_learning_state(config: AgentConfig) -> LearningState:
    path = Path(config.learning_path)
    if not path.exists():
        return LearningState(strategy_weights=DEFAULT_WEIGHTS.copy())

    raw = json.loads(path.read_text(encoding="utf-8"))
    weights = DEFAULT_WEIGHTS.copy()
    weights.update({key: float(value) for key, value in raw.get("strategy_weights", {}).items()})
    return LearningState(
        strategy_weights={key: _clamp_weight(value) for key, value in weights.items()},
        last_updated=raw.get("last_updated"),
        observations=int(raw.get("observations", 0)),
    )


def save_learning_state(config: AgentConfig, state: LearningState) -> None:
    path = Path(config.learning_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_weights": state.strategy_weights,
        "last_updated": state.last_updated,
        "observations": state.observations,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def learning_note(state: LearningState, strategy: StrategyName) -> str:
    weight = state.strategy_weights.get(strategy.value, 1.0)
    if state.observations == 0:
        return f"No closed-trade learning sample yet; using neutral {strategy.value} weight {weight:.2f}."
    return f"Using {strategy.value} weight {weight:.2f} from {state.observations} closed trade observations; risk rules unchanged."


def update_weekly_weights(config: AgentConfig) -> dict[str, Any]:
    current = load_learning_state(config)
    records = load_recent(config, limit=500)
    performance: dict[str, list[float]] = {key: [] for key in DEFAULT_WEIGHTS}

    for record in records:
        strategy = record.get("selected_strategy")
        execution = record.get("extra", {}).get("paper_execution", {})
        pnl = execution.get("realized_pnl")
        if strategy in performance and pnl is not None:
            performance[strategy].append(float(pnl))

    weights = current.strategy_weights.copy()
    observations = sum(len(values) for values in performance.values())
    for strategy, pnls in performance.items():
        if not pnls:
            continue
        average_pnl = sum(pnls) / len(pnls)
        adjustment = 0.03 if average_pnl > 0 else -0.03 if average_pnl < 0 else 0.0
        weights[strategy] = _clamp_weight(weights.get(strategy, 1.0) + adjustment)

    updated = LearningState(
        strategy_weights=weights,
        last_updated=datetime.now(timezone.utc).isoformat(),
        observations=observations,
    )
    save_learning_state(config, updated)

    return {
        "strategy_weights": updated.strategy_weights,
        "observations": observations,
        "performance_samples": {key: len(value) for key, value in performance.items()},
        "note": "Strategy weights updated only from closed paper trades. Core risk rules were not modified.",
    }


def _clamp_weight(value: float) -> float:
    return round(max(0.7, min(1.3, value)), 4)
