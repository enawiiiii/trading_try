from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig
from trading_agent.models import Decision


def append_decision(config: AgentConfig, decision: Decision, extra: dict[str, Any] | None = None) -> None:
    path = Path(config.journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(asdict(decision))
    payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
    if extra:
        payload["extra"] = _json_safe(extra)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=True) + "\n")


def load_recent(config: AgentConfig, limit: int = 20) -> list[dict[str, Any]]:
    path = Path(config.journal_path)
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()]


def summarize_learning(config: AgentConfig) -> dict[str, Any]:
    from trading_agent.learning import load_learning_state

    records = load_recent(config, limit=200)
    trades = [r for r in records if r.get("action") in {"BUY", "SELL"}]
    no_trades = [r for r in records if r.get("action") == "NO TRADE"]
    by_strategy: dict[str, int] = {}
    realized_pnl_by_strategy: dict[str, float] = {}
    for record in records:
        strategy = record.get("selected_strategy", "unknown")
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
        pnl = record.get("extra", {}).get("paper_execution", {}).get("realized_pnl")
        if pnl is not None:
            realized_pnl_by_strategy[strategy] = realized_pnl_by_strategy.get(strategy, 0.0) + float(pnl)

    return {
        "records": len(records),
        "trades": len(trades),
        "no_trades": len(no_trades),
        "strategy_usage": by_strategy,
        "realized_pnl_by_strategy": realized_pnl_by_strategy,
        "strategy_weights": load_learning_state(config).strategy_weights,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value
