from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig


def build_learning_monitor_report(config: AgentConfig) -> dict[str, Any]:
    training_report = _load_json(Path(config.ml_report_path))
    daemon_status = _load_json(Path(config.ml_report_path).with_name("training_daemon_status.json"))
    models = training_report.get("models", {}) if isinstance(training_report, dict) else {}

    rows = [_model_row(key, value) for key, value in models.items() if isinstance(value, dict)]
    approved = [row for row in rows if row["approved"]]
    rejected = [row for row in rows if not row["approved"] and not row["error"]]
    errors = [row for row in rows if row["error"]]
    warnings = _warnings(rows, daemon_status)
    scenario_distribution = Counter(row["current_scenario"] for row in rows if row["current_scenario"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "daemon": {
            "state": daemon_status.get("state", "unknown") if isinstance(daemon_status, dict) else "unknown",
            "cycle": daemon_status.get("cycle") if isinstance(daemon_status, dict) else None,
            "started_at": daemon_status.get("started_at") if isinstance(daemon_status, dict) else None,
            "finished_at": daemon_status.get("finished_at") if isinstance(daemon_status, dict) else None,
            "sleep_minutes": daemon_status.get("sleep_minutes") if isinstance(daemon_status, dict) else None,
        },
        "training_summary": training_report.get("summary", {}) if isinstance(training_report, dict) else {},
        "model_health": {
            "total": len(rows),
            "approved": len(approved),
            "rejected": len(rejected),
            "errors": len(errors),
            "approval_rate_pct": _pct(len(approved), len(rows)),
            "avg_accuracy": _avg(row["accuracy"] for row in rows),
            "avg_backtest_accuracy": _avg(row["backtest_accuracy"] for row in rows),
            "avg_accuracy_backtest_gap": _avg(row["accuracy_backtest_gap"] for row in rows),
            "current_scenario_distribution": dict(scenario_distribution),
        },
        "risk_flags": warnings,
        "top_models": _top(rows, reverse=True),
        "weak_models": _top(rows, reverse=False),
        "notes": _notes(rows, warnings),
    }


def save_learning_monitor_report(config: AgentConfig, report: dict[str, Any]) -> Path:
    path = Path(config.ml_report_path).with_name("learning_monitor_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def _model_row(key: str, item: dict[str, Any]) -> dict[str, Any]:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
    prediction = item.get("current_prediction", {}) if isinstance(item.get("current_prediction"), dict) else {}
    by_label = metrics.get("by_label", {}) if isinstance(metrics.get("by_label"), dict) else {}
    low_label_accuracy = [
        label
        for label, values in by_label.items()
        if isinstance(values, dict) and float(values.get("samples", 0) or 0) > 0 and float(values.get("accuracy", 0) or 0) < 10
    ]
    accuracy = float(metrics.get("accuracy", 0.0) or 0.0)
    backtest_accuracy = float(item.get("backtest_accuracy", 0.0) or 0.0)
    samples = int(metrics.get("samples", 0) or 0)

    return {
        "key": key,
        "approved": bool(item.get("approved", False)),
        "error": item.get("error"),
        "candles": int(item.get("candles", 0) or 0),
        "samples": samples,
        "accuracy": accuracy,
        "backtest_accuracy": backtest_accuracy,
        "accuracy_backtest_gap": round(accuracy - backtest_accuracy, 2),
        "current_scenario": prediction.get("scenario"),
        "current_confidence": float(prediction.get("confidence", 0.0) or 0.0),
        "low_label_accuracy": low_label_accuracy,
    }


def _warnings(rows: list[dict[str, Any]], daemon_status: dict[str, Any]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if daemon_status and daemon_status.get("state") not in {"running", "sleeping"}:
        warnings.append({"level": "high", "message": f"Training daemon state is {daemon_status.get('state', 'unknown')}."})

    small_sample_models = [row["key"] for row in rows if row["samples"] and row["samples"] < 300]
    if small_sample_models:
        warnings.append({"level": "medium", "message": f"Small validation samples: {', '.join(small_sample_models[:8])}."})

    large_gaps = [row["key"] for row in rows if row["accuracy_backtest_gap"] >= 10]
    if large_gaps:
        warnings.append({"level": "medium", "message": f"Accuracy is much higher than backtest on: {', '.join(large_gaps[:8])}."})

    label_blind = [row["key"] for row in rows if row["low_label_accuracy"]]
    if label_blind:
        warnings.append({"level": "high", "message": f"Some models are nearly blind to a class: {', '.join(label_blind[:8])}."})

    scenario_counts = Counter(row["current_scenario"] for row in rows if row["current_scenario"])
    total_predictions = sum(scenario_counts.values())
    if total_predictions:
        scenario, count = scenario_counts.most_common(1)[0]
        if count / total_predictions >= 0.65:
            warnings.append({"level": "medium", "message": f"Current predictions are biased toward {scenario} ({_pct(count, total_predictions)}%)."})

    return warnings


def _top(rows: list[dict[str, Any]], reverse: bool) -> list[dict[str, Any]]:
    clean = [row for row in rows if not row["error"]]
    clean.sort(key=lambda row: (row["accuracy"], row["backtest_accuracy"], row["samples"]), reverse=reverse)
    return clean[:8]


def _notes(rows: list[dict[str, Any]], warnings: list[dict[str, str]]) -> list[str]:
    notes = [
        "This monitor does not change trading decisions, risk rules, or execution behavior.",
        "Accuracy is treated as a weak signal until F1, confusion matrix, and walk-forward validation are added.",
    ]
    if warnings:
        notes.append("Use the risk flags to decide what to inspect; do not approve models only because accuracy is above threshold.")
    if rows:
        notes.append("The current phase is observation, not final evaluation.")
    return notes


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON in {path}"}


def _avg(values) -> float:
    clean = [float(value) for value in values if value is not None]
    return round(sum(clean) / len(clean), 2) if clean else 0.0


def _pct(value: int, total: int) -> float:
    return round((value / total) * 100, 2) if total else 0.0
