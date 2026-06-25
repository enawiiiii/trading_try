from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig
from trading_agent.data import fetch_market_klines
from trading_agent.models import Candle
from trading_agent.trade_outcome import horizon_for_interval, long_outcome


def analyze_edge_outcomes(
    config: AgentConfig,
    exchange: str,
    hours: int = 24,
    limit: int = 1000,
) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    events = _load_events(config, cutoff)
    buy_events = [event for event in events if _is_observed_buy_setup(event)]
    candles_by_key: dict[str, list[Candle]] = {}
    rows: list[dict[str, Any]] = []

    for event in buy_events:
        symbol = str(event.get("symbol", "")).upper()
        interval = str(event.get("interval", "1h"))
        if not symbol:
            continue
        key = f"{symbol}_{interval}"
        if key not in candles_by_key:
            candles_by_key[key] = fetch_market_klines(exchange, symbol, interval, limit, config.bybit_market_testnet)
        rows.append(_evaluate_event(config, event, candles_by_key[key]))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hours": hours,
        "events": len(events),
        "buy_events": len(buy_events),
        "evaluated_buy_events": len(rows),
        "summary": _summarize(rows),
        "by_edge_action": _summarize_by(rows, "edge_action"),
        "by_symbol_interval": _summarize_by(rows, "symbol_interval"),
        "by_edge_symbol_interval": _summarize_by(rows, "edge_symbol_interval"),
        "recent_pending": [row for row in rows if row["status"] == "pending"][-10:],
    }
    path = Path(config.edge_events_path).with_name("edge_outcome_report.json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    report["saved_path"] = str(path)
    return report


def _load_events(config: AgentConfig, cutoff: datetime) -> list[dict[str, Any]]:
    path = Path(config.edge_events_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        timestamp = _parse_time(event.get("timestamp"))
        if timestamp and timestamp >= cutoff:
            rows.append(event)
    return rows


def _is_observed_buy_setup(event: dict[str, Any]) -> bool:
    return event.get("signal_action") == "BUY" or event.get("final_action") == "BUY"


def _evaluate_event(config: AgentConfig, event: dict[str, Any], candles: list[Candle]) -> dict[str, Any]:
    timestamp = _parse_time(event.get("timestamp"))
    interval = str(event.get("interval", "1h"))
    horizon = horizon_for_interval(interval)
    edge = event.get("edge_filter") or {}
    price = _float_or_none(event.get("price"))
    future = [candle for candle in candles if timestamp and candle.close_time > timestamp]
    base = {
        "timestamp": event.get("timestamp"),
        "symbol": str(event.get("symbol", "")).upper(),
        "interval": interval,
        "symbol_interval": f"{str(event.get('symbol', '')).upper()}_{interval}",
        "edge_action": edge.get("monitor_action", "UNKNOWN"),
        "edge_symbol_interval": f"{edge.get('monitor_action', 'UNKNOWN')}|{str(event.get('symbol', '')).upper()}_{interval}",
        "signal_action": event.get("signal_action"),
        "final_action": event.get("final_action"),
        "score": edge.get("score"),
        "estimated_ev_pct": edge.get("estimated_ev_pct"),
        "status": "pending",
    }
    if timestamp is None or price is None:
        return {**base, "status": "invalid"}
    if len(future) < horizon:
        return {**base, "future_candles": len(future)}
    outcome = long_outcome(price, future[:horizon], config.take_profit_pct, config.stop_loss_pct)
    return {**base, **outcome, "status": "evaluated", "future_candles": horizon}


def _summarize_by(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "UNKNOWN"))].append(row)
    return [
        {"key": key, **_summarize(items)}
        for key, items in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    ]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    pending = [row for row in rows if row.get("status") == "pending"]
    wins = [row for row in evaluated if float(row.get("realized_pct", 0) or 0) > 0]
    losses = [row for row in evaluated if float(row.get("realized_pct", 0) or 0) < 0]
    realized = [float(row.get("realized_pct", 0) or 0) for row in evaluated]
    outcomes: dict[str, int] = defaultdict(int)
    for row in evaluated:
        outcomes[str(row.get("outcome", "unknown"))] += 1
    return {
        "total": len(rows),
        "evaluated": len(evaluated),
        "pending": len(pending),
        "win_rate_pct": round(len(wins) / len(evaluated) * 100, 2) if evaluated else 0.0,
        "loss_rate_pct": round(len(losses) / len(evaluated) * 100, 2) if evaluated else 0.0,
        "avg_realized_pct": round(sum(realized) / len(realized), 4) if realized else 0.0,
        "best_realized_pct": round(max(realized), 4) if realized else 0.0,
        "worst_realized_pct": round(min(realized), 4) if realized else 0.0,
        "outcomes": dict(sorted(outcomes.items())),
    }


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
