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


def build_blocked_winners_audit(
    config: AgentConfig,
    exchange: str,
    hours: int = 24,
    limit: int = 1000,
    min_evaluated: int = 3,
) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    events = _load_events(config, cutoff)
    candidates = [event for event in events if _is_rejected_buy(event)]
    candles_by_key: dict[str, list[Candle]] = {}
    rows: list[dict[str, Any]] = []

    for event in candidates:
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
        "rejected_buy_events": len(candidates),
        "evaluated_rejected_buy_events": len(rows),
        "summary": _summarize(rows),
        "best_pockets": _best_pockets(rows, min_evaluated=min_evaluated),
        "by_symbol_interval_strategy_regime": _summarize_by(rows, "symbol_interval_strategy_regime"),
        "by_symbol_interval_strategy": _summarize_by(rows, "symbol_interval_strategy"),
        "by_symbol_interval": _summarize_by(rows, "symbol_interval"),
        "by_edge_action": _summarize_by(rows, "edge_action"),
        "recent_winners": [row for row in rows if row.get("status") == "evaluated" and float(row.get("realized_pct", 0) or 0) > 0][-20:],
    }
    path = Path(config.edge_events_path).with_name("blocked_winners_audit.json")
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


def _is_rejected_buy(event: dict[str, Any]) -> bool:
    if event.get("signal_action") != "BUY" and event.get("final_action") != "BUY":
        return False
    edge = event.get("edge_filter") or {}
    action = edge.get("monitor_action") or edge.get("tier") or event.get("edge_action")
    return action in {"BLOCK", "OBSERVE"}


def _evaluate_event(config: AgentConfig, event: dict[str, Any], candles: list[Candle]) -> dict[str, Any]:
    timestamp = _parse_time(event.get("timestamp"))
    interval = str(event.get("interval", "1h"))
    symbol = str(event.get("symbol", "")).upper()
    edge = event.get("edge_filter") or {}
    edge_action = edge.get("monitor_action") or edge.get("tier") or "UNKNOWN"
    strategy = str(event.get("strategy") or "UNKNOWN")
    regime = str(edge.get("regime") or "UNKNOWN")
    price = _float_or_none(event.get("price"))
    base = {
        "timestamp": event.get("timestamp"),
        "symbol": symbol,
        "interval": interval,
        "symbol_interval": f"{symbol}_{interval}",
        "strategy": strategy,
        "regime": regime,
        "edge_action": edge_action,
        "score": edge.get("score"),
        "estimated_ev_pct": edge.get("estimated_ev_pct"),
        "symbol_interval_strategy": f"{symbol}_{interval}|{strategy}",
        "symbol_interval_strategy_regime": f"{symbol}_{interval}|{strategy}|{regime}",
        "status": "pending",
    }
    if timestamp is None or price is None:
        return {**base, "status": "invalid"}
    horizon = horizon_for_interval(interval)
    future = [candle for candle in candles if candle.close_time > timestamp]
    if len(future) < horizon:
        return {**base, "future_candles": len(future)}
    outcome = long_outcome(price, future[:horizon], config.take_profit_pct, config.stop_loss_pct)
    return {**base, **outcome, "status": "evaluated", "future_candles": horizon}


def _best_pockets(rows: list[dict[str, Any]], min_evaluated: int) -> list[dict[str, Any]]:
    pockets = _summarize_by(rows, "symbol_interval_strategy_regime")
    candidates = [
        item
        for item in pockets
        if int(item.get("evaluated", 0) or 0) >= min_evaluated
        and float(item.get("avg_realized_pct", 0.0) or 0.0) > 0
    ]
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("avg_realized_pct", 0.0) or 0.0),
            float(item.get("win_rate_pct", 0.0) or 0.0),
            int(item.get("evaluated", 0) or 0),
        ),
        reverse=True,
    )[:20]


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
    realized = [float(row.get("realized_pct", 0.0) or 0.0) for row in evaluated]
    wins = [value for value in realized if value > 0]
    losses = [value for value in realized if value < 0]
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
