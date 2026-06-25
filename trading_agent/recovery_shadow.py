from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig
from trading_agent.data import fetch_market_klines
from trading_agent.models import Candle
from trading_agent.trade_outcome import horizon_for_interval, long_outcome


def update_recovery_shadow_trades(
    config: AgentConfig,
    exchange: str,
    limit: int = 1000,
    open_from_watch: bool = True,
) -> dict[str, Any]:
    path = _shadow_path(config)
    state = _load_shadow_state(path)
    opened: list[dict[str, Any]] = []
    if open_from_watch:
        opened = _open_ready_watch_trades(config, state)

    candles_by_key: dict[str, list[Candle]] = {}
    closed: list[dict[str, Any]] = []
    for trade in state["trades"]:
        if trade.get("status") != "open":
            continue
        key = f"{trade['symbol']}_{trade['interval']}"
        if key not in candles_by_key:
            candles_by_key[key] = fetch_market_klines(exchange, trade["symbol"], trade["interval"], limit, config.bybit_market_testnet)
        _mark_open_trade(config, trade, candles_by_key[key])
        result = _evaluate_open_trade(config, trade, candles_by_key[key])
        if result:
            trade.update(result)
            closed.append(trade)

    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    report = {
        "generated_at": state["updated_at"],
        "saved_path": str(path),
        "opened_now": opened,
        "closed_now": closed,
        "summary": _summarize(state["trades"]),
        "promotion": evaluate_recovery_promotion(state["trades"]),
        "open_trades": [trade for trade in state["trades"] if trade.get("status") == "open"],
        "recent_closed": [trade for trade in state["trades"] if trade.get("status") == "closed"][-20:],
    }
    report_path = Path(config.edge_events_path).with_name("recovery_shadow_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def evaluate_recovery_promotion(trades: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [trade for trade in trades if trade.get("status") == "closed"]
    recent = closed[-5:]
    realized = [float(trade.get("realized_pct", 0.0) or 0.0) for trade in closed]
    recent_realized = [float(trade.get("realized_pct", 0.0) or 0.0) for trade in recent]
    wins = [value for value in realized if value > 0]
    avg = round(sum(realized) / len(realized), 4) if realized else 0.0
    win_rate = round(len(wins) / len(closed) * 100, 2) if closed else 0.0
    last_three_losses = len(recent_realized) >= 3 and all(value < 0 for value in recent_realized[-3:])
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in realized:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)

    criteria = {
        "min_closed_trades": {"required": 5, "actual": len(closed), "passed": len(closed) >= 5},
        "min_win_rate_pct": {"required": 55.0, "actual": win_rate, "passed": win_rate >= 55.0},
        "min_avg_realized_pct": {"required": 0.2, "actual": avg, "passed": avg >= 0.2},
        "no_three_recent_losses": {"required": True, "actual": not last_three_losses, "passed": not last_three_losses},
        "max_drawdown_pct": {"required": -3.0, "actual": round(max_drawdown, 4), "passed": max_drawdown >= -3.0},
    }
    eligible = all(item["passed"] for item in criteria.values())
    return {
        "status": "ELIGIBLE_FOR_PAPER_ALLOW_SMALL" if eligible else "NOT_ELIGIBLE",
        "execution_enabled": False,
        "reason": "Promotion criteria met; manual approval still required." if eligible else _promotion_block_reason(criteria),
        "closed_trades": len(closed),
        "win_rate_pct": win_rate,
        "avg_realized_pct": avg,
        "max_drawdown_pct": round(max_drawdown, 4),
        "criteria": criteria,
    }


def _open_ready_watch_trades(config: AgentConfig, state: dict[str, Any]) -> list[dict[str, Any]]:
    watch_path = Path(config.edge_events_path).with_name("recovery_watch_report.json")
    if not watch_path.exists():
        return []
    try:
        watch = json.loads(watch_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    opened: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for row in watch.get("ready_now", []) if isinstance(watch, dict) else []:
        symbol = str(row.get("symbol", "")).upper()
        interval = str(row.get("interval", "1h"))
        strategy = str(row.get("strategy", "UNKNOWN"))
        if not symbol or _has_open_trade(state, symbol, interval, strategy):
            continue
        trade = {
            "id": f"{symbol}_{interval}_{strategy}_{now}",
            "status": "open",
            "opened_at": now,
            "symbol": symbol,
            "interval": interval,
            "strategy": strategy,
            "entry_price": float(row.get("price") or 0.0),
            "source_key": row.get("key"),
            "source": "recovery_watch",
            "audit": row.get("audit"),
            "rsi": row.get("rsi"),
            "support": row.get("support"),
            "resistance": row.get("resistance"),
            "sideways_probability": row.get("sideways_probability"),
            "edge_action": row.get("edge_action"),
        }
        if trade["entry_price"] <= 0:
            continue
        state["trades"].append(trade)
        opened.append(trade)
    return opened


def _evaluate_open_trade(config: AgentConfig, trade: dict[str, Any], candles: list[Candle]) -> dict[str, Any] | None:
    opened_at = _parse_time(trade.get("opened_at"))
    entry = _float_or_none(trade.get("entry_price"))
    if opened_at is None or entry is None:
        return {"status": "closed", "closed_at": datetime.now(timezone.utc).isoformat(), "outcome": "invalid", "realized_pct": 0.0}
    horizon = horizon_for_interval(str(trade.get("interval", "1h")))
    future = [candle for candle in candles if candle.close_time > opened_at]
    if future:
        partial_outcome = long_outcome(entry, future[: min(horizon, len(future))], config.take_profit_pct, config.stop_loss_pct)
        if partial_outcome.get("outcome") in {"take_profit", "stop_loss", "ambiguous"}:
            return {
                "status": "closed",
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "future_candles": len(future[: min(horizon, len(future))]),
                **partial_outcome,
            }
    if len(future) < horizon:
        trade["future_candles"] = len(future)
        trade["required_candles"] = horizon
        return None
    outcome = long_outcome(entry, future[:horizon], config.take_profit_pct, config.stop_loss_pct)
    return {
        "status": "closed",
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "future_candles": horizon,
        **outcome,
    }


def _mark_open_trade(config: AgentConfig, trade: dict[str, Any], candles: list[Candle]) -> None:
    entry = _float_or_none(trade.get("entry_price"))
    if entry is None or entry <= 0 or not candles:
        return
    current = candles[-1].close
    take_profit = entry * (1 + config.take_profit_pct / 100)
    stop_loss = entry * (1 - config.stop_loss_pct / 100)
    trade["current_price"] = current
    trade["unrealized_pct"] = round(((current - entry) / entry) * 100, 4)
    trade["take_profit_price"] = round(take_profit, 8)
    trade["stop_loss_price"] = round(stop_loss, 8)
    trade["distance_to_tp_pct"] = round(((take_profit - current) / current) * 100, 4)
    trade["distance_to_sl_pct"] = round(((current - stop_loss) / current) * 100, 4)
    trade["marked_at"] = datetime.now(timezone.utc).isoformat()


def _summarize(trades: list[dict[str, Any]]) -> dict[str, Any]:
    open_trades = [trade for trade in trades if trade.get("status") == "open"]
    closed = [trade for trade in trades if trade.get("status") == "closed"]
    realized = [float(trade.get("realized_pct", 0.0) or 0.0) for trade in closed]
    unrealized = [float(trade.get("unrealized_pct", 0.0) or 0.0) for trade in open_trades]
    wins = [value for value in realized if value > 0]
    losses = [value for value in realized if value < 0]
    outcomes: dict[str, int] = {}
    for trade in closed:
        outcome = str(trade.get("outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {
        "total": len(trades),
        "open": len(open_trades),
        "closed": len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 2) if closed else 0.0,
        "avg_realized_pct": round(sum(realized) / len(realized), 4) if realized else 0.0,
        "avg_unrealized_pct": round(sum(unrealized) / len(unrealized), 4) if unrealized else 0.0,
        "outcomes": dict(sorted(outcomes.items())),
    }


def _promotion_block_reason(criteria: dict[str, dict[str, Any]]) -> str:
    failed = [name for name, item in criteria.items() if not item.get("passed")]
    if not failed:
        return "Promotion criteria met; manual approval still required."
    return "Waiting for: " + ", ".join(failed)


def _has_open_trade(state: dict[str, Any], symbol: str, interval: str, strategy: str) -> bool:
    for trade in state.get("trades", []):
        if (
            trade.get("status") == "open"
            and trade.get("symbol") == symbol
            and trade.get("interval") == interval
            and trade.get("strategy") == strategy
        ):
            return True
    return False


def _shadow_path(config: AgentConfig) -> Path:
    return Path(config.edge_events_path).with_name("recovery_shadow_trades.json")


def _load_shadow_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = {}
    else:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    trades = raw.get("trades")
    if not isinstance(trades, list):
        trades = []
    return {"trades": trades, "updated_at": raw.get("updated_at")}


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
