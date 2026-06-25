from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig
from trading_agent.data import fetch_market_klines
from trading_agent.edge_filter import recent_edge_events
from trading_agent.models import Candle
from trading_agent.trade_outcome import horizon_for_interval, long_outcome


PAPER_QUOTE_AMOUNT = 10.0
PAPER_ALLOW_ACTIONS = {"ALLOW", "WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR"}
ENABLED_PAPER_LANES = {"EXPLORATION_LONG_PAPER", "RECOVERY_PAPER", "EDGE_ALLOW_PAPER"}
EXPLORATION_LONG_POCKETS = {
    "ASTERUSDT_4h",
    "NEARUSDT_1h",
}
LANE_MAX_RECENT_LOSSES = 3
LANE_MAX_DRAWDOWN_PCT = -3.0


def update_paper_simulator(
    config: AgentConfig,
    exchange: str,
    limit: int = 1000,
    source_hours: int = 6,
    max_new: int = 12,
    min_rejected_score: int = 15,
) -> dict[str, Any]:
    path = _paper_path(config)
    state = _load_paper_state(path)
    now = datetime.now(timezone.utc)
    if not state.get("started_at"):
        state["started_at"] = now.isoformat()

    opened = _open_from_edge_events(
        config,
        state,
        now=now,
        source_hours=source_hours,
        max_new=max_new,
        min_rejected_score=min_rejected_score,
    )

    candles_by_key: dict[str, list[Candle]] = {}
    closed: list[dict[str, Any]] = []
    for trade in state["trades"]:
        if trade.get("status") != "open":
            continue
        key = f"{trade['symbol']}_{trade['interval']}"
        if key not in candles_by_key:
            candles_by_key[key] = fetch_market_klines(
                exchange,
                trade["symbol"],
                trade["interval"],
                limit,
                config.bybit_market_testnet,
            )
        _mark_open_trade(trade, candles_by_key[key])
        result = _evaluate_open_trade(config, trade, candles_by_key[key])
        if result:
            trade.update(result)
            closed.append(trade)

    state["updated_at"] = now.isoformat()
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    report = {
        "generated_at": state["updated_at"],
        "mode": "paper_simulator_monitor_only",
        "execution_enabled": False,
        "quote_amount": PAPER_QUOTE_AMOUNT,
        "source_hours": source_hours,
        "saved_path": str(path),
        "opened_now": opened,
        "closed_now": closed,
        "summary": _summarize(state["trades"]),
        "by_lane": _summarize_by_lane(state["trades"]),
        "lane_guards": paper_lane_guards(state["trades"]),
        "open_trades": [trade for trade in state["trades"] if trade.get("status") == "open"],
        "recent_closed": [trade for trade in state["trades"] if trade.get("status") == "closed"][-30:],
    }
    report_path = Path(config.edge_events_path).with_name("paper_simulator_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _open_from_edge_events(
    config: AgentConfig,
    state: dict[str, Any],
    *,
    now: datetime,
    source_hours: int,
    max_new: int,
    min_rejected_score: int,
) -> list[dict[str, Any]]:
    cutoff = now - timedelta(hours=max(1, source_hours))
    started_at = _parse_time(state.get("started_at"))
    if started_at is not None and started_at > cutoff:
        cutoff = started_at
    opened: list[dict[str, Any]] = []
    events = recent_edge_events(config, limit=5000)
    for event in events:
        if len(opened) >= max_new:
            break
        opened_at = _parse_time(event.get("timestamp"))
        if opened_at is None or opened_at < cutoff:
            continue
        lane = _paper_lane(event, min_rejected_score=min_rejected_score)
        if lane is None:
            continue
        lane_guard = _lane_guard(state["trades"], lane)
        if lane_guard["blocked"]:
            continue
        symbol = str(event.get("symbol", "")).upper()
        interval = str(event.get("interval", "1h"))
        strategy = str(event.get("strategy", "UNKNOWN"))
        price = _float_or_none(event.get("price"))
        if not symbol or price is None or price <= 0:
            continue
        event_id = _event_id(event, lane)
        if _has_trade_id(state, event_id):
            continue
        if _has_recent_lane_trade(state, symbol, interval, strategy, lane, opened_at):
            continue
        trade = {
            "id": event_id,
            "status": "open",
            "opened_at": opened_at.isoformat(),
            "symbol": symbol,
            "interval": interval,
            "strategy": strategy,
            "lane": lane,
            "source": "edge_filter_event",
            "entry_price": price,
            "quote_amount": PAPER_QUOTE_AMOUNT,
            "signal_action": event.get("signal_action"),
            "final_action": event.get("final_action"),
            "confidence": event.get("confidence"),
            "edge_action": (event.get("edge_filter") or {}).get("monitor_action"),
            "score": (event.get("edge_filter") or {}).get("score"),
            "estimated_ev_pct": (event.get("edge_filter") or {}).get("estimated_ev_pct"),
            "regime": (event.get("edge_filter") or {}).get("regime"),
        }
        state["trades"].append(trade)
        opened.append(trade)
    return opened


def _paper_lane(event: dict[str, Any], *, min_rejected_score: int) -> str | None:
    signal_action = str(event.get("signal_action", "")).upper()
    edge = event.get("edge_filter") or {}
    action = str(edge.get("monitor_action", "OBSERVE")).upper()
    score = int(_float_or_none(edge.get("score")) or 0)
    ev = _float_or_none(edge.get("estimated_ev_pct"))
    if signal_action == "NO TRADE" and action == "OBSERVE" and _is_exploration_long_candidate(event, score, ev):
        return _enabled_lane("EXPLORATION_LONG_PAPER")
    if signal_action != "BUY":
        return None
    if action in PAPER_ALLOW_ACTIONS:
        return _enabled_lane("EDGE_ALLOW_PAPER")
    if action == "RECOVERY_MONITOR":
        return _enabled_lane("RECOVERY_PAPER")
    if action in {"BLOCK", "OBSERVE"} and score >= min_rejected_score:
        return _enabled_lane("REJECTED_BUY_PAPER")
    return None


def _enabled_lane(lane: str) -> str | None:
    return lane if lane in ENABLED_PAPER_LANES else None


def _is_exploration_long_candidate(event: dict[str, Any], score: int, ev: float | None) -> bool:
    key = f"{str(event.get('symbol', '')).upper()}_{str(event.get('interval', '1h'))}"
    if key not in EXPLORATION_LONG_POCKETS:
        return False
    if score < 30:
        return False
    if ev is not None and ev <= -0.15:
        return False
    return True


def _lane_guard(trades: list[dict[str, Any]], lane: str) -> dict[str, Any]:
    lane_trades = [trade for trade in trades if trade.get("lane") == lane]
    closed = [trade for trade in lane_trades if trade.get("status") == "closed"]
    realized = [float(trade.get("realized_pct", 0.0) or 0.0) for trade in closed]
    recent_losses = 0
    for value in reversed(realized):
        if value < 0:
            recent_losses += 1
        else:
            break
    drawdown = _max_drawdown(realized)
    blocked = recent_losses >= LANE_MAX_RECENT_LOSSES or drawdown <= LANE_MAX_DRAWDOWN_PCT
    return {
        "lane": lane,
        "blocked": blocked,
        "recent_losses": recent_losses,
        "max_drawdown_pct": drawdown,
        "reason": "lane risk brake" if blocked else "ok",
    }


def paper_lane_guards(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lanes = sorted({str(trade.get("lane", "UNKNOWN")) for trade in trades} | ENABLED_PAPER_LANES)
    return [_lane_guard(trades, lane) for lane in lanes]


def _evaluate_open_trade(config: AgentConfig, trade: dict[str, Any], candles: list[Candle]) -> dict[str, Any] | None:
    opened_at = _parse_time(trade.get("opened_at"))
    entry = _float_or_none(trade.get("entry_price"))
    if opened_at is None or entry is None:
        return {"status": "closed", "closed_at": datetime.now(timezone.utc).isoformat(), "outcome": "invalid", "realized_pct": 0.0, "realized_quote": 0.0}
    horizon = horizon_for_interval(str(trade.get("interval", "1h")))
    future = [candle for candle in candles if candle.close_time > opened_at]
    if future:
        window = future[: min(horizon, len(future))]
        partial = long_outcome(entry, window, config.take_profit_pct, config.stop_loss_pct)
        if partial.get("outcome") in {"take_profit", "stop_loss", "ambiguous"}:
            return _closed_payload(partial, len(window), trade)
    if len(future) < horizon:
        trade["future_candles"] = len(future)
        trade["required_candles"] = horizon
        return None
    outcome = long_outcome(entry, future[:horizon], config.take_profit_pct, config.stop_loss_pct)
    return _closed_payload(outcome, horizon, trade)


def _closed_payload(outcome: dict[str, Any], future_candles: int, trade: dict[str, Any]) -> dict[str, Any]:
    realized_pct = float(outcome.get("realized_pct", 0.0) or 0.0)
    quote = float(trade.get("quote_amount", PAPER_QUOTE_AMOUNT) or PAPER_QUOTE_AMOUNT)
    return {
        "status": "closed",
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "future_candles": future_candles,
        **outcome,
        "realized_quote": round(quote * realized_pct / 100, 6),
    }


def _mark_open_trade(trade: dict[str, Any], candles: list[Candle]) -> None:
    entry = _float_or_none(trade.get("entry_price"))
    if entry is None or entry <= 0 or not candles:
        return
    current = candles[-1].close
    unrealized_pct = ((current - entry) / entry) * 100
    quote = float(trade.get("quote_amount", PAPER_QUOTE_AMOUNT) or PAPER_QUOTE_AMOUNT)
    trade["current_price"] = current
    trade["unrealized_pct"] = round(unrealized_pct, 4)
    trade["unrealized_quote"] = round(quote * unrealized_pct / 100, 6)
    trade["marked_at"] = datetime.now(timezone.utc).isoformat()


def _summarize(trades: list[dict[str, Any]]) -> dict[str, Any]:
    open_trades = [trade for trade in trades if trade.get("status") == "open"]
    closed = [trade for trade in trades if trade.get("status") == "closed"]
    realized = [float(trade.get("realized_pct", 0.0) or 0.0) for trade in closed]
    unrealized = [float(trade.get("unrealized_pct", 0.0) or 0.0) for trade in open_trades]
    realized_quote = [float(trade.get("realized_quote", 0.0) or 0.0) for trade in closed]
    unrealized_quote = [float(trade.get("unrealized_quote", 0.0) or 0.0) for trade in open_trades]
    wins = [value for value in realized if value > 0]
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
        "realized_quote": round(sum(realized_quote), 6),
        "unrealized_quote": round(sum(unrealized_quote), 6),
        "max_drawdown_pct": _max_drawdown(realized),
        "outcomes": dict(sorted(outcomes.items())),
    }


def _summarize_by_lane(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lanes = sorted({str(trade.get("lane", "UNKNOWN")) for trade in trades})
    return [{"lane": lane, **_summarize([trade for trade in trades if str(trade.get("lane", "UNKNOWN")) == lane])} for lane in lanes]


def _max_drawdown(realized: list[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in realized:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = min(drawdown, cumulative - peak)
    return round(drawdown, 4)


def _event_id(event: dict[str, Any], lane: str) -> str:
    return "|".join(
        [
            lane,
            str(event.get("timestamp", "")),
            str(event.get("symbol", "")).upper(),
            str(event.get("interval", "")),
            str(event.get("strategy", "")),
        ]
    )


def _has_trade_id(state: dict[str, Any], trade_id: str) -> bool:
    return any(trade.get("id") == trade_id for trade in state.get("trades", []))


def _has_recent_lane_trade(
    state: dict[str, Any],
    symbol: str,
    interval: str,
    strategy: str,
    lane: str,
    opened_at: datetime,
) -> bool:
    cooldown = timedelta(hours=12 if interval == "4h" else 6)
    for trade in state.get("trades", []):
        if (
            trade.get("symbol") != symbol
            or trade.get("interval") != interval
            or trade.get("strategy") != strategy
            or trade.get("lane") != lane
        ):
            continue
        trade_time = _parse_time(trade.get("opened_at"))
        if trade.get("status") == "open":
            return True
        if trade_time is not None and abs(opened_at - trade_time) < cooldown:
            return True
    return False


def _paper_path(config: AgentConfig) -> Path:
    return Path(config.edge_events_path).with_name("paper_simulator_trades.json")


def _load_paper_state(path: Path) -> dict[str, Any]:
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
    return {"trades": trades, "started_at": raw.get("started_at"), "updated_at": raw.get("updated_at")}


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
