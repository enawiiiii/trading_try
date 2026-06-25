from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig
from trading_agent.edge_filter import evaluate_edge_filter
from trading_agent.historical_store import load_candles
from trading_agent.learning import load_learning_state
from trading_agent.market import analyze_market
from trading_agent.models import Action, Candle
from trading_agent.paper_simulator import _paper_lane
from trading_agent.risk import RiskContext, apply_risk_rules
from trading_agent.strategies import select_signal
from trading_agent.trade_outcome import horizon_for_interval, long_outcome


def backtest_edge_filter(
    config: AgentConfig,
    provider: str,
    symbols: list[str],
    intervals: list[str],
    lookback: int = 80,
    max_events_per_model: int = 500,
    stride: int = 6,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "provider": provider,
        "symbols": symbols,
        "intervals": intervals,
        "lookback": lookback,
        "stride": stride,
        "models": {},
        "summary": _empty_summary(),
    }

    for symbol in symbols:
        for interval in intervals:
            key = f"{symbol.upper()}_{interval}"
            try:
                item = _backtest_one(config, provider, symbol, interval, lookback, max_events_per_model, stride)
                report["models"][key] = item
                _merge_summary(report["summary"], item["summary"])
            except Exception as exc:
                report["models"][key] = {"error": str(exc), "summary": _empty_summary()}
                report["summary"]["errors"] += 1

    _finalize_summary(report["summary"])
    report["pocket_stats_path"] = str(save_pocket_stats(config, report))
    return report


def save_pocket_stats(config: AgentConfig, report: dict[str, Any]) -> Path:
    output: dict[str, Any] = {}
    for key, item in (report.get("models") or {}).items():
        if not isinstance(item, dict) or "summary" not in item:
            continue
        summary = item["summary"]
        buy_events = int(summary.get("buy_events", 0) or 0)
        output[key] = {
            "buy_events": buy_events,
            "avg_realized_pct": float(summary.get("all_buy_avg_realized_pct", 0.0) or 0.0),
            "win_rate_pct": float(summary.get("all_buy_win_rate_pct", 0.0) or 0.0),
            "source": "edge_backtest",
        }
    path = Path(config.edge_events_path).with_name("edge_pocket_stats.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def _backtest_one(
    config: AgentConfig,
    provider: str,
    symbol: str,
    interval: str,
    lookback: int,
    max_events: int,
    stride: int,
) -> dict[str, Any]:
    candles = load_candles(config, provider, symbol, interval)
    horizon = horizon_for_interval(interval)
    if len(candles) < lookback + horizon + 1:
        raise ValueError(f"Not enough stored candles for {symbol} {interval}.")

    learning_state = load_learning_state(config)
    rows: list[dict[str, Any]] = []
    max_start = len(candles) - horizon
    start = max(lookback, max_start - max_events * max(1, stride))

    for end in range(start, max_start, max(1, stride)):
        window = candles[end - lookback : end]
        future = candles[end : end + horizon]
        state = analyze_market(window)
        signal = select_signal(state, learning_state.strategy_weights, has_position=False)
        strategy = signal.strategy
        risk = apply_risk_rules(signal, state, config, RiskContext(has_position=False))
        edge = evaluate_edge_filter(
            symbol=symbol,
            interval=interval,
            state=state,
            signal=signal,
            risk=risk,
            config=config,
        )
        row = {
            "timestamp": candles[end - 1].close_time.isoformat(),
            "action": risk.action.value,
            "signal_action": signal.action.value,
            "strategy": strategy.value,
            "edge_action": edge.monitor_action,
            "score": edge.score,
            "ev_estimate_pct": edge.estimated_ev_pct,
            "macro_f1": edge.ml_macro_f1,
            "regime": edge.regime,
        }
        paper_lane = _paper_lane(
            {
                "signal_action": signal.action.value,
                "edge_filter": {
                    "monitor_action": edge.monitor_action,
                    "score": edge.score,
                    "estimated_ev_pct": edge.estimated_ev_pct,
                },
            },
            min_rejected_score=15,
        )
        if paper_lane:
            row["paper_lane"] = paper_lane
            row.update(long_outcome(window[-1].close, future, config.take_profit_pct, config.stop_loss_pct))
        if risk.action == Action.BUY:
            outcome = long_outcome(window[-1].close, future, config.take_profit_pct, config.stop_loss_pct)
            row.update(outcome)
        rows.append(row)

    summary = _summarize_rows(rows)
    return {
        "candles": len(candles),
        "events": len(rows),
        "summary": summary,
        "recent_buy_events": [row for row in rows if row["action"] == "BUY"][-12:],
        "recent_paper_events": [row for row in rows if row.get("paper_lane")][-12:],
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _empty_summary()
    edge_counts = Counter(row["edge_action"] for row in rows)
    final_counts = Counter(row["action"] for row in rows)
    summary["events"] = len(rows)
    summary["allow"] = edge_counts.get("ALLOW", 0)
    summary["weak_allow"] = edge_counts.get("WEAK_ALLOW_MONITOR", 0)
    summary["strong_allow"] = edge_counts.get("STRONG_ALLOW_MONITOR", 0)
    summary["recovery_monitor"] = edge_counts.get("RECOVERY_MONITOR", 0)
    summary["block"] = edge_counts.get("BLOCK", 0)
    summary["observe"] = edge_counts.get("OBSERVE", 0)
    summary["buy"] = final_counts.get("BUY", 0)
    summary["sell"] = final_counts.get("SELL", 0)
    summary["no_trade"] = final_counts.get("NO TRADE", 0)

    buy_rows = [row for row in rows if row["action"] == "BUY" and "realized_pct" in row]
    blocked_buy_rows = [row for row in buy_rows if row["edge_action"] == "BLOCK"]
    allowed_actions = {"ALLOW", "WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR"}
    allowed_buy_rows = [row for row in buy_rows if row["edge_action"] in allowed_actions]
    exploration_rows = [row for row in rows if row.get("paper_lane") == "EXPLORATION_LONG_PAPER" and "realized_pct" in row]
    edge_allow_paper_rows = [row for row in rows if row.get("paper_lane") == "EDGE_ALLOW_PAPER" and "realized_pct" in row]
    recovery_paper_rows = [row for row in rows if row.get("paper_lane") == "RECOVERY_PAPER" and "realized_pct" in row]
    summary["buy_events"] = len(buy_rows)
    summary["blocked_buy_events"] = len(blocked_buy_rows)
    summary["allowed_buy_events"] = len(allowed_buy_rows)
    summary["all_buy_avg_realized_pct"] = _avg(row["realized_pct"] for row in buy_rows)
    summary["all_buy_win_rate_pct"] = _win_rate(buy_rows)
    summary["blocked_buy_avg_realized_pct"] = _avg(row["realized_pct"] for row in blocked_buy_rows)
    summary["allowed_buy_avg_realized_pct"] = _avg(row["realized_pct"] for row in allowed_buy_rows)
    summary["blocked_buy_win_rate_pct"] = _win_rate(blocked_buy_rows)
    summary["allowed_buy_win_rate_pct"] = _win_rate(allowed_buy_rows)
    summary["exploration_long_events"] = len(exploration_rows)
    summary["exploration_long_avg_realized_pct"] = _avg(row["realized_pct"] for row in exploration_rows)
    summary["exploration_long_win_rate_pct"] = _win_rate(exploration_rows)
    summary["edge_allow_paper_events"] = len(edge_allow_paper_rows)
    summary["recovery_paper_events"] = len(recovery_paper_rows)
    summary["avg_score"] = _avg(row["score"] for row in rows)
    return summary


def _empty_summary() -> dict[str, Any]:
    return {
        "events": 0,
        "allow": 0,
        "weak_allow": 0,
        "strong_allow": 0,
        "recovery_monitor": 0,
        "block": 0,
        "observe": 0,
        "buy": 0,
        "sell": 0,
        "no_trade": 0,
        "buy_events": 0,
        "blocked_buy_events": 0,
        "allowed_buy_events": 0,
        "all_buy_avg_realized_pct": 0.0,
        "all_buy_win_rate_pct": 0.0,
        "blocked_buy_avg_realized_pct": 0.0,
        "allowed_buy_avg_realized_pct": 0.0,
        "blocked_buy_win_rate_pct": 0.0,
        "allowed_buy_win_rate_pct": 0.0,
        "exploration_long_events": 0,
        "exploration_long_avg_realized_pct": 0.0,
        "exploration_long_win_rate_pct": 0.0,
        "edge_allow_paper_events": 0,
        "recovery_paper_events": 0,
        "avg_score": 0.0,
        "errors": 0,
    }


def _merge_summary(total: dict[str, Any], item: dict[str, Any]) -> None:
    for key in (
        "events",
        "allow",
        "weak_allow",
        "strong_allow",
        "recovery_monitor",
        "block",
        "observe",
        "buy",
        "sell",
        "no_trade",
        "buy_events",
        "blocked_buy_events",
        "allowed_buy_events",
        "exploration_long_events",
        "edge_allow_paper_events",
        "recovery_paper_events",
    ):
        total[key] += int(item.get(key, 0) or 0)
    total.setdefault("_buy_realized_sum", 0.0)
    total.setdefault("_blocked_realized_sum", 0.0)
    total.setdefault("_allowed_realized_sum", 0.0)
    total.setdefault("_blocked_wins", 0)
    total.setdefault("_buy_wins", 0)
    total.setdefault("_allowed_wins", 0)
    total.setdefault("_score_sum", 0.0)
    total.setdefault("_exploration_realized_sum", 0.0)
    total.setdefault("_exploration_wins", 0)
    total["_buy_realized_sum"] += float(item.get("all_buy_avg_realized_pct", 0.0) or 0.0) * int(item.get("buy_events", 0) or 0)
    total["_buy_wins"] += round(float(item.get("all_buy_win_rate_pct", 0.0) or 0.0) * int(item.get("buy_events", 0) or 0) / 100)
    total["_blocked_realized_sum"] += float(item.get("blocked_buy_avg_realized_pct", 0.0) or 0.0) * int(item.get("blocked_buy_events", 0) or 0)
    total["_allowed_realized_sum"] += float(item.get("allowed_buy_avg_realized_pct", 0.0) or 0.0) * int(item.get("allowed_buy_events", 0) or 0)
    total["_blocked_wins"] += round(float(item.get("blocked_buy_win_rate_pct", 0.0) or 0.0) * int(item.get("blocked_buy_events", 0) or 0) / 100)
    total["_allowed_wins"] += round(float(item.get("allowed_buy_win_rate_pct", 0.0) or 0.0) * int(item.get("allowed_buy_events", 0) or 0) / 100)
    total["_score_sum"] += float(item.get("avg_score", 0.0) or 0.0) * int(item.get("events", 0) or 0)
    total["_exploration_realized_sum"] += float(item.get("exploration_long_avg_realized_pct", 0.0) or 0.0) * int(item.get("exploration_long_events", 0) or 0)
    total["_exploration_wins"] += round(float(item.get("exploration_long_win_rate_pct", 0.0) or 0.0) * int(item.get("exploration_long_events", 0) or 0) / 100)


def _finalize_summary(summary: dict[str, Any]) -> None:
    buy_sum = summary.pop("_buy_realized_sum", 0.0)
    blocked_sum = summary.pop("_blocked_realized_sum", 0.0)
    allowed_sum = summary.pop("_allowed_realized_sum", 0.0)
    summary["all_buy_avg_realized_pct"] = round(buy_sum / summary["buy_events"], 4) if summary["buy_events"] else 0.0
    summary["blocked_buy_avg_realized_pct"] = round(blocked_sum / summary["blocked_buy_events"], 4) if summary["blocked_buy_events"] else 0.0
    summary["allowed_buy_avg_realized_pct"] = round(allowed_sum / summary["allowed_buy_events"], 4) if summary["allowed_buy_events"] else 0.0
    blocked_wins = summary.pop("_blocked_wins", 0)
    allowed_wins = summary.pop("_allowed_wins", 0)
    buy_wins = summary.pop("_buy_wins", 0)
    summary["all_buy_win_rate_pct"] = round(buy_wins / summary["buy_events"] * 100, 2) if summary["buy_events"] else 0.0
    summary["blocked_buy_win_rate_pct"] = round(blocked_wins / summary["blocked_buy_events"] * 100, 2) if summary["blocked_buy_events"] else 0.0
    summary["allowed_buy_win_rate_pct"] = round(allowed_wins / summary["allowed_buy_events"] * 100, 2) if summary["allowed_buy_events"] else 0.0
    summary["avg_score"] = round(summary.pop("_score_sum", 0.0) / summary["events"], 2) if summary["events"] else 0.0
    exploration_sum = summary.pop("_exploration_realized_sum", 0.0)
    exploration_wins = summary.pop("_exploration_wins", 0)
    summary["exploration_long_avg_realized_pct"] = round(exploration_sum / summary["exploration_long_events"], 4) if summary["exploration_long_events"] else 0.0
    summary["exploration_long_win_rate_pct"] = round(exploration_wins / summary["exploration_long_events"] * 100, 2) if summary["exploration_long_events"] else 0.0


_long_outcome = long_outcome
_horizon = horizon_for_interval


def _avg(values) -> float:
    rows = [float(value) for value in values]
    return round(sum(rows) / len(rows), 4) if rows else 0.0


def _win_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    wins = sum(1 for row in rows if row.get("realized_pct", 0) > 0)
    return round(wins / len(rows) * 100, 2)
