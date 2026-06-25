from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig
from trading_agent.historical_store import load_candles
from trading_agent.learning import load_learning_state
from trading_agent.market import analyze_market
from trading_agent.models import Action
from trading_agent.strategies import run_strategy, select_strategy
from trading_agent.trade_outcome import horizon_for_interval, long_outcome


def build_signal_quality_report(
    config: AgentConfig,
    provider: str,
    symbols: list[str],
    intervals: list[str],
    lookback: int = 80,
    max_events_per_model: int = 700,
    stride: int = 6,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    learning_state = load_learning_state(config)

    for symbol in symbols:
        for interval in intervals:
            try:
                rows.extend(_collect_rows(config, provider, symbol, interval, lookback, max_events_per_model, stride, learning_state.strategy_weights))
            except Exception as exc:
                errors.append({"key": f"{symbol.upper()}_{interval}", "error": str(exc)})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "symbols": [symbol.upper() for symbol in symbols],
        "intervals": intervals,
        "lookback": lookback,
        "stride": stride,
        "rows": len(rows),
        "errors": errors,
        "summary": _summarize(rows),
        "by_symbol_interval": _summarize_by(rows, "symbol_interval"),
        "by_strategy": _summarize_by(rows, "strategy"),
        "by_regime": _summarize_by(rows, "regime"),
        "by_symbol_interval_strategy": _summarize_by(rows, "symbol_interval_strategy"),
        "by_strategy_regime": _summarize_by(rows, "strategy_regime"),
    }
    path = signal_quality_report_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    report["saved_path"] = str(path)
    return report


def signal_quality_report_path(config: AgentConfig) -> Path:
    return Path(config.edge_events_path).with_name("signal_quality_report.json")


def signal_quality_stats(config: AgentConfig, symbol: str, interval: str, strategy: str, regime: str) -> dict[str, Any] | None:
    path = signal_quality_report_path(config)
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    candidates = [
        ("by_symbol_interval_strategy", f"{symbol.upper()}_{interval}|{strategy}"),
        ("by_strategy_regime", f"{strategy}|{regime}"),
        ("by_symbol_interval", f"{symbol.upper()}_{interval}"),
    ]
    for group, key in candidates:
        for item in report.get(group, []) if isinstance(report, dict) else []:
            if item.get("key") == key and int(item.get("buy_events", 0) or 0) >= 20:
                return item if isinstance(item, dict) else None
    return None


def pocket_rule_stats(config: AgentConfig, symbol: str, interval: str, strategy: str) -> dict[str, Any] | None:
    path = signal_quality_report_path(config)
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    key = f"{symbol.upper()}_{interval}|{strategy}"
    for item in report.get("by_symbol_interval_strategy", []) if isinstance(report, dict) else []:
        if item.get("key") == key and int(item.get("buy_events", 0) or 0) >= 20:
            return item if isinstance(item, dict) else None
    return None


def pocket_rule_score(stats: dict[str, Any] | None) -> int:
    if not stats:
        return 0
    buy_events = int(stats.get("buy_events", 0) or 0)
    avg = float(stats.get("avg_realized_pct", 0.0) or 0.0)
    win_rate = float(stats.get("win_rate_pct", 0.0) or 0.0)
    if buy_events < 20:
        return 0
    if avg >= 0.2 and win_rate >= 43:
        return 16
    if avg >= 0.05 and win_rate >= 43:
        return 8
    if avg <= -0.15 or win_rate < 30:
        return -18
    if avg < 0:
        return -8
    return 0


def signal_quality_score(stats: dict[str, Any] | None) -> int:
    if not stats:
        return 0
    buy_events = int(stats.get("buy_events", 0) or 0)
    avg = float(stats.get("avg_realized_pct", 0.0) or 0.0)
    win_rate = float(stats.get("win_rate_pct", 0.0) or 0.0)
    if buy_events < 20:
        return 0
    score = 0
    if avg >= 0.25 and win_rate >= 45:
        score += 18
    elif avg >= 0.08 and win_rate >= 40:
        score += 10
    elif avg > 0 and win_rate >= 35:
        score += 5
    elif avg <= -0.15 or win_rate < 30:
        score -= 18
    elif avg < 0:
        score -= 10
    return max(-22, min(22, score))


def _collect_rows(
    config: AgentConfig,
    provider: str,
    symbol: str,
    interval: str,
    lookback: int,
    max_events: int,
    stride: int,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    candles = load_candles(config, provider, symbol, interval)
    horizon = horizon_for_interval(interval)
    if len(candles) < lookback + horizon + 1:
        raise ValueError(f"Not enough stored candles for {symbol} {interval}.")

    rows: list[dict[str, Any]] = []
    max_start = len(candles) - horizon
    start = max(lookback, max_start - max_events * max(1, stride))
    for end in range(start, max_start, max(1, stride)):
        window = candles[end - lookback : end]
        future = candles[end : end + horizon]
        state = analyze_market(window)
        strategy = select_strategy(state, weights)
        signal = run_strategy(strategy, state)
        if signal.action != Action.BUY:
            continue
        regime = _regime_from_state(state)
        outcome = long_outcome(window[-1].close, future, config.take_profit_pct, config.stop_loss_pct)
        rows.append(
            {
                "timestamp": candles[end - 1].close_time.isoformat(),
                "symbol": symbol.upper(),
                "interval": interval,
                "symbol_interval": f"{symbol.upper()}_{interval}",
                "strategy": strategy.value,
                "regime": regime,
                "symbol_interval_strategy": f"{symbol.upper()}_{interval}|{strategy.value}",
                "strategy_regime": f"{strategy.value}|{regime}",
                **outcome,
            }
        )
    return rows


def _summarize_by(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "UNKNOWN"))].append(row)
    return [
        {"key": key, **_summarize(items)}
        for key, items in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    ]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    realized = [float(row.get("realized_pct", 0.0) or 0.0) for row in rows]
    wins = [value for value in realized if value > 0]
    losses = [value for value in realized if value < 0]
    outcomes: dict[str, int] = defaultdict(int)
    for row in rows:
        outcomes[str(row.get("outcome", "unknown"))] += 1
    return {
        "buy_events": len(rows),
        "win_rate_pct": round(len(wins) / len(rows) * 100, 2) if rows else 0.0,
        "loss_rate_pct": round(len(losses) / len(rows) * 100, 2) if rows else 0.0,
        "avg_realized_pct": round(sum(realized) / len(realized), 4) if realized else 0.0,
        "best_realized_pct": round(max(realized), 4) if realized else 0.0,
        "worst_realized_pct": round(min(realized), 4) if realized else 0.0,
        "outcomes": dict(sorted(outcomes.items())),
    }


def _regime_from_state(state) -> str:
    if state.volatility == "high":
        return "high_volatility"
    if state.trend == "sideways" or state.sideways_probability >= max(state.bullish_probability, state.bearish_probability):
        return "sideways"
    if state.trend in {"bullish", "bearish"}:
        return "trending"
    return "mixed"
