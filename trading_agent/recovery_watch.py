from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig
from trading_agent.data import fetch_market_klines
from trading_agent.edge_filter import evaluate_edge_filter
from trading_agent.market import analyze_market
from trading_agent.models import Action, StrategyName
from trading_agent.risk import RiskContext, apply_risk_rules
from trading_agent.strategies import run_strategy


def build_recovery_watch_report(
    config: AgentConfig,
    exchange: str,
    limit: int = 240,
    max_pockets: int = 12,
) -> dict[str, Any]:
    audit = _load_audit(config)
    pockets = audit.get("best_pockets", [])[:max(1, max_pockets)]
    rows: list[dict[str, Any]] = []
    for pocket in pockets:
        parsed = _parse_pocket_key(str(pocket.get("key", "")))
        if not parsed:
            continue
        symbol, interval, strategy_name, expected_regime = parsed
        try:
            rows.append(_evaluate_pocket(config, exchange, symbol, interval, strategy_name, expected_regime, pocket, limit))
        except Exception as exc:
            rows.append(
                {
                    "key": pocket.get("key"),
                    "status": "error",
                    "error": str(exc),
                    "audit": _audit_summary(pocket),
                }
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exchange": exchange,
        "audit_generated_at": audit.get("generated_at"),
        "pockets_checked": len(rows),
        "ready_now": [row for row in rows if row.get("status") == "ready_now"],
        "near_miss": [row for row in rows if row.get("status") == "near_miss"],
        "blocked_now": [row for row in rows if row.get("status") == "blocked_now"],
        "not_ready": [row for row in rows if row.get("status") == "not_ready"],
        "errors": [row for row in rows if row.get("status") == "error"],
        "rows": rows,
    }
    path = Path(config.edge_events_path).with_name("recovery_watch_report.json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    report["saved_path"] = str(path)
    return report


def _evaluate_pocket(
    config: AgentConfig,
    exchange: str,
    symbol: str,
    interval: str,
    strategy_name: StrategyName,
    expected_regime: str,
    pocket: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    candles = fetch_market_klines(exchange, symbol, interval, limit, config.bybit_market_testnet)
    state = analyze_market(candles)
    signal = run_strategy(strategy_name, state)
    risk = apply_risk_rules(signal, state, config, RiskContext())
    edge = evaluate_edge_filter(symbol=symbol, interval=interval, state=state, signal=signal, risk=risk, config=config)
    actual_regime = edge.regime
    blockers = _blockers(expected_regime, actual_regime, signal, state, edge)
    status = _status(edge.monitor_action, signal.action, blockers)
    return {
        "key": f"{symbol}_{interval}|{strategy_name.value}|{expected_regime}",
        "status": status,
        "symbol": symbol,
        "interval": interval,
        "strategy": strategy_name.value,
        "expected_regime": expected_regime,
        "actual_regime": actual_regime,
        "signal_action": signal.action.value,
        "signal_confidence": signal.confidence,
        "risk_action": risk.action.value,
        "edge_action": edge.monitor_action,
        "score": edge.score,
        "estimated_ev_pct": edge.estimated_ev_pct,
        "price": state.current_price,
        "rsi": state.rsi,
        "support": state.support,
        "resistance": state.resistance,
        "sideways_probability": state.sideways_probability,
        "bullish_probability": state.bullish_probability,
        "bearish_probability": state.bearish_probability,
        "liquidity_score": state.liquidity_score,
        "fake_breakout_probability": state.fake_breakout_probability,
        "blockers": blockers,
        "signal_reasoning": signal.reasoning,
        "risk_reason": risk.reason,
        "edge_reasons": edge.reasons,
        "audit": _audit_summary(pocket),
    }


def _status(edge_action: str, signal_action: Action, blockers: list[str]) -> str:
    if edge_action == "RECOVERY_MONITOR":
        return "ready_now"
    if signal_action == Action.BUY and len(blockers) <= 1:
        return "near_miss"
    if signal_action == Action.BUY:
        return "blocked_now"
    return "not_ready"


def _blockers(expected_regime: str, actual_regime: str, signal, state, edge) -> list[str]:
    blockers: list[str] = []
    if actual_regime != expected_regime:
        blockers.append(f"regime mismatch: expected {expected_regime}, got {actual_regime}")
    if signal.action != Action.BUY:
        blockers.append(f"strategy signal is {signal.action.value}, not BUY")
    if expected_regime == "sideways":
        if state.rsi >= 32:
            blockers.append(f"RSI not stretched enough for mean reversion: {state.rsi:.2f} >= 32")
        if state.current_price > state.support * 1.008:
            distance = ((state.current_price / state.support) - 1) * 100 if state.support else 0.0
            blockers.append(f"price not close enough to support: {distance:.2f}% above support")
    if edge.monitor_action not in {"RECOVERY_MONITOR", "WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR"}:
        blockers.append(f"edge action is {edge.monitor_action}")
    return blockers


def _load_audit(config: AgentConfig) -> dict[str, Any]:
    path = Path(config.edge_events_path).with_name("blocked_winners_audit.json")
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _parse_pocket_key(key: str) -> tuple[str, str, StrategyName, str] | None:
    parts = key.split("|")
    if len(parts) != 3 or "_" not in parts[0]:
        return None
    symbol, interval = parts[0].rsplit("_", 1)
    strategy = _strategy_from_value(parts[1])
    if strategy is None:
        return None
    return symbol.upper(), interval, strategy, parts[2]


def _strategy_from_value(value: str) -> StrategyName | None:
    for strategy in StrategyName:
        if strategy.value == value:
            return strategy
    return None


def _audit_summary(pocket: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluated": pocket.get("evaluated"),
        "pending": pocket.get("pending"),
        "win_rate_pct": pocket.get("win_rate_pct"),
        "avg_realized_pct": pocket.get("avg_realized_pct"),
        "outcomes": pocket.get("outcomes"),
    }
