from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig
from trading_agent.models import Action, MarketState
from trading_agent.risk import RiskDecision
from trading_agent.signal_quality import pocket_rule_score, pocket_rule_stats, signal_quality_score, signal_quality_stats
from trading_agent.strategies import StrategySignal


@dataclass(frozen=True)
class EdgeFilterResult:
    mode: str
    monitor_action: str
    would_allow_trade: bool
    score: int
    estimated_ev_pct: float | None
    ml_accuracy: float | None
    ml_macro_f1: float | None
    ml_sideways_accuracy: float | None
    regime: str
    reasons: list[str]
    observed_action: str
    tier: str
    pocket_score: int
    outcome_score: int
    blocked_outcome_score: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_edge_filter(
    *,
    symbol: str,
    interval: str,
    state: MarketState,
    signal: StrategySignal,
    risk: RiskDecision,
    config: AgentConfig,
) -> EdgeFilterResult:
    model_metrics = _model_metrics(config, symbol, interval)
    pocket = historical_pocket_stats(config, symbol, interval)
    live_outcome = live_outcome_stats(config, symbol, interval)
    weak_outcome = live_edge_outcome_stats(config, "WEAK_ALLOW_MONITOR", symbol, interval)
    blocked_outcome = live_edge_outcome_stats(config, "BLOCK", symbol, interval)
    global_weak_outcome = live_edge_action_stats(config, "WEAK_ALLOW_MONITOR")
    global_strong_outcome = live_edge_action_stats(config, "STRONG_ALLOW_MONITOR")
    ev = _float_or_none(model_metrics.get("long_expected_value_pct"))
    macro_f1 = _float_or_none(model_metrics.get("macro_f1"))
    accuracy = _float_or_none(model_metrics.get("accuracy"))
    sideways_accuracy = _float_or_none(
        ((model_metrics.get("by_label") or {}).get("sideways") or {}).get("accuracy")
    )
    regime = _regime(state)
    quality_stats = signal_quality_stats(config, symbol, interval, signal.strategy.value, regime)
    quality_score = signal_quality_score(quality_stats)
    pocket_rule = pocket_rule_stats(config, symbol, interval, signal.strategy.value)
    rule_score = pocket_rule_score(pocket_rule)
    recovery_audit = recovery_audit_stats(config, symbol, interval, signal.strategy.value, regime)

    reasons: list[str] = []
    score = 45
    pocket_score = 0
    outcome_score = 0
    blocked_outcome_score = 0
    tier = "OBSERVE"
    monitor_action = "OBSERVE"
    would_allow = False
    hard_block = False
    candidate_buy = signal.action == Action.BUY

    if risk.action == Action.NO_TRADE:
        reasons.append("Final decision is NO TRADE; edge filter is observing only.")
        score -= 10
    elif risk.action == Action.BUY:
        monitor_action = "ALLOW"
        would_allow = True
    else:
        reasons.append("Final decision is SELL; buy edge filter is observing only.")

    if candidate_buy:
        if ev is None:
            score -= 10
            reasons.append("No EV estimate found for this symbol/timeframe.")
        elif ev <= -0.12:
            score -= 25
            hard_block = True
            reasons.append(f"Long EV is strongly negative ({ev:.4f}%).")
        elif ev <= 0:
            score -= 8
            reasons.append(f"Long EV is slightly negative ({ev:.4f}%).")
        else:
            score += 18
            reasons.append(f"Long EV is positive ({ev:.4f}%).")

    if macro_f1 is None:
        score -= 10
        reasons.append("Macro F1 is unavailable.")
    elif macro_f1 < 36:
        score -= 15
        reasons.append(f"Macro F1 is weak ({macro_f1:.2f}%).")
    elif macro_f1 >= 42:
        score += 14
        reasons.append(f"Macro F1 is strong enough for monitoring ({macro_f1:.2f}%).")
    else:
        score += 5

    if state.volatility == "high":
        score -= 15
        reasons.append("High volatility increases execution and fake-move risk.")
    elif state.volatility == "medium":
        score += 8
    elif state.volatility == "low":
        score += 3

    if state.liquidity_score < config.min_liquidity_score + 15:
        score -= 10
        reasons.append(f"Liquidity score is thin for edge confirmation ({state.liquidity_score}).")
    elif state.liquidity_score >= 70:
        score += 8

    if state.fake_breakout_probability > config.max_fake_breakout_probability - 10:
        score -= 10
        reasons.append(f"Fake breakout risk is elevated ({state.fake_breakout_probability}%).")
    elif state.fake_breakout_probability <= 35:
        score += 6

    if state.sideways_probability >= 40 and sideways_accuracy is not None and sideways_accuracy < 25:
        score -= 10
        reasons.append(f"Sideways probability is high but sideways model accuracy is weak ({sideways_accuracy:.2f}%).")

    if risk.confidence < 60:
        score -= 10
        reasons.append(f"Decision confidence is not high enough for edge confirmation ({risk.confidence}%).")
    elif risk.confidence >= 75:
        score += 8

    if candidate_buy and pocket:
        pocket_score = _pocket_score(pocket)
        score += pocket_score
        if pocket_score > 0:
            reasons.append(
                f"Historical pocket is constructive: avg {pocket.get('avg_realized_pct', 0):.4f}% "
                f"over {pocket.get('buy_events', 0)} BUY events."
            )
        elif pocket_score < 0:
            reasons.append(
                f"Historical pocket is weak: avg {pocket.get('avg_realized_pct', 0):.4f}% "
                f"over {pocket.get('buy_events', 0)} BUY events."
            )

    if candidate_buy and live_outcome:
        outcome_score = _outcome_score(live_outcome)
        score += outcome_score
        if outcome_score > 0:
            reasons.append(
                f"Recent live outcome pocket is constructive: avg {live_outcome.get('avg_realized_pct', 0):.4f}% "
                f"over {live_outcome.get('evaluated', 0)} evaluated BUY events."
            )
        elif outcome_score < 0:
            reasons.append(
                f"Recent live outcome pocket is weak: avg {live_outcome.get('avg_realized_pct', 0):.4f}% "
                f"over {live_outcome.get('evaluated', 0)} evaluated BUY events."
            )

    if candidate_buy and blocked_outcome:
        blocked_outcome_score = _blocked_outcome_score(blocked_outcome)
        blocked_has_confirmation = _blocked_outcome_has_independent_confirmation(
            ev=ev,
            macro_f1=macro_f1,
            pocket_score=pocket_score,
            quality_score=quality_score,
            rule_score=rule_score,
        )
        if blocked_outcome_score > 0 and not blocked_has_confirmation:
            reasons.append(
                f"Recent blocked BUY outcomes were constructive but lack independent confirmation: avg "
                f"{blocked_outcome.get('avg_realized_pct', 0):.4f}% over "
                f"{blocked_outcome.get('evaluated', 0)} evaluated blocked BUY events."
            )
            blocked_outcome_score = 0
        score += blocked_outcome_score
        if blocked_outcome_score > 0:
            reasons.append(
                f"Recent blocked BUY outcomes were constructive: avg {blocked_outcome.get('avg_realized_pct', 0):.4f}% "
                f"over {blocked_outcome.get('evaluated', 0)} evaluated blocked BUY events."
            )
        elif blocked_outcome_score < 0:
            reasons.append(
                f"Recent blocked BUY outcomes were weak: avg {blocked_outcome.get('avg_realized_pct', 0):.4f}% "
                f"over {blocked_outcome.get('evaluated', 0)} evaluated blocked BUY events."
            )

    if candidate_buy and quality_score:
        score += quality_score
        if quality_score > 0:
            reasons.append(
                f"Signal quality memory is constructive: avg {quality_stats.get('avg_realized_pct', 0):.4f}% "
                f"over {quality_stats.get('buy_events', 0)} BUY signals."
            )
        else:
            reasons.append(
                f"Signal quality memory is weak: avg {quality_stats.get('avg_realized_pct', 0):.4f}% "
                f"over {quality_stats.get('buy_events', 0)} BUY signals."
            )

    if candidate_buy and rule_score:
        score += rule_score
        if rule_score > 0:
            reasons.append(
                f"Pocket rule is constructive for this exact setup: avg {pocket_rule.get('avg_realized_pct', 0):.4f}% "
                f"over {pocket_rule.get('buy_events', 0)} BUY signals."
            )
        else:
            reasons.append(
                f"Pocket rule is weak for this exact setup: avg {pocket_rule.get('avg_realized_pct', 0):.4f}% "
                f"over {pocket_rule.get('buy_events', 0)} BUY signals."
            )

    score = max(0, min(100, score))
    if candidate_buy:
        live_edge_score = max(outcome_score, blocked_outcome_score)
        negative_ev_without_live_edge = ev is not None and ev < -0.08 and live_edge_score < 20
        bad_historical_without_live_edge = pocket_score <= -14 and live_edge_score < 20
        bad_signal_quality_without_live_edge = quality_score <= -18 and live_edge_score < 20
        bad_pocket_rule_without_live_edge = rule_score <= -18 and live_edge_score < 20
        if hard_block:
            tier = "BLOCK"
            monitor_action = "BLOCK"
            would_allow = False
        elif score < 30:
            tier = "BLOCK"
            monitor_action = "BLOCK"
            would_allow = False
        elif score < 60:
            tier = "OBSERVE"
            monitor_action = "OBSERVE"
            would_allow = False
        elif negative_ev_without_live_edge:
            tier = "OBSERVE"
            monitor_action = "OBSERVE"
            would_allow = False
            reasons.append("EV is negative without a strong recent live outcome edge; downgraded to OBSERVE.")
        elif bad_historical_without_live_edge:
            tier = "OBSERVE"
            monitor_action = "OBSERVE"
            would_allow = False
            reasons.append("Historical pocket is too weak without a strong recent live outcome edge; downgraded to OBSERVE.")
        elif bad_signal_quality_without_live_edge:
            tier = "OBSERVE"
            monitor_action = "OBSERVE"
            would_allow = False
            reasons.append("Signal quality memory is too weak without a strong recent live outcome edge; downgraded to OBSERVE.")
        elif bad_pocket_rule_without_live_edge:
            tier = "OBSERVE"
            monitor_action = "OBSERVE"
            would_allow = False
            reasons.append("Pocket rule is too weak without a strong recent live outcome edge; downgraded to OBSERVE.")
        elif score < 80:
            if _allow_tier_is_globally_bad(global_weak_outcome, min_evaluated=10):
                tier = "OBSERVE"
                monitor_action = "OBSERVE"
                would_allow = False
                reasons.append("Global WEAK_ALLOW outcomes are poor; WEAK_ALLOW is suspended to OBSERVE.")
            elif _weak_allow_is_bad(weak_outcome):
                tier = "OBSERVE"
                monitor_action = "OBSERVE"
                would_allow = False
                reasons.append("Recent WEAK_ALLOW outcomes for this pocket are poor; downgraded to OBSERVE.")
            else:
                tier = "WEAK_ALLOW_MONITOR"
                monitor_action = "WEAK_ALLOW_MONITOR"
                would_allow = risk.action != Action.NO_TRADE
        else:
            if _allow_tier_is_globally_bad(global_strong_outcome, min_evaluated=3):
                tier = "OBSERVE"
                monitor_action = "OBSERVE"
                would_allow = False
                reasons.append("Global STRONG_ALLOW outcomes are poor; STRONG_ALLOW is suspended to OBSERVE.")
            elif _weak_allow_is_bad(weak_outcome):
                tier = "OBSERVE"
                monitor_action = "OBSERVE"
                would_allow = False
                reasons.append("Recent WEAK_ALLOW outcomes for this pocket are poor; high score was downgraded to OBSERVE.")
            else:
                tier = "STRONG_ALLOW_MONITOR"
                monitor_action = "STRONG_ALLOW_MONITOR"
                would_allow = risk.action != Action.NO_TRADE

    if candidate_buy and monitor_action in {"BLOCK", "OBSERVE"} and _is_recovery_candidate(
        blocked_outcome,
        pocket_rule=pocket_rule,
        recovery_audit=recovery_audit,
    ):
        tier = "RECOVERY_MONITOR"
        monitor_action = "RECOVERY_MONITOR"
        would_allow = False
        if recovery_audit:
            reasons.append(
                f"Recovery audit pocket is strong for this exact setup: avg "
                f"{recovery_audit.get('avg_realized_pct', 0):.4f}% over "
                f"{recovery_audit.get('evaluated', 0)} evaluated rejected BUY events."
            )
        reasons.append(
            f"Recovery monitor candidate: recent BLOCK outcomes for this pocket are strong "
            f"({(blocked_outcome or {}).get('avg_realized_pct', 0):.4f}% avg, "
            f"{(blocked_outcome or {}).get('win_rate_pct', 0):.2f}% win rate)."
        )

    if monitor_action in {"WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR", "RECOVERY_MONITOR"} and risk.action == Action.NO_TRADE:
        reasons.append("Opportunity tier is monitor-only because the independent risk engine still rejected execution.")

    if not reasons:
        reasons.append("No major edge warning detected.")

    return EdgeFilterResult(
        mode="monitor_only",
        monitor_action=monitor_action,
        would_allow_trade=would_allow,
        score=score,
        estimated_ev_pct=ev,
        ml_accuracy=accuracy,
        ml_macro_f1=macro_f1,
        ml_sideways_accuracy=sideways_accuracy,
        regime=regime,
        reasons=reasons,
        observed_action=risk.action.value,
        tier=tier,
        pocket_score=pocket_score,
        outcome_score=outcome_score,
        blocked_outcome_score=blocked_outcome_score,
    )

def append_edge_event(config: AgentConfig, event: dict[str, Any]) -> None:
    path = Path(config.edge_events_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_safe(event), ensure_ascii=True) + "\n")


def recent_edge_events(config: AgentConfig, limit: int = 200) -> list[dict[str, Any]]:
    path = Path(config.edge_events_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_edge_events(config: AgentConfig, limit: int = 500) -> dict[str, Any]:
    events = recent_edge_events(config, limit)
    status = _load_edge_monitor_status(config)
    if not events:
        return {"events": 0, "allow": 0, "block": 0, "observe": 0, "daemon": status}
    counts = {"ALLOW": 0, "BLOCK": 0, "OBSERVE": 0, "WEAK_ALLOW_MONITOR": 0, "STRONG_ALLOW_MONITOR": 0, "RECOVERY_MONITOR": 0}
    scores: list[float] = []
    evs: list[float] = []
    for event in events:
        edge = event.get("edge_filter", {})
        action = str(edge.get("monitor_action", "OBSERVE")).upper()
        counts[action] = counts.get(action, 0) + 1
        if edge.get("score") is not None:
            scores.append(float(edge["score"]))
        if edge.get("estimated_ev_pct") is not None:
            evs.append(float(edge["estimated_ev_pct"]))
    return {
        "events": len(events),
        "allow": counts.get("ALLOW", 0),
        "weak_allow": counts.get("WEAK_ALLOW_MONITOR", 0),
        "strong_allow": counts.get("STRONG_ALLOW_MONITOR", 0),
        "recovery_monitor": counts.get("RECOVERY_MONITOR", 0),
        "block": counts.get("BLOCK", 0),
        "observe": counts.get("OBSERVE", 0),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "avg_ev_pct": round(sum(evs) / len(evs), 4) if evs else 0.0,
        "daemon": status,
        "recent": events[-12:],
    }


def _model_metrics(config: AgentConfig, symbol: str, interval: str) -> dict[str, Any]:
    path = Path(config.ml_report_path)
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    model = (report.get("models") or {}).get(f"{symbol.upper()}_{interval}")
    return model.get("metrics", {}) if isinstance(model, dict) else {}


def historical_pocket_stats(config: AgentConfig, symbol: str, interval: str) -> dict[str, Any] | None:
    path = Path(config.edge_events_path).with_name("edge_pocket_stats.json")
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    pocket = raw.get(f"{symbol.upper()}_{interval}")
    return pocket if isinstance(pocket, dict) else None


def live_outcome_stats(config: AgentConfig, symbol: str, interval: str) -> dict[str, Any] | None:
    path = Path(config.edge_events_path).with_name("edge_outcome_report.json")
    report = _load_json_dict(path)
    for item in report.get("by_symbol_interval", []) if isinstance(report, dict) else []:
        if item.get("key") == f"{symbol.upper()}_{interval}":
            return item if isinstance(item, dict) else None
    return None


def live_edge_outcome_stats(config: AgentConfig, edge_action: str, symbol: str, interval: str) -> dict[str, Any] | None:
    path = Path(config.edge_events_path).with_name("edge_outcome_report.json")
    report = _load_json_dict(path)
    key = f"{edge_action}|{symbol.upper()}_{interval}"
    for item in report.get("by_edge_symbol_interval", []) if isinstance(report, dict) else []:
        if item.get("key") == key:
            return item if isinstance(item, dict) else None
    return None


def live_edge_action_stats(config: AgentConfig, edge_action: str) -> dict[str, Any] | None:
    path = Path(config.edge_events_path).with_name("edge_outcome_report.json")
    report = _load_json_dict(path)
    for item in report.get("by_edge_action", []) if isinstance(report, dict) else []:
        if item.get("key") == edge_action:
            return item if isinstance(item, dict) else None
    return None


def recovery_audit_stats(config: AgentConfig, symbol: str, interval: str, strategy: str, regime: str) -> dict[str, Any] | None:
    path = Path(config.edge_events_path).with_name("blocked_winners_audit.json")
    report = _load_json_dict(path)
    if not report:
        return None
    generated_at = _parse_time(report.get("generated_at"))
    if generated_at is None or datetime.now(timezone.utc) - generated_at > timedelta(hours=6):
        return None
    key = f"{symbol.upper()}_{interval}|{strategy}|{regime}"
    for item in report.get("best_pockets", []) if isinstance(report, dict) else []:
        if item.get("key") == key:
            return item if isinstance(item, dict) else None
    return None


def _pocket_score(pocket: dict[str, Any]) -> int:
    buy_events = int(_float_or_none(pocket.get("buy_events")) or 0)
    avg = _float_or_none(pocket.get("avg_realized_pct")) or 0.0
    win_rate = _float_or_none(pocket.get("win_rate_pct")) or 0.0
    if buy_events < 20:
        return 0
    if buy_events >= 20 and win_rate < 25:
        return -14 if avg >= 0 else -20
    score = 0
    if avg > 0.2 and buy_events >= 30 and win_rate >= 40:
        score += 22
    elif avg > 0.05 and win_rate >= 35:
        score += 14
    elif avg > 0 and win_rate >= 30:
        score += 8
    elif avg < -0.1:
        score -= 12
    if win_rate >= 45 and buy_events >= 30:
        score += 8
    elif win_rate < 35 and buy_events >= 30:
        score -= 6
    return max(-20, min(30, score))


def _outcome_score(stats: dict[str, Any]) -> int:
    evaluated = int(_float_or_none(stats.get("evaluated")) or 0)
    avg = _float_or_none(stats.get("avg_realized_pct")) or 0.0
    win_rate = _float_or_none(stats.get("win_rate_pct")) or 0.0
    if evaluated < 20:
        return 0
    score = 0
    if avg >= 0.75 and win_rate >= 55:
        score += 28
    elif avg >= 0.25 and win_rate >= 45:
        score += 16
    elif avg >= 0.05:
        score += 6
    elif avg <= -0.15:
        score -= 18
    elif avg < 0:
        score -= 8
    if win_rate >= 60 and evaluated >= 50:
        score += 8
    elif win_rate < 30:
        score -= 8
    return max(-25, min(35, score))


def _blocked_outcome_score(stats: dict[str, Any]) -> int:
    evaluated = int(_float_or_none(stats.get("evaluated")) or 0)
    avg = _float_or_none(stats.get("avg_realized_pct")) or 0.0
    win_rate = _float_or_none(stats.get("win_rate_pct")) or 0.0
    if evaluated < 20:
        return 0
    score = 0
    if avg >= 0.75 and win_rate >= 55:
        score += 28
    elif avg >= 0.25 and win_rate >= 50:
        score += 16
    elif avg > 0 and win_rate >= 45:
        score += 8
    elif avg <= -0.15:
        score -= 12
    if win_rate >= 65 and evaluated >= 40:
        score += 6
    return max(-15, min(32, score))


def _blocked_outcome_has_independent_confirmation(
    *,
    ev: float | None,
    macro_f1: float | None,
    pocket_score: int,
    quality_score: int,
    rule_score: int,
) -> bool:
    return (
        (ev is not None and ev > 0)
        or (macro_f1 is not None and macro_f1 >= 38)
        or pocket_score >= 22
        or quality_score >= 10
        or rule_score >= 8
    )


def _weak_allow_is_bad(stats: dict[str, Any] | None) -> bool:
    if not stats:
        return False
    evaluated = int(_float_or_none(stats.get("evaluated")) or 0)
    avg = _float_or_none(stats.get("avg_realized_pct")) or 0.0
    win_rate = _float_or_none(stats.get("win_rate_pct")) or 0.0
    return evaluated >= 20 and (avg <= 0.02 or win_rate < 30)


def _allow_tier_is_globally_bad(stats: dict[str, Any] | None, *, min_evaluated: int) -> bool:
    if not stats:
        return False
    evaluated = int(_float_or_none(stats.get("evaluated")) or 0)
    avg = _float_or_none(stats.get("avg_realized_pct")) or 0.0
    win_rate = _float_or_none(stats.get("win_rate_pct")) or 0.0
    if evaluated < min_evaluated:
        return False
    return avg <= -0.5 or win_rate < 25


def _is_recovery_candidate(
    stats: dict[str, Any] | None,
    *,
    pocket_rule: dict[str, Any] | None,
    recovery_audit: dict[str, Any] | None = None,
) -> bool:
    if recovery_audit and _audit_recovery_candidate(recovery_audit):
        return True
    if not stats or not pocket_rule:
        return False
    rule_events = int(_float_or_none(pocket_rule.get("buy_events")) or 0)
    rule_avg = _float_or_none(pocket_rule.get("avg_realized_pct")) or 0.0
    rule_win_rate = _float_or_none(pocket_rule.get("win_rate_pct")) or 0.0
    if rule_events < 20 or rule_avg < 0.5 or rule_win_rate < 55:
        return False
    evaluated = int(_float_or_none(stats.get("evaluated")) or 0)
    pending = int(_float_or_none(stats.get("pending")) or 0)
    avg = _float_or_none(stats.get("avg_realized_pct")) or 0.0
    win_rate = _float_or_none(stats.get("win_rate_pct")) or 0.0
    if evaluated < 20:
        return False
    if pending > evaluated * 2:
        return False
    return avg >= 0.5 and win_rate >= 60


def _audit_recovery_candidate(stats: dict[str, Any]) -> bool:
    evaluated = int(_float_or_none(stats.get("evaluated")) or 0)
    pending = int(_float_or_none(stats.get("pending")) or 0)
    avg = _float_or_none(stats.get("avg_realized_pct")) or 0.0
    win_rate = _float_or_none(stats.get("win_rate_pct")) or 0.0
    if evaluated < 10:
        return False
    if pending > evaluated * 2:
        return False
    return avg >= 0.5 and win_rate >= 60


def _load_edge_monitor_status(config: AgentConfig) -> dict[str, Any] | None:
    path = Path(config.edge_events_path).with_name("edge_monitor_status.json")
    if not path.exists():
        return None
    status = _load_json_dict(path)
    if not status:
        return {"state": "invalid"}
    return status


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _regime(state: MarketState) -> str:
    if state.volatility == "high":
        return "high_volatility"
    if state.trend == "sideways" or state.sideways_probability >= max(state.bullish_probability, state.bearish_probability):
        return "sideways"
    if state.trend in {"bullish", "bearish"}:
        return "trending"
    return "mixed"


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


def _json_safe(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
