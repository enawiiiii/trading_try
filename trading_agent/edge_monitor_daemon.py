from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_agent.agent import decide
from trading_agent.blocked_winners_audit import build_blocked_winners_audit
from trading_agent.config import AgentConfig
from trading_agent.data import fetch_market_klines
from trading_agent.edge_outcomes import analyze_edge_outcomes
from trading_agent.paper_simulator import update_paper_simulator
from trading_agent.portfolio import load_portfolio
from trading_agent.recovery_shadow import update_recovery_shadow_trades
from trading_agent.recovery_watch import build_recovery_watch_report
from trading_agent.risk import RiskContext


def run_edge_monitor_daemon(
    config: AgentConfig,
    symbols: list[str],
    interval: str = "1h",
    exchange: str | None = None,
    limit: int = 240,
    sleep_minutes: int = 5,
    top_ev_symbols: int = 8,
) -> None:
    status_path = Path(config.edge_events_path).with_name("edge_monitor_status.json")
    stop_path = Path(config.edge_events_path).with_name("edge_monitor.stop")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    base_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    intervals = [item.strip() for item in interval.split(",") if item.strip()] or ["1h"]
    selected_exchange = exchange or config.exchange
    cycle = 0

    while True:
        if stop_path.exists():
            _write_status(status_path, {"state": "stopped", "reason": "stop file exists", "cycle": cycle})
            return

        cycle += 1
        interval_symbols = {
            current_interval: _selected_symbols(config, base_symbols, current_interval, top_ev_symbols)
            for current_interval in intervals
        }
        clean_symbols = sorted({symbol for values in interval_symbols.values() for symbol in values})
        started_at = datetime.now(timezone.utc).isoformat()
        summary = {"events": 0, "errors": 0, "symbols": {}, "intervals": {}}
        _write_status(
            status_path,
            {
                "state": "running",
                "cycle": cycle,
                "started_at": started_at,
                "exchange": selected_exchange,
                "interval": ",".join(intervals),
                "sleep_minutes": sleep_minutes,
                "symbols": clean_symbols,
                "interval_symbols": interval_symbols,
            },
        )

        portfolio = load_portfolio(config)
        for current_interval, symbols_for_interval in interval_symbols.items():
            summary["intervals"][current_interval] = {"events": 0, "errors": 0, "symbols": {}}
            for symbol in symbols_for_interval:
                if stop_path.exists():
                    break
                try:
                    candles = fetch_market_klines(selected_exchange, symbol, current_interval, limit, config.bybit_market_testnet)
                    context = RiskContext(
                        daily_pnl_pct=portfolio.daily_pnl_pct,
                        consecutive_losses=portfolio.consecutive_losses,
                        has_position=portfolio.has_position(symbol),
                    )
                    decision = decide(symbol, candles, config, context, current_interval)
                    row = {
                        "status": "ok",
                        "action": decision.action.value,
                        "edge_action": (decision.edge_filter or {}).get("monitor_action"),
                        "score": (decision.edge_filter or {}).get("score"),
                        "ev": (decision.edge_filter or {}).get("estimated_ev_pct"),
                    }
                    summary["events"] += 1
                    summary["intervals"][current_interval]["events"] += 1
                    summary["symbols"][f"{symbol}_{current_interval}"] = row
                    summary["intervals"][current_interval]["symbols"][symbol] = row
                except Exception as exc:
                    row = {"status": "error", "error": str(exc)}
                    summary["errors"] += 1
                    summary["intervals"][current_interval]["errors"] += 1
                    summary["symbols"][f"{symbol}_{current_interval}"] = row
                    summary["intervals"][current_interval]["symbols"][symbol] = row

        _write_status(
            status_path,
            {
                "state": "sleeping",
                "cycle": cycle,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "exchange": selected_exchange,
                "interval": ",".join(intervals),
                "sleep_minutes": sleep_minutes,
                "symbols": clean_symbols,
                "interval_symbols": interval_symbols,
                "summary": summary,
            },
        )
        _refresh_outcomes_if_stale(config, selected_exchange, max(15, sleep_minutes * 3))
        _refresh_recovery_pipeline_if_stale(config, selected_exchange, max(15, sleep_minutes * 3))
        _refresh_paper_simulator_if_stale(config, selected_exchange, max(5, sleep_minutes))
        _sleep_with_stop_check(stop_path, max(1, sleep_minutes) * 60)


def _sleep_with_stop_check(stop_path: Path, seconds: int) -> None:
    end_time = time.time() + seconds
    while time.time() < end_time:
        if stop_path.exists():
            return
        time.sleep(min(15, max(1, int(end_time - time.time()))))


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _refresh_outcomes_if_stale(config: AgentConfig, exchange: str, max_age_minutes: int) -> None:
    path = Path(config.edge_events_path).with_name("edge_outcome_report.json")
    if path.exists():
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if age < timedelta(minutes=max_age_minutes):
            return
    try:
        analyze_edge_outcomes(config, exchange=exchange, hours=24, limit=1000)
    except Exception:
        return


def _refresh_recovery_pipeline_if_stale(config: AgentConfig, exchange: str, max_age_minutes: int) -> None:
    audit_path = Path(config.edge_events_path).with_name("blocked_winners_audit.json")
    watch_path = Path(config.edge_events_path).with_name("recovery_watch_report.json")
    shadow_path = Path(config.edge_events_path).with_name("recovery_shadow_report.json")
    try:
        if _is_stale(audit_path, max(30, max_age_minutes * 2)):
            build_blocked_winners_audit(config, exchange=exchange, hours=72, limit=1000, min_evaluated=3)
        if _is_stale(watch_path, max_age_minutes):
            build_recovery_watch_report(config, exchange=exchange, limit=240, max_pockets=12)
        if _is_stale(shadow_path, max_age_minutes):
            update_recovery_shadow_trades(config, exchange=exchange, limit=1000, open_from_watch=True)
    except Exception:
        return


def _refresh_paper_simulator_if_stale(config: AgentConfig, exchange: str, max_age_minutes: int) -> None:
    path = Path(config.edge_events_path).with_name("paper_simulator_report.json")
    try:
        if _is_stale(path, max_age_minutes):
            update_paper_simulator(
                config,
                exchange=exchange,
                limit=1000,
                source_hours=6,
                max_new=12,
                min_rejected_score=15,
            )
    except Exception:
        return


def _is_stale(path: Path, max_age_minutes: int) -> bool:
    if not path.exists():
        return True
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return age >= timedelta(minutes=max_age_minutes)


def _selected_symbols(config: AgentConfig, base_symbols: list[str], interval: str, top_ev_symbols: int) -> list[str]:
    selected: list[str] = []
    for symbol in base_symbols:
        if symbol not in selected:
            selected.append(symbol)

    if top_ev_symbols <= 0:
        return selected

    for symbol in _top_ev_symbols(config, interval, top_ev_symbols):
        if symbol not in selected:
            selected.append(symbol)
    for symbol in _top_signal_quality_symbols(config, interval, max(2, top_ev_symbols // 2)):
        if symbol not in selected:
            selected.append(symbol)
    return selected


def _top_ev_symbols(config: AgentConfig, interval: str, limit: int) -> list[str]:
    path = Path(config.ml_report_path)
    if not path.exists():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    candidates: list[tuple[float, float, float, int, str]] = []
    for key, item in (report.get("models") or {}).items():
        if not isinstance(item, dict) or not key.endswith(f"_{interval}"):
            continue
        metrics = item.get("metrics") or {}
        ev = _to_float(metrics.get("long_expected_value_pct"))
        accuracy = _to_float(metrics.get("accuracy"))
        macro_f1 = _to_float(metrics.get("macro_f1"))
        samples = int(_to_float(metrics.get("samples")) or 0)
        if ev is None or macro_f1 is None or accuracy is None:
            continue
        if samples < 200:
            continue
        if macro_f1 < 32:
            continue
        if ev < -0.06:
            continue
        symbol = key[: -len(f"_{interval}")]
        candidates.append((ev, macro_f1, accuracy, samples, symbol))

    candidates.sort(reverse=True)
    return [symbol for _, _, _, _, symbol in candidates[:limit]]


def _top_signal_quality_symbols(config: AgentConfig, interval: str, limit: int) -> list[str]:
    path = Path(config.edge_events_path).with_name("signal_quality_report.json")
    if not path.exists() or limit <= 0:
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    candidates: list[tuple[float, float, int, str]] = []
    suffix = f"_{interval}"
    for item in report.get("by_symbol_interval", []) if isinstance(report, dict) else []:
        key = str(item.get("key", ""))
        if not key.endswith(suffix):
            continue
        buy_events = int(_to_float(item.get("buy_events")) or 0)
        avg = _to_float(item.get("avg_realized_pct"))
        win_rate = _to_float(item.get("win_rate_pct"))
        if avg is None or win_rate is None or buy_events < 20:
            continue
        if avg <= 0 or win_rate < 35:
            continue
        symbol = key[: -len(suffix)]
        candidates.append((avg, win_rate, buy_events, symbol))
    candidates.sort(reverse=True)
    return [symbol for _, _, _, symbol in candidates[:limit]]


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
