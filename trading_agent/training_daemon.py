from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.auto_trainer import run_auto_training
from trading_agent.config import AgentConfig
from trading_agent.universe import scan_bybit_universe


def run_training_daemon(
    config: AgentConfig,
    provider: str = "bybit",
    sleep_minutes: int = 30,
    scan_universe: bool = True,
) -> None:
    status_path = Path(config.ml_report_path).with_name("training_daemon_status.json")
    stop_path = Path(config.ml_report_path).with_name("training_daemon.stop")
    status_path.parent.mkdir(parents=True, exist_ok=True)

    cycle = 0
    while True:
        if stop_path.exists():
            _write_status(status_path, {"state": "stopped", "reason": "stop file exists", "cycle": cycle})
            return

        cycle += 1
        started_at = datetime.now(timezone.utc).isoformat()
        _write_status(
            status_path,
            {
                "state": "running",
                "cycle": cycle,
                "started_at": started_at,
                "provider": provider,
                "sleep_minutes": sleep_minutes,
            },
        )

        try:
            universe = None
            if scan_universe and provider == "bybit":
                universe = scan_bybit_universe(config)
            report = run_auto_training(config, provider)
            _write_status(
                status_path,
                {
                    "state": "sleeping",
                    "cycle": cycle,
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "provider": provider,
                    "sleep_minutes": sleep_minutes,
                    "universe_symbols": universe.get("selected_symbols", []) if universe else None,
                    "summary": report.get("summary", report),
                },
            )
        except Exception as exc:
            _write_status(
                status_path,
                {
                    "state": "error_sleeping",
                    "cycle": cycle,
                    "started_at": started_at,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "provider": provider,
                    "sleep_minutes": sleep_minutes,
                    "error": str(exc),
                },
            )

        _sleep_with_stop_check(stop_path, sleep_minutes * 60)


def _sleep_with_stop_check(stop_path: Path, seconds: int) -> None:
    end_time = time.time() + seconds
    while time.time() < end_time:
        if stop_path.exists():
            return
        time.sleep(min(30, max(1, int(end_time - time.time()))))


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
