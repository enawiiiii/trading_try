from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.backtest import backtest_probabilities
from trading_agent.calibration import save_calibration
from trading_agent.config import AgentConfig
from trading_agent.data import fetch_historical_market_klines, fetch_market_klines
from trading_agent.historical_store import load_candles, save_candles
from trading_agent.ml_model import build_dataset, predict_model, save_model, train_model
from trading_agent.universe import load_universe_symbols


def run_auto_training(config: AgentConfig, provider: str = "bybit") -> dict[str, Any]:
    lock_path = Path(config.ml_report_path).with_suffix(".lock")
    if lock_path.exists():
        return {
            "status": "already_running",
            "lock_path": str(lock_path),
            "message": "Auto training is already running or a stale lock exists.",
        }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "symbols": load_universe_symbols(config),
        "intervals": list(config.training_intervals),
        "models": {},
        "summary": {
            "trained": 0,
            "approved": 0,
            "rejected": 0,
            "errors": 0,
        },
    }
    _save_report(config, report)

    try:
        for symbol in load_universe_symbols(config):
            for interval in config.training_intervals:
                key = f"{symbol}_{interval}"
                report["current_task"] = key
                _save_report(config, report)
                try:
                    item = _train_one(config, provider, symbol, interval)
                    report["models"][key] = item
                    report["summary"]["trained"] += 1
                    if item["approved"]:
                        report["summary"]["approved"] += 1
                    else:
                        report["summary"]["rejected"] += 1
                except Exception as exc:
                    report["models"][key] = {"error": str(exc), "approved": False}
                    report["summary"]["errors"] += 1
                report["last_checkpoint_at"] = datetime.now(timezone.utc).isoformat()
                _save_report(config, report)

        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report.pop("current_task", None)
        _save_report(config, report)
        return report
    finally:
        lock_path.unlink(missing_ok=True)


def _train_one(config: AgentConfig, provider: str, symbol: str, interval: str) -> dict[str, Any]:
    history_limit = _history_limit(config, interval)
    candles = fetch_historical_market_klines(provider, symbol, interval, history_limit, config.bybit_market_testnet)
    save_candles(config, provider, symbol, interval, candles)
    stored = load_candles(config, provider, symbol, interval)

    lookback, horizon, threshold, epochs, learning_rate = _training_profile(interval)
    label_mode = "tp_sl_path"
    take_profit_pct = threshold
    stop_loss_pct = threshold
    dataset = build_dataset(
        stored,
        lookback=lookback,
        horizon=horizon,
        move_threshold_pct=threshold,
        label_mode=label_mode,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    model, metrics = train_model(
        dataset,
        lookback=lookback,
        horizon=horizon,
        move_threshold_pct=threshold,
        label_mode=label_mode,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    model_path = f"{config.ml_model_dir}/{provider}_{symbol.upper()}_{interval}_scenario.json"
    save_model(model, model_path)

    backtest = backtest_probabilities(stored, lookback=lookback, horizon=horizon, move_threshold_pct=threshold)
    if interval == "4h":
        save_calibration(config, symbol, interval, backtest)

    live_candles = fetch_market_klines(config.exchange, symbol, interval, max(240, lookback), config.bybit_market_testnet)
    prediction = predict_model(model, live_candles)
    approved = metrics["accuracy"] >= config.ml_min_accuracy

    return {
        "candles": len(stored),
        "model_path": model_path,
        "lookback": lookback,
        "horizon": horizon,
        "move_threshold_pct": threshold,
        "label_mode": label_mode,
        "take_profit_pct": take_profit_pct,
        "stop_loss_pct": stop_loss_pct,
        "metrics": metrics,
        "backtest_accuracy": backtest["accuracy"],
        "current_prediction": prediction,
        "approved": approved,
        "approval_reason": (
            f"Accuracy {metrics['accuracy']} >= required {config.ml_min_accuracy}"
            if approved
            else f"Accuracy {metrics['accuracy']} below required {config.ml_min_accuracy}"
        ),
    }


def _history_limit(config: AgentConfig, interval: str) -> int:
    limits = config.training_history_limits or {}
    return int(limits.get(interval, 5000 if interval == "4h" else 10000))


def _training_profile(interval: str) -> tuple[int, int, float, int, float]:
    if interval == "4h":
        return 80, 3, 0.75, 220, 0.018
    if interval == "15m":
        return 96, 8, 0.35, 180, 0.018
    return 80, 6, 0.35, 260, 0.018


def _save_report(config: AgentConfig, report: dict[str, Any]) -> None:
    path = Path(config.ml_report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
