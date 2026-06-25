from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    quote_asset: str = "USDT"
    paper_starting_cash: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    daily_loss_limit_pct: float = 3.0
    max_consecutive_losses: int = 3
    min_buy_confidence: int = 70
    min_sell_confidence: int = 65
    max_fake_breakout_probability: int = 65
    min_liquidity_score: int = 35
    journal_path: str = "data/journal.jsonl"
    learning_path: str = "data/learning_state.json"
    calibration_path: str = "data/calibration_state.json"
    historical_data_dir: str = "data/historical"
    ml_model_dir: str = "data/models"
    portfolio_path: str = "data/paper_portfolio.json"
    exchange: str = "bybit"
    bybit_market_testnet: bool = False
    bybit_testnet: bool = True
    bybit_demo: bool = False
    bybit_time_offset_ms: int = 0
    live_trading_enabled: bool = False
    max_live_order_quote: float = 25.0
    stop_loss_pct: float = 1.2
    take_profit_pct: float = 2.4
    trailing_stop_pct: float = 0.9
    min_exit_bearish_probability: int = 45
    training_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "LTCUSDT")
    training_intervals: tuple[str, ...] = ("1h", "4h")
    training_history_limits: dict[str, int] | None = None
    ml_min_accuracy: float = 38.0
    ml_report_path: str = "data/models/auto_training_report.json"
    edge_events_path: str = "data/edge_filter_events.jsonl"
    universe_path: str = "data/universe.json"
    universe_top_n: int = 20
    universe_min_turnover_24h: float = 1_000_000.0
    universe_max_spread_pct: float = 0.25


def load_config(path: str | Path = "config.json") -> AgentConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AgentConfig()

    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    allowed = set(AgentConfig.__dataclass_fields__)
    clean = {key: value for key, value in raw.items() if key in allowed}
    if isinstance(clean.get("training_symbols"), list):
        clean["training_symbols"] = tuple(clean["training_symbols"])
    if isinstance(clean.get("training_intervals"), list):
        clean["training_intervals"] = tuple(clean["training_intervals"])
    return AgentConfig(**clean)
