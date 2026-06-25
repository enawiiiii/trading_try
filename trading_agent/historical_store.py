from __future__ import annotations

import csv
from pathlib import Path

from trading_agent.config import AgentConfig
from trading_agent.models import Candle


HEADER = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades"]


def save_candles(config: AgentConfig, provider: str, symbol: str, interval: str, candles: list[Candle]) -> Path:
    path = _path(config, provider, symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _merge_existing(path, candles)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows.values())
    return path


def load_candles(config: AgentConfig, provider: str, symbol: str, interval: str) -> list[Candle]:
    path = _path(config, provider, symbol, interval)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [_row_to_candle(row) for row in reader]


def _merge_existing(path: Path, candles: list[Candle]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows[row["open_time"]] = row
    for candle in candles:
        row = _candle_to_row(candle)
        rows[row["open_time"]] = row
    return dict(sorted(rows.items(), key=lambda item: item[0]))


def _path(config: AgentConfig, provider: str, symbol: str, interval: str) -> Path:
    filename = f"{provider.lower()}_{symbol.upper()}_{interval}.csv"
    return Path(config.historical_data_dir) / filename


def _candle_to_row(candle: Candle) -> dict:
    return {
        "open_time": candle.open_time.isoformat(),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "close_time": candle.close_time.isoformat(),
        "quote_volume": candle.quote_volume,
        "trades": candle.trades,
    }


def _row_to_candle(row: dict) -> Candle:
    from datetime import datetime

    return Candle(
        open_time=datetime.fromisoformat(row["open_time"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        close_time=datetime.fromisoformat(row["close_time"]),
        quote_volume=float(row["quote_volume"]),
        trades=int(float(row["trades"])),
    )
