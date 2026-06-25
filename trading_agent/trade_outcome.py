from __future__ import annotations

from typing import Any

from trading_agent.models import Candle


def long_outcome(entry: float, future: list[Candle], take_profit_pct: float, stop_loss_pct: float, fee_pct: float = 0.1) -> dict[str, Any]:
    take_profit = entry * (1 + take_profit_pct / 100)
    stop_loss = entry * (1 - stop_loss_pct / 100)
    for candle in future:
        hit_tp = candle.high >= take_profit
        hit_sl = candle.low <= stop_loss
        if hit_tp and hit_sl:
            return {"outcome": "ambiguous", "realized_pct": round(-fee_pct, 4)}
        if hit_tp:
            return {"outcome": "take_profit", "realized_pct": round(take_profit_pct - fee_pct, 4)}
        if hit_sl:
            return {"outcome": "stop_loss", "realized_pct": round(-stop_loss_pct - fee_pct, 4)}
    close = future[-1].close if future else entry
    return {"outcome": "timeout", "realized_pct": round(((close - entry) / entry) * 100 - fee_pct, 4)}


def horizon_for_interval(interval: str) -> int:
    if interval == "4h":
        return 3
    if interval == "15m":
        return 8
    return 6
