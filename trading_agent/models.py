from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO TRADE"


class StrategyName(str, Enum):
    TREND_FOLLOWING = "Trend Following Strategy"
    BREAKOUT = "Breakout Strategy"
    MEAN_REVERSION = "Mean Reversion Strategy"


@dataclass(frozen=True)
class Candle:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    quote_volume: float
    trades: int

    @classmethod
    def from_binance(cls, row: list) -> "Candle":
        return cls(
            open_time=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            close_time=datetime.fromtimestamp(row[6] / 1000, tz=timezone.utc),
            quote_volume=float(row[7]),
            trades=int(row[8]),
        )


@dataclass(frozen=True)
class MarketState:
    trend: str
    volatility: str
    liquidity: str
    liquidity_score: int
    structure: str
    fake_breakout_probability: int
    current_price: float
    atr_pct: float
    rsi: float
    support: float
    resistance: float
    bullish_probability: int
    bearish_probability: int
    sideways_probability: int
    evidence_summary: str
    explanation: str


@dataclass(frozen=True)
class Decision:
    action: Action
    confidence: int
    selected_strategy: StrategyName
    market_state: str
    reasoning: str
    risk_level: str
    learning_note: str
    symbol: str
    price: float
    timestamp: datetime
    edge_filter: dict[str, Any] | None = None
