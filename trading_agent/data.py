from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from trading_agent.models import Candle


BINANCE_SPOT_BASE_URL = "https://api.binance.com"


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 240) -> list[Candle]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(max(limit, 50), 1000),
        }
    )
    url = f"{BINANCE_SPOT_BASE_URL}/api/v3/klines?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "spot-trading-agent/0.1"})

    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if isinstance(payload, dict) and "code" in payload:
        raise RuntimeError(f"Binance API error {payload.get('code')}: {payload.get('msg')}")

    return _closed_candles([Candle.from_binance(row) for row in payload])


def fetch_bybit_klines(symbol: str, interval: str = "60", limit: int = 240, testnet: bool = False) -> list[Candle]:
    try:
        from pybit.unified_trading import HTTP
    except ImportError as exc:
        raise RuntimeError("pybit is not installed. Run: python -m pip install -r requirements.txt") from exc

    session = HTTP(testnet=testnet)
    response = session.get_kline(
        category="spot",
        symbol=symbol.upper(),
        interval=_to_bybit_interval(interval),
        limit=min(max(limit, 50), 1000),
    )
    if response.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error {response.get('retCode')}: {response.get('retMsg')}")

    rows = response.get("result", {}).get("list", [])
    candles = [_candle_from_bybit(row, _to_bybit_interval(interval)) for row in rows]
    return _closed_candles(sorted(candles, key=lambda candle: candle.open_time))


def fetch_market_klines(exchange: str, symbol: str, interval: str, limit: int, testnet: bool = False) -> list[Candle]:
    if exchange.lower() == "bybit":
        return fetch_bybit_klines(symbol, interval, limit, testnet)
    if exchange.lower() == "binance":
        return fetch_klines(symbol, interval, limit)
    raise ValueError(f"Unsupported exchange: {exchange}")


def fetch_historical_market_klines(
    exchange: str,
    symbol: str,
    interval: str,
    total: int,
    testnet: bool = False,
) -> list[Candle]:
    if exchange.lower() == "bybit":
        return fetch_bybit_historical_klines(symbol, interval, total, testnet)
    if exchange.lower() == "binance":
        return fetch_binance_historical_klines(symbol, interval, total)
    raise ValueError(f"Unsupported exchange: {exchange}")


def fetch_binance_historical_klines(symbol: str, interval: str = "1h", total: int = 3000) -> list[Candle]:
    candles: list[Candle] = []
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    remaining = total

    while remaining > 0:
        batch_limit = min(1000, remaining)
        params = urllib.parse.urlencode(
            {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": batch_limit,
                "endTime": end_time,
            }
        )
        url = f"{BINANCE_SPOT_BASE_URL}/api/v3/klines?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": "spot-trading-agent/0.1"})
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict) and "code" in payload:
            raise RuntimeError(f"Binance API error {payload.get('code')}: {payload.get('msg')}")
        if not payload:
            break
        batch = [Candle.from_binance(row) for row in payload]
        candles = batch + candles
        end_time = int(batch[0].open_time.timestamp() * 1000) - 1
        remaining -= len(batch)
        time.sleep(0.08)
        if len(batch) < batch_limit:
            break

    return _dedupe_sorted_closed(candles)[-total:]


def fetch_bybit_historical_klines(symbol: str, interval: str = "1h", total: int = 3000, testnet: bool = False) -> list[Candle]:
    try:
        from pybit.unified_trading import HTTP
    except ImportError as exc:
        raise RuntimeError("pybit is not installed. Run: python -m pip install -r requirements.txt") from exc

    session = HTTP(testnet=testnet)
    bybit_interval = _to_bybit_interval(interval)
    candles: list[Candle] = []
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    remaining = total

    while remaining > 0:
        batch_limit = min(1000, remaining)
        response = session.get_kline(
            category="spot",
            symbol=symbol.upper(),
            interval=bybit_interval,
            end=end_time,
            limit=batch_limit,
        )
        if response.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error {response.get('retCode')}: {response.get('retMsg')}")
        rows = response.get("result", {}).get("list", [])
        if not rows:
            break
        batch = sorted([_candle_from_bybit(row, bybit_interval) for row in rows], key=lambda candle: candle.open_time)
        candles = batch + candles
        end_time = int(batch[0].open_time.timestamp() * 1000) - 1
        remaining -= len(batch)
        time.sleep(0.08)
        if len(batch) < batch_limit:
            break

    return _dedupe_sorted_closed(candles)[-total:]


def _to_bybit_interval(interval: str) -> str:
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "D": "D",
        "W": "W",
        "M": "M",
    }
    return mapping.get(str(interval), str(interval))


def _candle_from_bybit(row: list[str], interval: str) -> Candle:
    open_time = Candle.from_binance(
        [
            int(row[0]),
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            _estimated_close_ms(int(row[0]), interval),
            row[6],
            0,
        ]
    ).open_time
    return Candle(
        open_time=open_time,
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        close_time=open_time + _interval_delta(interval),
        quote_volume=float(row[6]),
        trades=0,
    )


def _estimated_close_ms(open_ms: int, interval: str) -> int:
    return int((Candle.from_binance([open_ms, 0, 0, 0, 0, 0, open_ms, 0, 0]).open_time + _interval_delta(interval)).timestamp() * 1000)


def _interval_delta(interval: str) -> timedelta:
    if interval == "D":
        return timedelta(days=1)
    if interval == "W":
        return timedelta(weeks=1)
    if interval == "M":
        return timedelta(days=30)
    return timedelta(minutes=int(interval))


def _closed_candles(candles: list[Candle]) -> list[Candle]:
    now = datetime.now(timezone.utc)
    closed = [candle for candle in candles if candle.close_time <= now]
    return closed if len(closed) >= 60 else candles


def _dedupe_sorted_closed(candles: list[Candle]) -> list[Candle]:
    closed = _closed_candles(candles)
    deduped = {candle.open_time: candle for candle in closed}
    return [deduped[key] for key in sorted(deduped)]
