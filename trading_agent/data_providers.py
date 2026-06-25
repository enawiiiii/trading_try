from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

from trading_agent.data import _interval_delta
from trading_agent.models import Candle


COINAPI_BASE_URL = "https://rest.coinapi.io"
CMC_BASE_URL = "https://pro-api.coinmarketcap.com"


def fetch_coinapi_ohlcv(
    symbol: str,
    interval: str = "1h",
    limit: int = 1000,
    exchange: str = "BYBIT",
) -> list[Candle]:
    api_key = os.getenv("COINAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing COINAPI_KEY environment variable.")

    symbol_id = _coinapi_symbol_id(symbol, exchange)
    period_id = _coinapi_period(interval)
    params = urllib.parse.urlencode({"period_id": period_id, "limit": min(limit, 100000)})
    url = f"{COINAPI_BASE_URL}/v1/ohlcv/{symbol_id}/history?{params}"
    request = urllib.request.Request(url, headers={"X-CoinAPI-Key": api_key})
    payload = _get_json(request)
    return [_coinapi_candle(row, interval) for row in payload]


def fetch_coinmarketcap_quotes(symbols: list[str], convert: str = "USD") -> dict:
    api_key = os.getenv("COINMARKETCAP_API_KEY")
    if not api_key:
        raise RuntimeError("Missing COINMARKETCAP_API_KEY environment variable.")

    clean_symbols = ",".join(_base_asset(symbol) for symbol in symbols)
    params = urllib.parse.urlencode({"symbol": clean_symbols, "convert": convert})
    url = f"{CMC_BASE_URL}/v1/cryptocurrency/quotes/latest?{params}"
    request = urllib.request.Request(url, headers={"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"})
    return _get_json(request)


def _get_json(request: urllib.request.Request):
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
            detail = payload.get("detail") or payload.get("error") or payload.get("status", {}).get("error_message")
        except json.JSONDecodeError:
            detail = body[:500]
        raise RuntimeError(f"Provider API error {exc.code}: {detail}") from exc


def _coinapi_symbol_id(symbol: str, exchange: str) -> str:
    base = _base_asset(symbol)
    quote = "USDT" if symbol.upper().endswith("USDT") else "USD"
    return f"{exchange.upper()}_SPOT_{base}_{quote}"


def _base_asset(symbol: str) -> str:
    symbol = symbol.upper()
    for quote in ("USDT", "USD", "BTC", "ETH"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    return symbol


def _coinapi_period(interval: str) -> str:
    mapping = {
        "1m": "1MIN",
        "5m": "5MIN",
        "15m": "15MIN",
        "30m": "30MIN",
        "1h": "1HRS",
        "2h": "2HRS",
        "4h": "4HRS",
        "1d": "1DAY",
    }
    return mapping.get(interval, interval.upper())


def _coinapi_candle(row: dict, interval: str) -> Candle:
    open_time = _parse_time(row["time_period_start"])
    close_time = _parse_time(row.get("time_period_end")) if row.get("time_period_end") else open_time + _interval_delta(_bybit_like_interval(interval))
    return Candle(
        open_time=open_time,
        open=float(row["price_open"]),
        high=float(row["price_high"]),
        low=float(row["price_low"]),
        close=float(row["price_close"]),
        volume=float(row.get("volume_traded", 0.0)),
        close_time=close_time,
        quote_volume=float(row.get("volume_traded", 0.0)) * float(row["price_close"]),
        trades=int(row.get("trades_count", 0) or 0),
    )


def _parse_time(value: str) -> datetime:
    clean = value.replace("Z", "+00:00")
    return datetime.fromisoformat(clean).astimezone(timezone.utc)


def _bybit_like_interval(interval: str) -> str:
    return {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "1d": "D",
    }.get(interval, interval)
