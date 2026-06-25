from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.config import AgentConfig


STABLE_BASES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE"}
LEVERAGED_HINTS = ("2L", "2S", "3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")


def scan_bybit_universe(config: AgentConfig, top_n: int | None = None) -> dict[str, Any]:
    try:
        from pybit.unified_trading import HTTP
    except ImportError as exc:
        raise RuntimeError("pybit is not installed. Run: python -m pip install -r requirements.txt") from exc

    session = HTTP(testnet=config.bybit_market_testnet)
    instruments = session.get_instruments_info(category="spot")
    tickers = session.get_tickers(category="spot")
    if instruments.get("retCode") != 0:
        raise RuntimeError(f"Bybit instruments error {instruments.get('retCode')}: {instruments.get('retMsg')}")
    if tickers.get("retCode") != 0:
        raise RuntimeError(f"Bybit tickers error {tickers.get('retCode')}: {tickers.get('retMsg')}")

    tradable = {
        item["symbol"]: item
        for item in instruments.get("result", {}).get("list", [])
        if item.get("quoteCoin") == "USDT"
        and item.get("status", "").lower() == "trading"
        and _safe_symbol(item.get("baseCoin", ""), item.get("symbol", ""))
    }

    candidates = []
    for ticker in tickers.get("result", {}).get("list", []):
        symbol = ticker.get("symbol", "")
        instrument = tradable.get(symbol)
        if not instrument:
            continue
        bid = _float(ticker.get("bid1Price"))
        ask = _float(ticker.get("ask1Price"))
        last = _float(ticker.get("lastPrice"))
        turnover = _float(ticker.get("turnover24h"))
        volume = _float(ticker.get("volume24h"))
        spread_pct = ((ask - bid) / last * 100) if bid > 0 and ask > 0 and last > 0 else 999.0
        if turnover < config.universe_min_turnover_24h or spread_pct > config.universe_max_spread_pct:
            continue
        score = _score(turnover, volume, spread_pct, _float(ticker.get("price24hPcnt")))
        candidates.append(
            {
                "symbol": symbol,
                "base": instrument.get("baseCoin"),
                "turnover24h": turnover,
                "volume24h": volume,
                "spread_pct": round(spread_pct, 5),
                "price24h_pct": round(_float(ticker.get("price24hPcnt")) * 100, 4),
                "last_price": last,
                "score": round(score, 4),
            }
        )

    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected = ranked[: (top_n or config.universe_top_n)]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "bybit_spot_public",
        "filters": {
            "quote": "USDT",
            "min_turnover_24h": config.universe_min_turnover_24h,
            "max_spread_pct": config.universe_max_spread_pct,
        },
        "selected_symbols": [item["symbol"] for item in selected],
        "selected": selected,
        "candidate_count": len(candidates),
    }
    save_universe(config, payload)
    return payload


def save_universe(config: AgentConfig, payload: dict[str, Any]) -> None:
    path = Path(config.universe_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_universe_symbols(config: AgentConfig) -> list[str]:
    path = Path(config.universe_path)
    if not path.exists():
        return list(config.training_symbols)
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = payload.get("selected_symbols") or []
    return [str(symbol).upper() for symbol in symbols] or list(config.training_symbols)


def _safe_symbol(base: str, symbol: str) -> bool:
    if "USD" in base:
        return False
    if base in STABLE_BASES:
        return False
    return not any(symbol.endswith(hint + "USDT") for hint in LEVERAGED_HINTS)


def _score(turnover: float, volume: float, spread_pct: float, change_24h: float) -> float:
    import math

    liquidity = math.log10(max(turnover, 1.0))
    spread_penalty = spread_pct * 12
    chaos_penalty = min(abs(change_24h * 100), 30) * 0.02
    return liquidity - spread_penalty - chaos_penalty + min(math.log10(max(volume, 1.0)), 8) * 0.15


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
