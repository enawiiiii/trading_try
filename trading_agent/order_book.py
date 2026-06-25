from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    quantity: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]


@dataclass(frozen=True)
class OrderBookAnalysis:
    symbol: str
    best_bid: float
    best_ask: float
    spread_pct: float
    bid_depth_quote: float
    ask_depth_quote: float
    depth_imbalance_pct: float
    pressure: str
    liquidity_quality: str
    estimated_slippage_pct_25_usdt: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread_pct": self.spread_pct,
            "bid_depth_quote": self.bid_depth_quote,
            "ask_depth_quote": self.ask_depth_quote,
            "depth_imbalance_pct": self.depth_imbalance_pct,
            "pressure": self.pressure,
            "liquidity_quality": self.liquidity_quality,
            "estimated_slippage_pct_25_usdt": self.estimated_slippage_pct_25_usdt,
            "note": self.note,
        }


def fetch_bybit_order_book(symbol: str, limit: int = 50, testnet: bool = False) -> OrderBookSnapshot:
    try:
        from pybit.unified_trading import HTTP
    except ImportError as exc:
        raise RuntimeError("pybit is not installed. Run: python -m pip install -r requirements.txt") from exc

    session = HTTP(testnet=testnet)
    response = session.get_orderbook(category="spot", symbol=symbol.upper(), limit=max(1, min(limit, 200)))
    if response.get("retCode") != 0:
        raise RuntimeError(f"Bybit order book error {response.get('retCode')}: {response.get('retMsg')}")

    result = response.get("result", {})
    return OrderBookSnapshot(
        symbol=symbol.upper(),
        bids=_levels(result.get("b", []), reverse=True),
        asks=_levels(result.get("a", []), reverse=False),
    )


def analyze_order_book(snapshot: OrderBookSnapshot, quote_amount: float = 25.0) -> OrderBookAnalysis:
    if not snapshot.bids or not snapshot.asks:
        raise ValueError("Order book requires at least one bid and one ask.")

    best_bid = snapshot.bids[0].price
    best_ask = snapshot.asks[0].price
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
    spread_pct = ((best_ask - best_bid) / mid) * 100 if mid else 0.0
    bid_depth = _quote_depth(snapshot.bids)
    ask_depth = _quote_depth(snapshot.asks)
    total_depth = bid_depth + ask_depth
    imbalance = ((bid_depth - ask_depth) / total_depth) * 100 if total_depth else 0.0
    slippage = _estimate_buy_slippage(snapshot.asks, quote_amount)
    pressure = _pressure(imbalance)
    quality = _liquidity_quality(spread_pct, bid_depth, ask_depth, slippage)

    return OrderBookAnalysis(
        symbol=snapshot.symbol,
        best_bid=round(best_bid, 8),
        best_ask=round(best_ask, 8),
        spread_pct=round(spread_pct, 4),
        bid_depth_quote=round(bid_depth, 2),
        ask_depth_quote=round(ask_depth, 2),
        depth_imbalance_pct=round(imbalance, 2),
        pressure=pressure,
        liquidity_quality=quality,
        estimated_slippage_pct_25_usdt=round(slippage, 4),
        note=_note(quality, pressure, spread_pct, slippage),
    )


def _levels(rows: list[list[str]], reverse: bool) -> list[OrderBookLevel]:
    levels = [OrderBookLevel(price=float(row[0]), quantity=float(row[1])) for row in rows]
    return sorted(levels, key=lambda level: level.price, reverse=reverse)


def _quote_depth(levels: list[OrderBookLevel]) -> float:
    return sum(level.price * level.quantity for level in levels)


def _estimate_buy_slippage(asks: list[OrderBookLevel], quote_amount: float) -> float:
    if not asks or quote_amount <= 0:
        return 0.0
    best_ask = asks[0].price
    remaining = quote_amount
    base_bought = 0.0
    spent = 0.0
    for level in asks:
        available_quote = level.price * level.quantity
        take_quote = min(remaining, available_quote)
        base_bought += take_quote / level.price
        spent += take_quote
        remaining -= take_quote
        if remaining <= 0:
            break
    if spent <= 0 or base_bought <= 0:
        return 0.0
    avg_price = spent / base_bought
    return ((avg_price - best_ask) / best_ask) * 100 if best_ask else 0.0


def _pressure(imbalance: float) -> str:
    if imbalance >= 18:
        return "buy support"
    if imbalance <= -18:
        return "sell wall"
    return "balanced"


def _liquidity_quality(spread_pct: float, bid_depth: float, ask_depth: float, slippage: float) -> str:
    near_depth = min(bid_depth, ask_depth)
    if spread_pct <= 0.04 and near_depth >= 50_000 and slippage <= 0.02:
        return "strong"
    if spread_pct <= 0.12 and near_depth >= 5_000 and slippage <= 0.08:
        return "acceptable"
    return "weak"


def _note(quality: str, pressure: str, spread_pct: float, slippage: float) -> str:
    return (
        f"Depth quality is {quality}; book pressure is {pressure}; "
        f"spread is {spread_pct:.4f}% and estimated 25 USDT buy slippage is {slippage:.4f}%."
    )
