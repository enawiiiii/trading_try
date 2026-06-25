from __future__ import annotations

import os
import time
from decimal import Decimal, ROUND_DOWN
from uuid import uuid4

from trading_agent.config import AgentConfig
from trading_agent.models import Action, Decision


class LiveTradingBlocked(RuntimeError):
    pass


def place_spot_market_order(decision: Decision, quote_amount: float, config: AgentConfig) -> dict:
    if not config.live_trading_enabled:
        raise LiveTradingBlocked("Live trading is disabled in config.")
    if os.getenv("ENABLE_LIVE_TRADING", "").lower() != "true":
        raise LiveTradingBlocked("Set ENABLE_LIVE_TRADING=true before live orders are allowed.")
    if decision.action not in {Action.BUY, Action.SELL}:
        raise LiveTradingBlocked("Decision is not BUY or SELL.")
    if quote_amount <= 0 or quote_amount > config.max_live_order_quote:
        raise LiveTradingBlocked("Quote amount is outside configured live order limits.")

    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        raise LiveTradingBlocked("Missing BYBIT_API_KEY or BYBIT_API_SECRET environment variables.")

    try:
        from pybit.unified_trading import HTTP
        import pybit._helpers as pybit_helpers
    except ImportError as exc:
        raise RuntimeError("pybit is not installed. Run: python -m pip install -r requirements.txt") from exc

    if config.bybit_time_offset_ms:
        pybit_helpers.generate_timestamp = lambda: int(time.time() * 1000) + int(config.bybit_time_offset_ms)

    session = HTTP(
        testnet=config.bybit_testnet,
        demo=config.bybit_demo,
        api_key=api_key,
        api_secret=api_secret,
    )

    side = "Buy" if decision.action == Action.BUY else "Sell"
    qty = _format_decimal(quote_amount if decision.action == Action.BUY else quote_amount / decision.price)
    order_link_id = f"spot-agent-{uuid4().hex[:16]}"

    return session.place_order(
        category="spot",
        symbol=decision.symbol,
        side=side,
        orderType="Market",
        qty=qty,
        timeInForce="IOC",
        orderLinkId=order_link_id,
        isLeverage=0,
        orderFilter="Order",
    )


def _format_decimal(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN).normalize())
