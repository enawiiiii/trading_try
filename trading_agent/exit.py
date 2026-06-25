from __future__ import annotations

from dataclasses import asdict, dataclass

from trading_agent.config import AgentConfig
from trading_agent.market import analyze_market
from trading_agent.models import Candle
from trading_agent.portfolio import PaperPortfolio


@dataclass(frozen=True)
class ExitDecision:
    action: str
    confidence: int
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    risk_level: str
    reason: str
    evidence: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_exit(symbol: str, candles: list[Candle], portfolio: PaperPortfolio, config: AgentConfig) -> ExitDecision:
    symbol = symbol.upper()
    quantity = portfolio.positions.get(symbol, 0.0)
    state = analyze_market(candles)
    current_price = state.current_price
    portfolio.last_prices[symbol] = current_price

    if quantity <= 0:
        return ExitDecision(
            action="HOLD",
            confidence=0,
            symbol=symbol,
            quantity=0.0,
            entry_price=0.0,
            current_price=current_price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            risk_level="low",
            reason="No open spot position to exit.",
            evidence=[],
        )

    cost_basis = portfolio.cost_basis.get(symbol, quantity * current_price)
    entry_price = cost_basis / quantity if quantity else current_price
    unrealized_pnl = quantity * current_price - cost_basis
    unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis else 0.0
    peak_price = max(portfolio.peak_prices.get(symbol, entry_price), current_price)
    portfolio.peak_prices[symbol] = peak_price
    drawdown_from_peak_pct = ((peak_price - current_price) / peak_price) * 100 if peak_price else 0.0

    evidence: list[str] = []
    confidence = 35

    if unrealized_pnl_pct <= -abs(config.stop_loss_pct):
        evidence.append(f"Stop loss hit: unrealized PnL {unrealized_pnl_pct:.2f}% <= -{config.stop_loss_pct:.2f}%.")
        return _sell(symbol, quantity, entry_price, current_price, unrealized_pnl, unrealized_pnl_pct, "high", 92, evidence)

    if unrealized_pnl_pct >= config.take_profit_pct:
        evidence.append(f"Take profit hit: unrealized PnL {unrealized_pnl_pct:.2f}% >= {config.take_profit_pct:.2f}%.")
        return _sell(symbol, quantity, entry_price, current_price, unrealized_pnl, unrealized_pnl_pct, "medium", 86, evidence)

    if unrealized_pnl_pct > 0 and drawdown_from_peak_pct >= config.trailing_stop_pct:
        evidence.append(f"Trailing stop hit: price pulled back {drawdown_from_peak_pct:.2f}% from peak.")
        return _sell(symbol, quantity, entry_price, current_price, unrealized_pnl, unrealized_pnl_pct, "medium", 84, evidence)

    if current_price < state.support:
        evidence.append("Price broke below recent support.")
        confidence += 22
    if state.rsi > 72 and current_price >= state.resistance * 0.992:
        evidence.append("RSI is overheated while price is near resistance.")
        confidence += 16
    if state.bearish_probability >= config.min_exit_bearish_probability:
        evidence.append(f"Bearish probability is {state.bearish_probability}% after calibration.")
        confidence += 18
    if state.structure == "breakout" and state.fake_breakout_probability >= 55:
        evidence.append("Breakout quality weakened and fake-move risk is elevated.")
        confidence += 14
    if _bearish_candle_rejection(candles):
        evidence.append("Latest candles show bearish rejection from highs.")
        confidence += 12

    if confidence >= 72 and evidence:
        return _sell(
            symbol,
            quantity,
            entry_price,
            current_price,
            unrealized_pnl,
            unrealized_pnl_pct,
            "medium" if unrealized_pnl_pct >= 0 else "high",
            min(confidence, 90),
            evidence,
        )

    hold_evidence = evidence or ["No exit trigger is strong enough; holding position."]
    return ExitDecision(
        action="HOLD",
        confidence=min(confidence, 70),
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        current_price=current_price,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        risk_level="low" if unrealized_pnl_pct >= -0.5 else "medium",
        reason="Exit conditions are not confirmed enough for a disciplined sell.",
        evidence=hold_evidence,
    )


def _sell(
    symbol: str,
    quantity: float,
    entry_price: float,
    current_price: float,
    unrealized_pnl: float,
    unrealized_pnl_pct: float,
    risk_level: str,
    confidence: int,
    evidence: list[str],
) -> ExitDecision:
    return ExitDecision(
        action="SELL",
        confidence=confidence,
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        current_price=current_price,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        risk_level=risk_level,
        reason="Exit engine found confirmed sell conditions.",
        evidence=evidence,
    )


def _bearish_candle_rejection(candles: list[Candle]) -> bool:
    if len(candles) < 3:
        return False
    last = candles[-1]
    previous = candles[-2]
    upper_wick = last.high - max(last.open, last.close)
    body = abs(last.close - last.open)
    bearish_close = last.close < last.open and last.close < previous.close
    return bearish_close and upper_wick > body * 1.2
