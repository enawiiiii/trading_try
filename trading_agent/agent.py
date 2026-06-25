from __future__ import annotations

from datetime import datetime, timezone

from trading_agent.calibration import apply_calibration, load_calibration
from trading_agent.config import AgentConfig
from trading_agent.edge_filter import append_edge_event, evaluate_edge_filter
from trading_agent.learning import learning_note, load_learning_state
from trading_agent.market import analyze_market
from trading_agent.models import Action, Candle, Decision
from trading_agent.risk import RiskContext, apply_risk_rules, position_size
from trading_agent.strategies import select_signal


def decide(symbol: str, candles: list[Candle], config: AgentConfig, context: RiskContext, interval: str = "") -> Decision:
    state = analyze_market(candles)
    state = apply_calibration(state, load_calibration(config), symbol, interval)
    learning_state = load_learning_state(config)
    signal = select_signal(state, learning_state.strategy_weights, has_position=context.has_position)
    strategy = signal.strategy
    risk = apply_risk_rules(signal, state, config, context)
    edge = evaluate_edge_filter(
        symbol=symbol,
        interval=interval or "1h",
        state=state,
        signal=signal,
        risk=risk,
        config=config,
    )

    reasoning = (
        f"{signal.reasoning} {risk.reason} "
        f"Scenario view: bullish {state.bullish_probability}%, bearish {state.bearish_probability}%, "
        f"sideways {state.sideways_probability}%. "
        f"Top evidence: {state.evidence_summary}. "
        "This agent only acts when confirmation, probabilities, liquidity, and risk limits agree."
    )

    decision = Decision(
        action=risk.action,
        confidence=risk.confidence,
        selected_strategy=strategy,
        market_state=state.explanation,
        reasoning=reasoning,
        risk_level=risk.risk_level,
        learning_note=learning_note(learning_state, strategy),
        symbol=symbol.upper(),
        price=state.current_price,
        timestamp=datetime.now(timezone.utc),
        edge_filter=edge.to_dict(),
    )
    append_edge_event(
        config,
        {
            "timestamp": decision.timestamp,
            "symbol": decision.symbol,
            "interval": interval or "1h",
            "strategy": strategy.value,
            "signal_action": signal.action.value,
            "final_action": decision.action.value,
            "confidence": decision.confidence,
            "price": decision.price,
            "edge_filter": edge.to_dict(),
        },
    )
    return decision


def paper_execute(decision: Decision, portfolio, state, config: AgentConfig) -> dict:
    symbol = decision.symbol.upper()
    portfolio.last_prices[symbol] = decision.price

    if decision.action == Action.BUY:
        quantity = position_size(portfolio.cash, decision.price, state, config)
        cost = quantity * decision.price
        if quantity <= 0 or cost > portfolio.cash:
            return {"executed": False, "reason": "Insufficient cash or zero position size."}
        portfolio.cash -= cost
        portfolio.positions[symbol] = portfolio.positions.get(symbol, 0.0) + quantity
        portfolio.cost_basis[symbol] = portfolio.cost_basis.get(symbol, 0.0) + cost
        portfolio.peak_prices[symbol] = max(portfolio.peak_prices.get(symbol, decision.price), decision.price)
        return {"executed": True, "side": "BUY", "quantity": quantity, "price": decision.price, "cost": cost}

    if decision.action == Action.SELL:
        quantity = portfolio.positions.get(symbol, 0.0)
        if quantity <= 0:
            return {"executed": False, "reason": "No paper position to sell."}
        proceeds = quantity * decision.price
        cost_basis = portfolio.cost_basis.get(symbol, 0.0)
        realized_pnl = proceeds - cost_basis
        realized_pnl_pct = (realized_pnl / cost_basis) * 100 if cost_basis else 0.0
        portfolio.cash += proceeds
        portfolio.positions[symbol] = 0.0
        portfolio.cost_basis[symbol] = 0.0
        portfolio.peak_prices[symbol] = 0.0
        portfolio.consecutive_losses = portfolio.consecutive_losses + 1 if realized_pnl < 0 else 0
        return {
            "executed": True,
            "side": "SELL",
            "quantity": quantity,
            "price": decision.price,
            "proceeds": proceeds,
            "realized_pnl": realized_pnl,
            "realized_pnl_pct": realized_pnl_pct,
        }

    return {"executed": False, "reason": "No trade decision."}
