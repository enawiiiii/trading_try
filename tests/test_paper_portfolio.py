import unittest
from datetime import datetime, timezone

from trading_agent.agent import paper_execute
from trading_agent.config import AgentConfig
from trading_agent.models import Action, Decision, MarketState, StrategyName
from trading_agent.portfolio import PaperPortfolio


def market_state(price: float) -> MarketState:
    return MarketState(
        trend="bullish",
        volatility="low",
        liquidity="high",
        liquidity_score=80,
        structure="trend continuation",
        fake_breakout_probability=20,
        current_price=price,
        atr_pct=1.0,
        rsi=55,
        support=price * 0.98,
        resistance=price * 1.02,
        bullish_probability=55,
        bearish_probability=20,
        sideways_probability=25,
        evidence_summary="test",
        explanation="test",
    )


def buy_decision(symbol: str, price: float) -> Decision:
    return Decision(
        action=Action.BUY,
        confidence=80,
        selected_strategy=StrategyName.TREND_FOLLOWING,
        market_state="test",
        reasoning="test",
        risk_level="low",
        learning_note="test",
        symbol=symbol,
        price=price,
        timestamp=datetime.now(timezone.utc),
    )


class PaperPortfolioTests(unittest.TestCase):
    def test_repeated_buys_accumulate_cost_basis(self):
        portfolio = PaperPortfolio(cash=1000.0, positions={}, last_prices={}, cost_basis={}, peak_prices={})
        config = AgentConfig(risk_per_trade_pct=1.0)

        first = paper_execute(buy_decision("BTCUSDT", 100.0), portfolio, market_state(100.0), config)
        second = paper_execute(buy_decision("BTCUSDT", 100.0), portfolio, market_state(100.0), config)

        self.assertTrue(first["executed"])
        self.assertTrue(second["executed"])
        self.assertAlmostEqual(portfolio.cost_basis["BTCUSDT"], first["cost"] + second["cost"])
        self.assertAlmostEqual(portfolio.positions["BTCUSDT"] * 100.0, portfolio.cost_basis["BTCUSDT"])


if __name__ == "__main__":
    unittest.main()
