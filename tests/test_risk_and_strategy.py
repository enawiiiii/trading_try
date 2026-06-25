import unittest
from datetime import datetime, timezone

from trading_agent.config import AgentConfig
from trading_agent.market import analyze_market
from trading_agent.models import Action, Candle, MarketState, StrategyName
from trading_agent.risk import RiskContext, apply_risk_rules
from trading_agent.strategies import StrategySignal, select_signal, select_strategy


def state(**overrides):
    base = {
        "trend": "bullish",
        "volatility": "medium",
        "liquidity": "high",
        "liquidity_score": 80,
        "structure": "trend continuation",
        "fake_breakout_probability": 25,
        "current_price": 100.0,
        "atr_pct": 2.0,
        "rsi": 55.0,
        "support": 95.0,
        "resistance": 105.0,
        "bullish_probability": 45,
        "bearish_probability": 25,
        "sideways_probability": 30,
        "evidence_summary": "test evidence",
        "explanation": "test state",
    }
    base.update(overrides)
    return MarketState(**base)


class StrategySelectionTests(unittest.TestCase):
    def test_selects_breakout_only_when_fakeout_risk_is_controlled(self):
        selected = select_strategy(state(structure="breakout", fake_breakout_probability=40))
        self.assertEqual(selected, StrategyName.BREAKOUT)

    def test_avoids_breakout_for_noisy_breakout(self):
        selected = select_strategy(state(structure="breakout", fake_breakout_probability=70))
        self.assertNotEqual(selected, StrategyName.BREAKOUT)

    def test_weights_can_prefer_valid_alternative_without_overriding_risk(self):
        selected = select_strategy(
            state(structure="range", trend="sideways"),
            {
                StrategyName.BREAKOUT.value: 1.3,
                StrategyName.TREND_FOLLOWING.value: 1.0,
                StrategyName.MEAN_REVERSION.value: 0.7,
            },
        )
        self.assertEqual(selected, StrategyName.MEAN_REVERSION)

    def test_bearish_market_without_position_does_not_select_sell_strategy_as_active(self):
        signal = select_signal(
            state(
                trend="bearish",
                structure="trend continuation",
                bullish_probability=20,
                bearish_probability=55,
                sideways_probability=25,
            ),
            has_position=False,
        )

        self.assertNotEqual(signal.action, Action.SELL)

    def test_bearish_market_with_position_can_select_exit_signal(self):
        signal = select_signal(
            state(
                trend="bearish",
                structure="trend continuation",
                bullish_probability=20,
                bearish_probability=55,
                sideways_probability=25,
            ),
            has_position=True,
        )

        self.assertEqual(signal.action, Action.SELL)


class RiskRuleTests(unittest.TestCase):
    def test_daily_loss_limit_blocks_trade(self):
        signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 85, "confirmed")
        risk = apply_risk_rules(
            signal,
            state(),
            AgentConfig(),
            RiskContext(daily_pnl_pct=-3.2, consecutive_losses=0, has_position=False),
        )
        self.assertEqual(risk.action, Action.NO_TRADE)
        self.assertEqual(risk.risk_level, "high")

    def test_low_liquidity_blocks_trade(self):
        signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 85, "confirmed")
        risk = apply_risk_rules(
            signal,
            state(liquidity="low", liquidity_score=20),
            AgentConfig(),
            RiskContext(),
        )
        self.assertEqual(risk.action, Action.NO_TRADE)

    def test_sell_without_position_is_rejected(self):
        signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.SELL, 80, "exit")
        risk = apply_risk_rules(signal, state(), AgentConfig(), RiskContext(has_position=False))
        self.assertEqual(risk.action, Action.NO_TRADE)

    def test_buy_requires_probability_edge(self):
        signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 85, "confirmed")
        risk = apply_risk_rules(
            signal,
            state(bullish_probability=42, bearish_probability=20, sideways_probability=38),
            AgentConfig(),
            RiskContext(),
        )
        self.assertEqual(risk.action, Action.NO_TRADE)


class MarketAnalysisTests(unittest.TestCase):
    def test_liquidity_uses_volume_when_trade_count_is_unavailable(self):
        candles = [
            Candle(
                open_time=datetime(2026, 1, 1, hour=i % 24, tzinfo=timezone.utc),
                open=100 + i * 0.05,
                high=101 + i * 0.05,
                low=99 + i * 0.05,
                close=100.5 + i * 0.05,
                volume=10,
                close_time=datetime(2026, 1, 1, hour=i % 24, tzinfo=timezone.utc),
                quote_volume=1000,
                trades=0,
            )
            for i in range(60)
        ]
        state = analyze_market(candles)
        self.assertGreaterEqual(state.liquidity_score, 90)

    def test_market_analysis_outputs_scenario_probabilities(self):
        candles = [
            Candle(
                open_time=datetime(2026, 1, 1, hour=i % 24, tzinfo=timezone.utc),
                open=100 + i,
                high=101 + i,
                low=99 + i,
                close=100.8 + i,
                volume=10,
                close_time=datetime(2026, 1, 1, hour=i % 24, tzinfo=timezone.utc),
                quote_volume=1000 + i * 10,
                trades=0,
            )
            for i in range(60)
        ]
        state = analyze_market(candles)
        total = state.bullish_probability + state.bearish_probability + state.sideways_probability
        self.assertEqual(total, 100)
        self.assertGreaterEqual(state.bullish_probability, state.bearish_probability)
        self.assertTrue(state.evidence_summary)


if __name__ == "__main__":
    unittest.main()
