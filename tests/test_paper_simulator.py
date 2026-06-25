import unittest
from datetime import datetime, timedelta, timezone

from trading_agent.config import AgentConfig
from trading_agent.models import Candle
from trading_agent.paper_simulator import _evaluate_open_trade, _lane_guard, _open_from_edge_events, _paper_lane, _summarize


class PaperSimulatorTests(unittest.TestCase):
    def test_rejected_buy_with_enough_score_becomes_paper_lane(self):
        event = {
            "signal_action": "BUY",
            "edge_filter": {"monitor_action": "BLOCK", "score": 19},
        }

        self.assertIsNone(_paper_lane(event, min_rejected_score=15))

    def test_sell_signal_is_not_paper_trade(self):
        event = {
            "signal_action": "SELL",
            "edge_filter": {"monitor_action": "BLOCK", "score": 99},
        }

        self.assertIsNone(_paper_lane(event, min_rejected_score=15))

    def test_no_trade_observe_can_be_exploration_paper_trade(self):
        event = {
            "symbol": "ASTERUSDT",
            "interval": "4h",
            "signal_action": "NO TRADE",
            "edge_filter": {"monitor_action": "OBSERVE", "score": 42, "estimated_ev_pct": -0.08},
        }

        self.assertEqual(_paper_lane(event, min_rejected_score=15), "EXPLORATION_LONG_PAPER")

    def test_exploration_rejects_unapproved_pocket(self):
        event = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "signal_action": "NO TRADE",
            "edge_filter": {"monitor_action": "OBSERVE", "score": 42, "estimated_ev_pct": -0.08},
        }

        self.assertIsNone(_paper_lane(event, min_rejected_score=15))

    def test_exploration_rejects_deep_negative_ev(self):
        event = {
            "symbol": "ASTERUSDT",
            "interval": "4h",
            "signal_action": "NO TRADE",
            "edge_filter": {"monitor_action": "OBSERVE", "score": 42, "estimated_ev_pct": -0.2},
        }

        self.assertIsNone(_paper_lane(event, min_rejected_score=15))

    def test_summarize_paper_trades(self):
        summary = _summarize(
            [
                {"status": "open", "unrealized_pct": 0.5, "unrealized_quote": 0.05},
                {"status": "closed", "realized_pct": 2.3, "realized_quote": 0.23, "outcome": "take_profit"},
                {"status": "closed", "realized_pct": -1.3, "realized_quote": -0.13, "outcome": "stop_loss"},
            ]
        )

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["open"], 1)
        self.assertEqual(summary["closed"], 2)
        self.assertEqual(summary["win_rate_pct"], 50.0)
        self.assertAlmostEqual(summary["realized_quote"], 0.1)

    def test_open_trade_closes_on_take_profit_before_horizon(self):
        opened_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        trade = {
            "status": "open",
            "opened_at": opened_at.isoformat(),
            "entry_price": 100.0,
            "interval": "1h",
            "quote_amount": 10.0,
        }
        candles = [
            Candle(
                open_time=opened_at,
                close_time=opened_at + timedelta(hours=1),
                open=100.0,
                high=103.0,
                low=99.8,
                close=102.0,
                volume=1,
                quote_volume=1,
                trades=1,
            )
        ]

        result = _evaluate_open_trade(AgentConfig(), trade, candles)

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "closed")
        self.assertEqual(result["outcome"], "take_profit")
        self.assertGreater(result["realized_quote"], 0)

    def test_open_from_events_respects_state_started_at(self):
        started_at = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        old_event = {
            "timestamp": (started_at - timedelta(minutes=5)).isoformat(),
            "symbol": "XPLUSDT",
            "interval": "1h",
            "strategy": "Breakout Strategy",
            "signal_action": "BUY",
            "final_action": "BUY",
            "confidence": 84,
            "price": 0.1,
            "edge_filter": {"monitor_action": "BLOCK", "score": 20},
        }
        state = {"started_at": started_at.isoformat(), "trades": []}

        import trading_agent.paper_simulator as paper_simulator

        original = paper_simulator.recent_edge_events
        paper_simulator.recent_edge_events = lambda config, limit=5000: [old_event]
        try:
            opened = _open_from_edge_events(
                AgentConfig(edge_events_path="unused"),
                state,
                now=started_at + timedelta(minutes=1),
                source_hours=6,
                max_new=12,
                min_rejected_score=15,
            )
        finally:
            paper_simulator.recent_edge_events = original

        self.assertEqual(opened, [])

    def test_lane_guard_blocks_three_recent_losses(self):
        trades = [
            {"lane": "EXPLORATION_LONG_PAPER", "status": "closed", "realized_pct": -1.3},
            {"lane": "EXPLORATION_LONG_PAPER", "status": "closed", "realized_pct": -1.3},
            {"lane": "EXPLORATION_LONG_PAPER", "status": "closed", "realized_pct": -1.3},
        ]

        guard = _lane_guard(trades, "EXPLORATION_LONG_PAPER")

        self.assertTrue(guard["blocked"])
        self.assertEqual(guard["recent_losses"], 3)

    def test_lane_guard_allows_after_win_resets_recent_losses(self):
        trades = [
            {"lane": "EXPLORATION_LONG_PAPER", "status": "closed", "realized_pct": -1.3},
            {"lane": "EXPLORATION_LONG_PAPER", "status": "closed", "realized_pct": 2.3},
        ]

        guard = _lane_guard(trades, "EXPLORATION_LONG_PAPER")

        self.assertFalse(guard["blocked"])
        self.assertEqual(guard["recent_losses"], 0)


if __name__ == "__main__":
    unittest.main()
