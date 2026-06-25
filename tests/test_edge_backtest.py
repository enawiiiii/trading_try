import unittest
from datetime import datetime, timedelta, timezone

from trading_agent.edge_backtest import _empty_summary, _finalize_summary, _long_outcome, _merge_summary
from trading_agent.models import Candle


class EdgeBacktestTests(unittest.TestCase):
    def test_long_outcome_uses_tp_before_timeout(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        future = [
            Candle(
                open_time=start,
                open=100,
                high=103,
                low=99.5,
                close=102,
                volume=1,
                close_time=start + timedelta(hours=1),
                quote_volume=100,
                trades=1,
            )
        ]
        result = _long_outcome(100, future, take_profit_pct=2.0, stop_loss_pct=1.0, fee_pct=0.1)
        self.assertEqual(result["outcome"], "take_profit")
        self.assertAlmostEqual(result["realized_pct"], 1.9)

    def test_merge_summary_aggregates_exploration_long_events(self):
        total = _empty_summary()
        _merge_summary(
            total,
            {
                "events": 2,
                "exploration_long_events": 2,
                "exploration_long_avg_realized_pct": 1.0,
                "exploration_long_win_rate_pct": 50.0,
            },
        )
        _finalize_summary(total)

        self.assertEqual(total["exploration_long_events"], 2)
        self.assertEqual(total["exploration_long_avg_realized_pct"], 1.0)
        self.assertEqual(total["exploration_long_win_rate_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
