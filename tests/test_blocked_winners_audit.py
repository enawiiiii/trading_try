import unittest

from trading_agent.blocked_winners_audit import _best_pockets, _summarize


class BlockedWinnersAuditTests(unittest.TestCase):
    def test_best_pockets_keeps_positive_evaluated_pockets(self):
        rows = [
            {
                "status": "evaluated",
                "symbol_interval_strategy_regime": "BTCUSDT_1h|Breakout Strategy|trending",
                "realized_pct": 2.3,
                "outcome": "take_profit",
            },
            {
                "status": "evaluated",
                "symbol_interval_strategy_regime": "BTCUSDT_1h|Breakout Strategy|trending",
                "realized_pct": -1.3,
                "outcome": "stop_loss",
            },
            {
                "status": "evaluated",
                "symbol_interval_strategy_regime": "BTCUSDT_1h|Breakout Strategy|trending",
                "realized_pct": 2.3,
                "outcome": "take_profit",
            },
            {
                "status": "evaluated",
                "symbol_interval_strategy_regime": "ETHUSDT_1h|Trend Following Strategy|trending",
                "realized_pct": -1.3,
                "outcome": "stop_loss",
            },
        ]

        pockets = _best_pockets(rows, min_evaluated=3)

        self.assertEqual(len(pockets), 1)
        self.assertEqual(pockets[0]["key"], "BTCUSDT_1h|Breakout Strategy|trending")
        self.assertEqual(pockets[0]["evaluated"], 3)
        self.assertGreater(pockets[0]["avg_realized_pct"], 0)

    def test_summarize_counts_pending_and_evaluated(self):
        summary = _summarize(
            [
                {"status": "evaluated", "realized_pct": 2.3, "outcome": "take_profit"},
                {"status": "pending"},
            ]
        )

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["evaluated"], 1)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["win_rate_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
