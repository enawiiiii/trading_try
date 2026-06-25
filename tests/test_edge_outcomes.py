import unittest

from trading_agent.edge_outcomes import _is_observed_buy_setup, _summarize


class EdgeOutcomeTests(unittest.TestCase):
    def test_summarize_evaluated_and_pending_rows(self):
        summary = _summarize(
            [
                {"status": "evaluated", "realized_pct": 0.25, "outcome": "take_profit"},
                {"status": "evaluated", "realized_pct": -1.3, "outcome": "stop_loss"},
                {"status": "pending"},
            ]
        )
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["evaluated"], 2)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["win_rate_pct"], 50.0)
        self.assertAlmostEqual(summary["avg_realized_pct"], -0.525)

    def test_signal_buy_counts_even_when_risk_rejects_execution(self):
        self.assertTrue(_is_observed_buy_setup({"signal_action": "BUY", "final_action": "NO TRADE"}))
        self.assertTrue(_is_observed_buy_setup({"signal_action": "NO TRADE", "final_action": "BUY"}))
        self.assertFalse(_is_observed_buy_setup({"signal_action": "SELL", "final_action": "NO TRADE"}))


if __name__ == "__main__":
    unittest.main()
