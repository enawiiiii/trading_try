import unittest

from trading_agent.signal_quality import pocket_rule_score, signal_quality_score


class SignalQualityTests(unittest.TestCase):
    def test_signal_quality_score_rewards_positive_edge(self):
        score = signal_quality_score({"buy_events": 50, "avg_realized_pct": 0.3, "win_rate_pct": 48.0})
        self.assertGreater(score, 0)

    def test_signal_quality_score_penalizes_weak_edge(self):
        score = signal_quality_score({"buy_events": 50, "avg_realized_pct": -0.2, "win_rate_pct": 25.0})
        self.assertLess(score, 0)

    def test_signal_quality_score_ignores_small_samples(self):
        score = signal_quality_score({"buy_events": 5, "avg_realized_pct": 2.0, "win_rate_pct": 80.0})
        self.assertEqual(score, 0)

    def test_pocket_rule_score_requires_specific_positive_edge(self):
        score = pocket_rule_score({"buy_events": 25, "avg_realized_pct": 0.24, "win_rate_pct": 45.0})
        self.assertGreater(score, 0)

    def test_pocket_rule_score_penalizes_specific_weak_edge(self):
        score = pocket_rule_score({"buy_events": 25, "avg_realized_pct": -0.2, "win_rate_pct": 44.0})
        self.assertLess(score, 0)


if __name__ == "__main__":
    unittest.main()
