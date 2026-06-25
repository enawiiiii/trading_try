import unittest
from datetime import datetime, timedelta, timezone

from trading_agent.config import AgentConfig
from trading_agent.models import Candle
from trading_agent.recovery_shadow import _evaluate_open_trade, _has_open_trade, _summarize, evaluate_recovery_promotion


class RecoveryShadowTests(unittest.TestCase):
    def test_has_open_trade_detects_existing_symbol_interval_strategy(self):
        state = {
            "trades": [
                {
                    "status": "open",
                    "symbol": "XRPUSDT",
                    "interval": "1h",
                    "strategy": "Mean Reversion Strategy",
                }
            ]
        }

        self.assertTrue(_has_open_trade(state, "XRPUSDT", "1h", "Mean Reversion Strategy"))
        self.assertFalse(_has_open_trade(state, "ETHUSDT", "1h", "Mean Reversion Strategy"))

    def test_summarize_shadow_trades(self):
        summary = _summarize(
            [
                {"status": "open", "unrealized_pct": 0.4},
                {"status": "closed", "realized_pct": 2.3, "outcome": "take_profit"},
                {"status": "closed", "realized_pct": -1.3, "outcome": "stop_loss"},
            ]
        )

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["open"], 1)
        self.assertEqual(summary["closed"], 2)
        self.assertEqual(summary["win_rate_pct"], 50.0)
        self.assertEqual(summary["avg_unrealized_pct"], 0.4)

    def test_promotion_requires_enough_profitable_closed_trades(self):
        trades = [
            {"status": "closed", "realized_pct": 2.3},
            {"status": "closed", "realized_pct": 2.3},
            {"status": "closed", "realized_pct": 2.3},
            {"status": "closed", "realized_pct": -1.3},
            {"status": "closed", "realized_pct": 0.5},
        ]

        promotion = evaluate_recovery_promotion(trades)

        self.assertEqual(promotion["status"], "ELIGIBLE_FOR_PAPER_ALLOW_SMALL")
        self.assertFalse(promotion["execution_enabled"])

    def test_promotion_rejects_small_sample(self):
        promotion = evaluate_recovery_promotion([{"status": "closed", "realized_pct": 2.3}])

        self.assertEqual(promotion["status"], "NOT_ELIGIBLE")
        self.assertFalse(promotion["criteria"]["min_closed_trades"]["passed"])

    def test_open_trade_closes_immediately_on_stop_loss_before_horizon(self):
        opened_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        trade = {
            "status": "open",
            "opened_at": opened_at.isoformat(),
            "entry_price": 100.0,
            "interval": "1h",
        }
        candles = [
            Candle(
                open_time=opened_at,
                close_time=opened_at + timedelta(hours=1),
                open=100.0,
                high=100.5,
                low=98.0,
                close=98.5,
                volume=1,
                quote_volume=1,
                trades=1,
            )
        ]

        result = _evaluate_open_trade(AgentConfig(), trade, candles)

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "closed")
        self.assertEqual(result["outcome"], "stop_loss")


if __name__ == "__main__":
    unittest.main()
