import unittest
from datetime import datetime, timedelta, timezone

from trading_agent.ml_model import build_dataset, predict_model, train_model
from trading_agent.models import Candle


def sample_candles(count: int = 140) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(count):
        price = 100 + i * 0.05 + ((i % 9) - 4) * 0.08
        candles.append(
            Candle(
                open_time=start + timedelta(hours=i),
                open=price * 0.999,
                high=price * 1.004,
                low=price * 0.996,
                close=price,
                volume=100 + i,
                close_time=start + timedelta(hours=i + 1),
                quote_volume=(100 + i) * price,
                trades=10,
            )
        )
    return candles


class MLModelTests(unittest.TestCase):
    def test_train_and_predict_model(self):
        candles = sample_candles()
        dataset = build_dataset(candles, lookback=60, horizon=4, move_threshold_pct=0.2)
        model, metrics = train_model(dataset, lookback=60, horizon=4, move_threshold_pct=0.2, epochs=4, learning_rate=0.01)
        prediction = predict_model(model, candles)
        self.assertGreater(metrics["train_samples"], 0)
        self.assertIn(prediction["scenario"], {"bearish", "sideways", "bullish"})
        total = sum(prediction["probabilities"].values())
        self.assertAlmostEqual(total, 100.0, places=1)

    def test_tp_sl_path_labels_use_future_candle_path(self):
        candles = sample_candles(90)
        entry_index = 60
        entry = candles[entry_index - 1].close
        candles[entry_index] = Candle(
            open_time=candles[entry_index].open_time,
            open=entry,
            high=entry * 1.01,
            low=entry * 0.999,
            close=entry * 1.005,
            volume=candles[entry_index].volume,
            close_time=candles[entry_index].close_time,
            quote_volume=candles[entry_index].quote_volume,
            trades=candles[entry_index].trades,
        )

        dataset = build_dataset(
            candles,
            lookback=60,
            horizon=4,
            move_threshold_pct=0.5,
            label_mode="tp_sl_path",
            take_profit_pct=0.5,
            stop_loss_pct=0.5,
        )

        self.assertEqual(dataset.label_mode, "tp_sl_path")
        self.assertEqual(dataset.y[0], 2)
        self.assertAlmostEqual(dataset.outcomes_pct[0], 0.5)

    def test_metrics_include_confusion_f1_and_expected_value(self):
        candles = sample_candles()
        dataset = build_dataset(
            candles,
            lookback=60,
            horizon=4,
            move_threshold_pct=0.2,
            label_mode="tp_sl_path",
        )
        model, metrics = train_model(
            dataset,
            lookback=60,
            horizon=4,
            move_threshold_pct=0.2,
            label_mode="tp_sl_path",
            epochs=4,
            learning_rate=0.01,
        )

        self.assertEqual(model.label_mode, "tp_sl_path")
        self.assertIn("macro_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("long_expected_value_pct", metrics)


if __name__ == "__main__":
    unittest.main()
