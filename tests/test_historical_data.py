import unittest
from datetime import datetime, timedelta, timezone

from trading_agent.data import _dedupe_sorted_closed
from trading_agent.models import Candle


def candle(hour: int, close: float) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hour)
    return Candle(
        open_time=start,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1,
        close_time=start + timedelta(hours=1),
        quote_volume=close,
        trades=1,
    )


class HistoricalDataTests(unittest.TestCase):
    def test_dedupe_sorted_closed_keeps_latest_unique_candles(self):
        candles = [candle(2, 102), candle(1, 101), candle(2, 202), candle(0, 100)]
        result = _dedupe_sorted_closed(candles)
        self.assertEqual([item.open_time.hour for item in result], [0, 1, 2])
        self.assertEqual(result[-1].close, 202)


if __name__ == "__main__":
    unittest.main()
