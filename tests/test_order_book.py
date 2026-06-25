import unittest

from trading_agent.order_book import OrderBookLevel, OrderBookSnapshot, analyze_order_book


class OrderBookTests(unittest.TestCase):
    def test_analyze_order_book_detects_buy_support(self):
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[
                OrderBookLevel(100.0, 20.0),
                OrderBookLevel(99.9, 20.0),
            ],
            asks=[
                OrderBookLevel(100.1, 2.0),
                OrderBookLevel(100.2, 2.0),
            ],
        )

        analysis = analyze_order_book(snapshot, quote_amount=25)

        self.assertEqual(analysis.symbol, "BTCUSDT")
        self.assertGreater(analysis.depth_imbalance_pct, 0)
        self.assertEqual(analysis.pressure, "buy support")
        self.assertGreaterEqual(analysis.estimated_slippage_pct_25_usdt, 0)


if __name__ == "__main__":
    unittest.main()
