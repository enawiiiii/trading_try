import unittest

from trading_agent.universe import _safe_symbol, _score


class UniverseTests(unittest.TestCase):
    def test_safe_symbol_rejects_stables_and_leveraged_suffixes(self):
        self.assertFalse(_safe_symbol("USDC", "USDCUSDT"))
        self.assertFalse(_safe_symbol("BTC", "BTC3LUSDT"))
        self.assertTrue(_safe_symbol("BTC", "BTCUSDT"))

    def test_score_rewards_liquidity_and_penalizes_spread(self):
        liquid = _score(10_000_000, 100_000, 0.02, 0.01)
        illiquid = _score(100_000, 100, 0.4, 0.01)
        self.assertGreater(liquid, illiquid)


if __name__ == "__main__":
    unittest.main()
