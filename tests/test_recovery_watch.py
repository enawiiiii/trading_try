import unittest

from trading_agent.models import Action
from trading_agent.recovery_watch import _parse_pocket_key, _status
from trading_agent.strategies import StrategyName


class RecoveryWatchTests(unittest.TestCase):
    def test_parse_pocket_key(self):
        parsed = _parse_pocket_key("SOLUSDT_1h|Mean Reversion Strategy|sideways")

        self.assertEqual(parsed, ("SOLUSDT", "1h", StrategyName.MEAN_REVERSION, "sideways"))

    def test_status_ready_now_requires_recovery_monitor(self):
        self.assertEqual(_status("RECOVERY_MONITOR", Action.BUY, []), "ready_now")
        self.assertEqual(_status("OBSERVE", Action.BUY, []), "near_miss")
        self.assertEqual(_status("OBSERVE", Action.SELL, []), "not_ready")


if __name__ == "__main__":
    unittest.main()
