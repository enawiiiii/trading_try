import json
import tempfile
import unittest
from pathlib import Path

from trading_agent.config import AgentConfig
from trading_agent.edge_backtest import save_pocket_stats


class EdgeBacktestStatsTests(unittest.TestCase):
    def test_save_pocket_stats_uses_all_buy_win_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(edge_events_path=str(Path(tmp) / "edge.jsonl"))
            report = {
                "models": {
                    "CCUSDT_4h": {
                        "summary": {
                            "buy_events": 20,
                            "all_buy_avg_realized_pct": 0.15,
                            "all_buy_win_rate_pct": 55.0,
                            "blocked_buy_win_rate_pct": 0.0,
                        }
                    }
                }
            }

            path = save_pocket_stats(config, report)
            saved = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(saved["CCUSDT_4h"]["win_rate_pct"], 55.0)


if __name__ == "__main__":
    unittest.main()
