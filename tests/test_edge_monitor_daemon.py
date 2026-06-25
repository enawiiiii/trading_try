import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_agent.config import AgentConfig
from trading_agent.edge_monitor_daemon import _selected_symbols, _top_signal_quality_symbols, run_edge_monitor_daemon
from trading_agent.models import Action, Decision, StrategyName


class EdgeMonitorDaemonTests(unittest.TestCase):
    def test_daemon_writes_status_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "edge_events.jsonl"
            stop_path = Path(tmp) / "edge_monitor.stop"
            config = AgentConfig(edge_events_path=str(events_path))

            def fake_decide(symbol, candles, config, context, interval):
                stop_path.write_text("", encoding="utf-8")
                return Decision(
                    action=Action.NO_TRADE,
                    confidence=40,
                    selected_strategy=StrategyName.TREND_FOLLOWING,
                    market_state="test",
                    reasoning="test",
                    risk_level="low",
                    learning_note="test",
                    symbol=symbol,
                    price=1.0,
                    timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                    edge_filter={"monitor_action": "OBSERVE", "score": 40, "estimated_ev_pct": -0.1},
                )

            with patch("trading_agent.edge_monitor_daemon.fetch_market_klines", return_value=[]), patch(
                "trading_agent.edge_monitor_daemon.decide", side_effect=fake_decide
            ), patch("trading_agent.edge_monitor_daemon.analyze_edge_outcomes", return_value={}):
                run_edge_monitor_daemon(config, ["BTCUSDT"], interval="1h", sleep_minutes=1)

            status_path = Path(tmp) / "edge_monitor_status.json"
            self.assertTrue(status_path.exists())
            self.assertIn("stopped", status_path.read_text(encoding="utf-8"))

    def test_selected_symbols_adds_quality_top_ev_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "TRXUSDT_1h": {
                      "metrics": {"long_expected_value_pct": 0.04, "macro_f1": 36.0, "accuracy": 43.0, "samples": 900}
                    },
                    "BADUSDT_1h": {
                      "metrics": {"long_expected_value_pct": 0.20, "macro_f1": 20.0, "accuracy": 50.0, "samples": 900}
                    },
                    "TINYUSDT_1h": {
                      "metrics": {"long_expected_value_pct": 0.10, "macro_f1": 40.0, "accuracy": 50.0, "samples": 20}
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report))
            selected = _selected_symbols(config, ["BTCUSDT"], "1h", 5)
            self.assertIn("BTCUSDT", selected)
            self.assertIn("TRXUSDT", selected)
            self.assertNotIn("BADUSDT", selected)
            self.assertNotIn("TINYUSDT", selected)

    def test_top_signal_quality_symbols_uses_positive_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            quality = Path(tmp) / "signal_quality_report.json"
            quality.write_text(
                """
                {
                  "by_symbol_interval": [
                    {"key": "XRPUSDT_1h", "buy_events": 50, "avg_realized_pct": 0.04, "win_rate_pct": 46.0},
                    {"key": "BADUSDT_1h", "buy_events": 50, "avg_realized_pct": -0.2, "win_rate_pct": 20.0},
                    {"key": "TINYUSDT_1h", "buy_events": 5, "avg_realized_pct": 1.0, "win_rate_pct": 80.0}
                  ]
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(edge_events_path=str(Path(tmp) / "edge_events.jsonl"))
            selected = _top_signal_quality_symbols(config, "1h", 5)
            self.assertIn("XRPUSDT", selected)
            self.assertNotIn("BADUSDT", selected)
            self.assertNotIn("TINYUSDT", selected)


if __name__ == "__main__":
    unittest.main()
