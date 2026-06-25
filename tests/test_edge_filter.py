import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_agent.config import AgentConfig
from trading_agent.edge_filter import evaluate_edge_filter, recent_edge_events, append_edge_event
from trading_agent.models import Action, MarketState
from trading_agent.risk import RiskDecision
from trading_agent.strategies import StrategySignal
from trading_agent.models import StrategyName


def state(**overrides):
    base = {
        "trend": "bullish",
        "volatility": "medium",
        "liquidity": "high",
        "liquidity_score": 80,
        "structure": "trend continuation",
        "fake_breakout_probability": 25,
        "current_price": 100.0,
        "atr_pct": 2.0,
        "rsi": 55.0,
        "support": 95.0,
        "resistance": 105.0,
        "bullish_probability": 45,
        "bearish_probability": 25,
        "sideways_probability": 30,
        "evidence_summary": "test evidence",
        "explanation": "test state",
    }
    base.update(overrides)
    return MarketState(**base)


class EdgeFilterTests(unittest.TestCase):
    def test_strong_negative_ev_blocks_only_in_monitor_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "BTCUSDT_1h": {
                      "metrics": {
                        "accuracy": 42.0,
                        "macro_f1": 40.0,
                        "long_expected_value_pct": -0.22,
                        "by_label": {"sideways": {"accuracy": 35.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 80, "confirmed")
            risk = RiskDecision(Action.BUY, 80, "medium", "risk allowed")

            result = evaluate_edge_filter(
                symbol="BTCUSDT",
                interval="1h",
                state=state(liquidity_score=80, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertEqual(result.mode, "monitor_only")
            self.assertEqual(result.observed_action, "BUY")
            self.assertEqual(result.monitor_action, "BLOCK")
            self.assertFalse(result.would_allow_trade)

    def test_positive_historical_pocket_can_create_weak_allow_monitor(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "BTCUSDT_1h": {
                      "metrics": {
                        "accuracy": 44.0,
                        "macro_f1": 40.0,
                        "long_expected_value_pct": -0.02,
                        "by_label": {"sideways": {"accuracy": 40.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            pocket = Path(tmp) / "edge_pocket_stats.json"
            pocket.write_text(
                """
                {
                  "BTCUSDT_1h": {
                    "buy_events": 45,
                    "avg_realized_pct": 0.10,
                    "win_rate_pct": 40.0
                  }
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 82, "confirmed")
            risk = RiskDecision(Action.BUY, 82, "low", "risk allowed")

            result = evaluate_edge_filter(
                symbol="BTCUSDT",
                interval="1h",
                state=state(liquidity_score=85, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertIn(result.monitor_action, {"WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR"})
            self.assertTrue(result.would_allow_trade)

    def test_low_win_rate_historical_pocket_does_not_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "CCUSDT_4h": {
                      "metrics": {
                        "accuracy": 44.0,
                        "macro_f1": 43.0,
                        "long_expected_value_pct": -0.02,
                        "by_label": {"sideways": {"accuracy": 40.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            pocket = Path(tmp) / "edge_pocket_stats.json"
            pocket.write_text(
                """
                {
                  "CCUSDT_4h": {
                    "buy_events": 30,
                    "avg_realized_pct": 0.18,
                    "win_rate_pct": 12.0
                  }
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 82, "confirmed")
            risk = RiskDecision(Action.BUY, 82, "low", "risk allowed")

            result = evaluate_edge_filter(
                symbol="CCUSDT",
                interval="4h",
                state=state(liquidity_score=85, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertLess(result.pocket_score, 0)
            self.assertNotEqual(result.monitor_action, "WEAK_ALLOW_MONITOR")

    def test_live_outcome_pocket_can_promote_observed_buy(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "SOLUSDT_1h": {
                      "metrics": {
                        "accuracy": 42.0,
                        "macro_f1": 39.0,
                        "long_expected_value_pct": -0.11,
                        "by_label": {"sideways": {"accuracy": 35.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            outcome = Path(tmp) / "edge_outcome_report.json"
            outcome.write_text(
                """
                {
                  "by_symbol_interval": [
                    {
                      "key": "SOLUSDT_1h",
                      "evaluated": 81,
                      "win_rate_pct": 77.78,
                      "avg_realized_pct": 1.3257
                    }
                  ],
                  "by_edge_symbol_interval": []
                }
                """,
                encoding="utf-8",
            )
            quality = Path(tmp) / "signal_quality_report.json"
            quality.write_text(
                """
                {
                  "by_symbol_interval_strategy": [
                    {
                      "key": "CCUSDT_1h|Breakout Strategy",
                      "buy_events": 25,
                      "win_rate_pct": 65.0,
                      "avg_realized_pct": 0.72
                    }
                  ],
                  "by_strategy_regime": [],
                  "by_symbol_interval": []
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 75, "confirmed")
            risk = RiskDecision(Action.BUY, 75, "medium", "risk allowed")

            result = evaluate_edge_filter(
                symbol="SOLUSDT",
                interval="1h",
                state=state(liquidity_score=80, fake_breakout_probability=25),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertGreater(result.outcome_score, 0)
            self.assertIn(result.monitor_action, {"WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR"})

    def test_good_signal_rejected_by_risk_can_be_monitor_allow_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "SOLUSDT_1h": {
                      "metrics": {
                        "accuracy": 44.0,
                        "macro_f1": 43.0,
                        "long_expected_value_pct": -0.02,
                        "by_label": {"sideways": {"accuracy": 40.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            outcome = Path(tmp) / "edge_outcome_report.json"
            outcome.write_text(
                """
                {
                  "by_symbol_interval": [
                    {
                      "key": "SOLUSDT_1h",
                      "evaluated": 81,
                      "win_rate_pct": 77.78,
                      "avg_realized_pct": 1.3257
                    }
                  ],
                  "by_edge_symbol_interval": []
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 75, "confirmed")
            risk = RiskDecision(Action.NO_TRADE, 75, "medium", "risk rejected")

            result = evaluate_edge_filter(
                symbol="SOLUSDT",
                interval="1h",
                state=state(liquidity_score=80, fake_breakout_probability=25),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertIn(result.monitor_action, {"WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR"})
            self.assertFalse(result.would_allow_trade)

    def test_bad_live_weak_allow_outcome_downgrades_to_observe(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "XRPUSDT_1h": {
                      "metrics": {
                        "accuracy": 44.0,
                        "macro_f1": 43.0,
                        "long_expected_value_pct": -0.02,
                        "by_label": {"sideways": {"accuracy": 40.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            outcome = Path(tmp) / "edge_outcome_report.json"
            outcome.write_text(
                """
                {
                  "by_symbol_interval": [
                    {
                      "key": "XRPUSDT_1h",
                      "evaluated": 26,
                      "win_rate_pct": 0.0,
                      "avg_realized_pct": -0.2001
                    }
                  ],
                  "by_edge_symbol_interval": [
                    {
                      "key": "WEAK_ALLOW_MONITOR|XRPUSDT_1h",
                      "evaluated": 23,
                      "win_rate_pct": 0.0,
                      "avg_realized_pct": -0.204
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 82, "confirmed")
            risk = RiskDecision(Action.BUY, 82, "low", "risk allowed")

            result = evaluate_edge_filter(
                symbol="XRPUSDT",
                interval="1h",
                state=state(liquidity_score=85, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertLess(result.outcome_score, 0)
            self.assertEqual(result.monitor_action, "OBSERVE")
            self.assertFalse(result.would_allow_trade)

    def test_sell_signal_does_not_count_as_buy_allow(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "BTCUSDT_1h": {
                      "metrics": {
                        "accuracy": 44.0,
                        "macro_f1": 43.0,
                        "long_expected_value_pct": 0.2,
                        "by_label": {"sideways": {"accuracy": 40.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.SELL, 82, "exit confirmed")
            risk = RiskDecision(Action.SELL, 82, "low", "risk allowed")

            result = evaluate_edge_filter(
                symbol="BTCUSDT",
                interval="1h",
                state=state(liquidity_score=85, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertEqual(result.monitor_action, "OBSERVE")
            self.assertFalse(result.would_allow_trade)

    def test_globally_bad_weak_allow_is_suspended(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "BTCUSDT_1h": {
                      "metrics": {
                        "accuracy": 44.0,
                        "macro_f1": 43.0,
                        "long_expected_value_pct": -0.02,
                        "by_label": {"sideways": {"accuracy": 40.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            outcome = Path(tmp) / "edge_outcome_report.json"
            outcome.write_text(
                """
                {
                  "by_edge_action": [
                    {
                      "key": "WEAK_ALLOW_MONITOR",
                      "evaluated": 10,
                      "win_rate_pct": 0.0,
                      "avg_realized_pct": -1.3
                    }
                  ],
                  "by_symbol_interval": [],
                  "by_edge_symbol_interval": []
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 65, "confirmed")
            risk = RiskDecision(Action.BUY, 65, "low", "risk allowed")

            result = evaluate_edge_filter(
                symbol="BTCUSDT",
                interval="1h",
                state=state(liquidity_score=85, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertEqual(result.monitor_action, "OBSERVE")
            self.assertFalse(result.would_allow_trade)
            self.assertTrue(any("WEAK_ALLOW is suspended" in reason for reason in result.reasons))

    def test_globally_bad_strong_allow_is_suspended(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "XRPUSDT_4h": {
                      "metrics": {
                        "accuracy": 44.0,
                        "macro_f1": 43.0,
                        "long_expected_value_pct": -0.02,
                        "by_label": {"sideways": {"accuracy": 40.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            pocket = Path(tmp) / "edge_pocket_stats.json"
            pocket.write_text(
                """
                {
                  "XRPUSDT_4h": {
                    "buy_events": 45,
                    "avg_realized_pct": 0.31,
                    "win_rate_pct": 52.0
                  }
                }
                """,
                encoding="utf-8",
            )
            outcome = Path(tmp) / "edge_outcome_report.json"
            outcome.write_text(
                """
                {
                  "by_edge_action": [
                    {
                      "key": "STRONG_ALLOW_MONITOR",
                      "evaluated": 3,
                      "win_rate_pct": 0.0,
                      "avg_realized_pct": -1.3
                    }
                  ],
                  "by_symbol_interval": [
                    {
                      "key": "XRPUSDT_4h",
                      "evaluated": 50,
                      "win_rate_pct": 70.0,
                      "avg_realized_pct": 1.0
                    }
                  ],
                  "by_edge_symbol_interval": []
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 82, "confirmed")
            risk = RiskDecision(Action.BUY, 82, "low", "risk allowed")

            result = evaluate_edge_filter(
                symbol="XRPUSDT",
                interval="4h",
                state=state(liquidity_score=85, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertEqual(result.monitor_action, "OBSERVE")
            self.assertFalse(result.would_allow_trade)
            self.assertTrue(any("STRONG_ALLOW is suspended" in reason for reason in result.reasons))

    def test_constructive_blocked_outcomes_can_promote_monitor_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "CCUSDT_1h": {
                      "metrics": {
                        "accuracy": 44.0,
                        "macro_f1": 43.0,
                        "long_expected_value_pct": -0.02,
                        "by_label": {"sideways": {"accuracy": 40.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            outcome = Path(tmp) / "edge_outcome_report.json"
            outcome.write_text(
                """
                {
                  "by_symbol_interval": [],
                  "by_edge_symbol_interval": [
                    {
                      "key": "BLOCK|CCUSDT_1h",
                      "evaluated": 65,
                      "win_rate_pct": 98.46,
                      "avg_realized_pct": 2.2446
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.MEAN_REVERSION, Action.BUY, 82, "confirmed")
            risk = RiskDecision(Action.BUY, 82, "low", "risk allowed")

            result = evaluate_edge_filter(
                symbol="CCUSDT",
                interval="1h",
                state=state(liquidity_score=85, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertGreater(result.blocked_outcome_score, 0)
            self.assertIn(result.monitor_action, {"WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR"})

    def test_strong_blocked_winner_becomes_recovery_monitor_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "CCUSDT_1h": {
                      "metrics": {
                        "accuracy": 41.0,
                        "macro_f1": 32.0,
                        "long_expected_value_pct": -0.13,
                        "by_label": {"sideways": {"accuracy": 10.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            outcome = Path(tmp) / "edge_outcome_report.json"
            outcome.write_text(
                """
                {
                  "by_edge_action": [],
                  "by_symbol_interval": [],
                  "by_edge_symbol_interval": [
                    {
                      "key": "BLOCK|CCUSDT_1h",
                      "evaluated": 65,
                      "pending": 0,
                      "win_rate_pct": 98.46,
                      "avg_realized_pct": 2.2446
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            quality = Path(tmp) / "signal_quality_report.json"
            quality.write_text(
                """
                {
                  "by_symbol_interval_strategy": [
                    {
                      "key": "CCUSDT_1h|Breakout Strategy",
                      "buy_events": 25,
                      "win_rate_pct": 65.0,
                      "avg_realized_pct": 0.72
                    }
                  ],
                  "by_strategy_regime": [],
                  "by_symbol_interval": []
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.BREAKOUT, Action.BUY, 80, "confirmed")
            risk = RiskDecision(Action.BUY, 80, "medium", "risk allowed")

            result = evaluate_edge_filter(
                symbol="CCUSDT",
                interval="1h",
                state=state(liquidity_score=85, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertEqual(result.monitor_action, "RECOVERY_MONITOR")
            self.assertFalse(result.would_allow_trade)

    def test_bad_signal_quality_memory_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "BTCUSDT_1h": {
                      "metrics": {
                        "accuracy": 44.0,
                        "macro_f1": 43.0,
                        "long_expected_value_pct": -0.02,
                        "by_label": {"sideways": {"accuracy": 40.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            quality = Path(tmp) / "signal_quality_report.json"
            quality.write_text(
                """
                {
                  "by_symbol_interval_strategy": [
                    {
                      "key": "BTCUSDT_1h|Trend Following Strategy",
                      "buy_events": 60,
                      "win_rate_pct": 25.0,
                      "avg_realized_pct": -0.2
                    }
                  ],
                  "by_strategy_regime": [],
                  "by_symbol_interval": []
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 82, "confirmed")
            risk = RiskDecision(Action.BUY, 82, "low", "risk allowed")

            result = evaluate_edge_filter(
                symbol="BTCUSDT",
                interval="1h",
                state=state(liquidity_score=85, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertEqual(result.monitor_action, "OBSERVE")
            self.assertFalse(result.would_allow_trade)

    def test_blocked_winners_audit_can_create_recovery_monitor_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "SOLUSDT_1h": {
                      "metrics": {
                        "accuracy": 39.0,
                        "macro_f1": 34.0,
                        "long_expected_value_pct": -0.13,
                        "by_label": {"sideways": {"accuracy": 10.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            audit = Path(tmp) / "blocked_winners_audit.json"
            audit.write_text(
                f"""
                {{
                  "generated_at": "{datetime.now(timezone.utc).isoformat()}",
                  "best_pockets": [
                    {{
                      "key": "SOLUSDT_1h|Mean Reversion Strategy|sideways",
                      "evaluated": 13,
                      "pending": 0,
                      "win_rate_pct": 100.0,
                      "avg_realized_pct": 2.3
                    }}
                  ]
                }}
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.MEAN_REVERSION, Action.BUY, 80, "confirmed")
            risk = RiskDecision(Action.NO_TRADE, 80, "medium", "risk rejected")

            result = evaluate_edge_filter(
                symbol="SOLUSDT",
                interval="1h",
                state=state(trend="sideways", sideways_probability=65, bullish_probability=20, bearish_probability=15),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertEqual(result.monitor_action, "RECOVERY_MONITOR")
            self.assertFalse(result.would_allow_trade)

    def test_constructive_exact_pocket_rule_can_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                """
                {
                  "models": {
                    "XRPUSDT_4h": {
                      "metrics": {
                        "accuracy": 44.0,
                        "macro_f1": 43.0,
                        "long_expected_value_pct": -0.02,
                        "by_label": {"sideways": {"accuracy": 40.0}}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            quality = Path(tmp) / "signal_quality_report.json"
            quality.write_text(
                """
                {
                  "by_symbol_interval_strategy": [
                    {
                      "key": "XRPUSDT_4h|Trend Following Strategy",
                      "buy_events": 32,
                      "win_rate_pct": 46.88,
                      "avg_realized_pct": 0.2974
                    }
                  ],
                  "by_strategy_regime": [],
                  "by_symbol_interval": []
                }
                """,
                encoding="utf-8",
            )
            config = AgentConfig(ml_report_path=str(report), edge_events_path=str(Path(tmp) / "events.jsonl"))
            signal = StrategySignal(StrategyName.TREND_FOLLOWING, Action.BUY, 82, "confirmed")
            risk = RiskDecision(Action.BUY, 82, "low", "risk allowed")

            result = evaluate_edge_filter(
                symbol="XRPUSDT",
                interval="4h",
                state=state(liquidity_score=85, fake_breakout_probability=20),
                signal=signal,
                risk=risk,
                config=config,
            )

            self.assertIn(result.monitor_action, {"WEAK_ALLOW_MONITOR", "STRONG_ALLOW_MONITOR"})

    def test_edge_event_log_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(edge_events_path=str(Path(tmp) / "edge.jsonl"))
            append_edge_event(config, {"symbol": "BTCUSDT", "edge_filter": {"monitor_action": "OBSERVE"}})
            rows = recent_edge_events(config)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["symbol"], "BTCUSDT")


if __name__ == "__main__":
    unittest.main()
