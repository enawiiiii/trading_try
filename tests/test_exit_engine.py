import unittest
from datetime import datetime, timedelta, timezone

from trading_agent.config import AgentConfig
from trading_agent.exit import evaluate_exit
from trading_agent.models import Candle
from trading_agent.portfolio import PaperPortfolio


def candles(last_close: float) -> list[Candle]:
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    output = []
    for i in range(80):
        price = 100 + i * 0.02
        if i == 79:
            price = last_close
        output.append(
            Candle(
                open_time=base_time + timedelta(hours=i),
                open=price * 0.999,
                high=price * 1.004,
                low=price * 0.996,
                close=price,
                volume=10,
                close_time=base_time + timedelta(hours=i + 1),
                quote_volume=1000,
                trades=0,
            )
        )
    return output


class ExitEngineTests(unittest.TestCase):
    def test_no_position_holds(self):
        portfolio = PaperPortfolio(1000, {}, {}, {}, {})
        decision = evaluate_exit("BTCUSDT", candles(100), portfolio, AgentConfig())
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.quantity, 0)

    def test_stop_loss_sells(self):
        portfolio = PaperPortfolio(
            cash=0,
            positions={"BTCUSDT": 1.0},
            last_prices={},
            cost_basis={"BTCUSDT": 100.0},
            peak_prices={"BTCUSDT": 100.0},
        )
        decision = evaluate_exit("BTCUSDT", candles(98.0), portfolio, AgentConfig(stop_loss_pct=1.2))
        self.assertEqual(decision.action, "SELL")
        self.assertEqual(decision.risk_level, "high")

    def test_take_profit_sells(self):
        portfolio = PaperPortfolio(
            cash=0,
            positions={"BTCUSDT": 1.0},
            last_prices={},
            cost_basis={"BTCUSDT": 100.0},
            peak_prices={"BTCUSDT": 103.0},
        )
        decision = evaluate_exit("BTCUSDT", candles(103.0), portfolio, AgentConfig(take_profit_pct=2.4))
        self.assertEqual(decision.action, "SELL")
        self.assertGreater(decision.unrealized_pnl_pct, 2.4)


if __name__ == "__main__":
    unittest.main()
