from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from trading_agent.config import AgentConfig


@dataclass
class PaperPortfolio:
    cash: float
    positions: dict[str, float]
    last_prices: dict[str, float]
    cost_basis: dict[str, float]
    peak_prices: dict[str, float]
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0

    @property
    def equity(self) -> float:
        position_value = sum(qty * self.last_prices.get(symbol, 0.0) for symbol, qty in self.positions.items())
        return self.cash + position_value

    def has_position(self, symbol: str) -> bool:
        return self.positions.get(symbol.upper(), 0.0) > 0


def load_portfolio(config: AgentConfig) -> PaperPortfolio:
    path = Path(config.portfolio_path)
    if not path.exists():
        return PaperPortfolio(
            cash=config.paper_starting_cash,
            positions={},
            last_prices={},
            cost_basis={},
            peak_prices={},
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PaperPortfolio(
        cash=float(raw.get("cash", config.paper_starting_cash)),
        positions={key: float(value) for key, value in raw.get("positions", {}).items()},
        last_prices={key: float(value) for key, value in raw.get("last_prices", {}).items()},
        cost_basis={key: float(value) for key, value in raw.get("cost_basis", {}).items()},
        peak_prices={key: float(value) for key, value in raw.get("peak_prices", {}).items()},
        daily_pnl_pct=float(raw.get("daily_pnl_pct", 0.0)),
        consecutive_losses=int(raw.get("consecutive_losses", 0)),
    )


def save_portfolio(config: AgentConfig, portfolio: PaperPortfolio) -> None:
    path = Path(config.portfolio_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cash": portfolio.cash,
        "positions": portfolio.positions,
        "last_prices": portfolio.last_prices,
        "cost_basis": portfolio.cost_basis,
        "peak_prices": portfolio.peak_prices,
        "daily_pnl_pct": portfolio.daily_pnl_pct,
        "consecutive_losses": portfolio.consecutive_losses,
        "equity": portfolio.equity,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
