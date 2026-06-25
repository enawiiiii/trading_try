from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from trading_agent.agent import decide
from trading_agent.config import load_config
from trading_agent.data import fetch_market_klines
from trading_agent.edge_filter import summarize_edge_events
from trading_agent.env import load_env_file
from trading_agent.exit import evaluate_exit
from trading_agent.journal import load_recent, summarize_learning
from trading_agent.learning_monitor import build_learning_monitor_report
from trading_agent.order_book import analyze_order_book, fetch_bybit_order_book
from trading_agent.portfolio import load_portfolio
from trading_agent.risk import RiskContext


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"


class DashboardHandler(SimpleHTTPRequestHandler):
    config_path = "config.demo.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/snapshot":
            self._send_json(self._snapshot(parsed.query))
            return
        if parsed.path == "/":
            self.path = "/dashboard.html"
        super().do_GET()

    def log_message(self, format: str, *args) -> None:
        return

    def _snapshot(self, query: str) -> dict:
        config = load_config(self.config_path)
        params = parse_qs(query)
        symbols = params.get("symbols", ["BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"])[0].split(",")
        interval = params.get("interval", ["1h"])[0]
        exchange = params.get("exchange", [config.exchange])[0]
        portfolio = load_portfolio(config)

        assets = []
        for symbol in [item.strip().upper() for item in symbols if item.strip()]:
            try:
                candles = fetch_market_klines(exchange, symbol, interval, 240, config.bybit_market_testnet)
                context = RiskContext(
                    daily_pnl_pct=portfolio.daily_pnl_pct,
                    consecutive_losses=portfolio.consecutive_losses,
                    has_position=portfolio.has_position(symbol),
                )
                decision = decide(symbol, candles, config, context, interval)
                exit_decision = evaluate_exit(symbol, candles, portfolio, config)
                order_book = _order_book_payload(symbol, config) if exchange.lower() == "bybit" else None
                assets.append(
                    {
                        "symbol": symbol,
                        "status": "ok",
                        "decision": _json_safe(asdict(decision)),
                        "exit": exit_decision.to_dict(),
                        "order_book": order_book,
                        "sparkline": [round(candle.close, 8) for candle in candles[-60:]],
                    }
                )
            except Exception as exc:
                assets.append({"symbol": symbol, "status": "error", "error": str(exc)})

        return {
            "exchange": exchange,
            "interval": interval,
            "config": {
                "live_trading_enabled": config.live_trading_enabled,
                "bybit_demo": config.bybit_demo,
                "max_live_order_quote": config.max_live_order_quote,
                "stop_loss_pct": config.stop_loss_pct,
                "take_profit_pct": config.take_profit_pct,
                "trailing_stop_pct": config.trailing_stop_pct,
            },
            "portfolio": {
                "cash": portfolio.cash,
                "positions": portfolio.positions,
                "cost_basis": portfolio.cost_basis,
                "last_prices": portfolio.last_prices,
                "equity": portfolio.equity,
                "consecutive_losses": portfolio.consecutive_losses,
            },
            "learning": summarize_learning(config),
            "learning_monitor": build_learning_monitor_report(config),
            "edge_filter_monitor": summarize_edge_events(config),
            "recovery": {
                "watch": _read_data_json("recovery_watch_report.json"),
                "shadow": _read_data_json("recovery_shadow_report.json"),
            },
            "paper_simulator": _read_data_json("paper_simulator_report.json"),
            "data_sources": {
                "historical": [
                    {"name": "Bybit", "status": "active", "usage": "primary candles and demo execution"},
                    {"name": "Binance", "status": "available", "usage": "free public historical candle confirmation"},
                ],
                "microstructure": [
                    {"name": "Bybit Order Book", "status": "active", "usage": "spread, depth, imbalance, and slippage monitoring"},
                ],
            },
            "recent": load_recent(config, 6),
            "assets": assets,
        }

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _order_book_payload(symbol: str, config) -> dict:
    try:
        snapshot = fetch_bybit_order_book(symbol, limit=50, testnet=config.bybit_market_testnet)
        analysis = analyze_order_book(snapshot, quote_amount=config.max_live_order_quote)
        return {"status": "ok", **analysis.to_dict()}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _read_data_json(name: str) -> dict:
    path = ROOT / "data" / name
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description="Run the trading dashboard server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--config", default="config.demo.json")
    args = parser.parse_args()

    DashboardHandler.config_path = args.config
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
