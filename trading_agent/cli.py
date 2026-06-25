from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from trading_agent.agent import decide, paper_execute
from trading_agent.auto_trainer import run_auto_training
from trading_agent.backtest import backtest_probabilities
from trading_agent.blocked_winners_audit import build_blocked_winners_audit
from trading_agent.bybit_client import LiveTradingBlocked, place_spot_market_order
from trading_agent.calibration import save_calibration
from trading_agent.config import load_config
from trading_agent.data import fetch_historical_market_klines, fetch_market_klines
from trading_agent.data_providers import fetch_coinapi_ohlcv, fetch_coinmarketcap_quotes
from trading_agent.edge_backtest import backtest_edge_filter
from trading_agent.edge_monitor_daemon import run_edge_monitor_daemon
from trading_agent.edge_outcomes import analyze_edge_outcomes
from trading_agent.env import load_env_file
from trading_agent.exit import evaluate_exit
from trading_agent.historical_store import load_candles, save_candles
from trading_agent.journal import append_decision, load_recent, summarize_learning
from trading_agent.learning import update_weekly_weights
from trading_agent.learning_monitor import build_learning_monitor_report, save_learning_monitor_report
from trading_agent.market import analyze_market
from trading_agent.ml_model import build_dataset, load_model, predict_model, save_model, train_model
from trading_agent.order_book import analyze_order_book, fetch_bybit_order_book
from trading_agent.paper_simulator import update_paper_simulator
from trading_agent.portfolio import load_portfolio, save_portfolio
from trading_agent.risk import RiskContext
from trading_agent.recovery_shadow import update_recovery_shadow_trades
from trading_agent.recovery_watch import build_recovery_watch_report
from trading_agent.signal_quality import build_signal_quality_report
from trading_agent.training_daemon import run_training_daemon
from trading_agent.universe import scan_bybit_universe


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser(description="Risk-first spot crypto trading agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a symbol and print a decision.")
    analyze.add_argument("--symbol", default="BTCUSDT")
    analyze.add_argument("--interval", default="1h")
    analyze.add_argument("--limit", type=int, default=240)
    analyze.add_argument("--config", default="config.json")
    analyze.add_argument("--exchange", choices=["bybit", "binance"], default=None)

    trade = subparsers.add_parser("trade", help="Run one paper-trading decision.")
    trade.add_argument("--symbol", default="BTCUSDT")
    trade.add_argument("--interval", default="1h")
    trade.add_argument("--limit", type=int, default=240)
    trade.add_argument("--config", default="config.json")
    trade.add_argument("--exchange", choices=["bybit", "binance"], default=None)
    trade.add_argument("--live", action="store_true", help="Attempt a real Bybit spot order when risk rules allow it.")
    trade.add_argument("--live-quote-amount", type=float, default=None)

    exit_check = subparsers.add_parser("exit-check", help="Evaluate whether an open spot position should be sold.")
    exit_check.add_argument("--symbol", default="BTCUSDT")
    exit_check.add_argument("--interval", default="1h")
    exit_check.add_argument("--limit", type=int, default=240)
    exit_check.add_argument("--config", default="config.json")
    exit_check.add_argument("--exchange", choices=["bybit", "binance"], default=None)
    exit_check.add_argument("--execute-paper", action="store_true", help="Apply SELL to the paper portfolio if exit is confirmed.")

    journal = subparsers.add_parser("journal", help="Show recent decisions and learning summary.")
    journal.add_argument("--config", default="config.json")
    journal.add_argument("--limit", type=int, default=10)

    portfolio_cmd = subparsers.add_parser("portfolio", help="Show paper portfolio state.")
    portfolio_cmd.add_argument("--config", default="config.json")

    learn = subparsers.add_parser("learn-weekly", help="Update strategy weights from closed trade outcomes.")
    learn.add_argument("--config", default="config.json")

    backtest = subparsers.add_parser("backtest", help="Backtest probability predictions on historical candles.")
    backtest.add_argument("--symbol", default="BTCUSDT")
    backtest.add_argument("--interval", default="1h")
    backtest.add_argument("--limit", type=int, default=500)
    backtest.add_argument("--lookback", type=int, default=80)
    backtest.add_argument("--horizon", type=int, default=6)
    backtest.add_argument("--move-threshold-pct", type=float, default=0.35)
    backtest.add_argument("--config", default="config.json")
    backtest.add_argument("--exchange", choices=["bybit", "binance"], default=None)

    calibrate = subparsers.add_parser("calibrate", help="Backtest and save probability calibration for a symbol/timeframe.")
    calibrate.add_argument("--symbol", default="BTCUSDT")
    calibrate.add_argument("--interval", default="1h")
    calibrate.add_argument("--limit", type=int, default=500)
    calibrate.add_argument("--lookback", type=int, default=80)
    calibrate.add_argument("--horizon", type=int, default=6)
    calibrate.add_argument("--move-threshold-pct", type=float, default=0.35)
    calibrate.add_argument("--config", default="config.json")
    calibrate.add_argument("--exchange", choices=["bybit", "binance"], default=None)

    import_history = subparsers.add_parser("import-history", help="Import historical OHLCV data into local storage.")
    import_history.add_argument("--provider", choices=["coinapi", "bybit", "binance"], default="coinapi")
    import_history.add_argument("--symbol", default="BTCUSDT")
    import_history.add_argument("--interval", default="1h")
    import_history.add_argument("--limit", type=int, default=1000)
    import_history.add_argument("--source-exchange", default="BYBIT")
    import_history.add_argument("--config", default="config.json")

    backtest_history = subparsers.add_parser("backtest-history", help="Backtest using locally stored historical candles.")
    backtest_history.add_argument("--provider", default="coinapi")
    backtest_history.add_argument("--symbol", default="BTCUSDT")
    backtest_history.add_argument("--interval", default="1h")
    backtest_history.add_argument("--lookback", type=int, default=80)
    backtest_history.add_argument("--horizon", type=int, default=6)
    backtest_history.add_argument("--move-threshold-pct", type=float, default=0.35)
    backtest_history.add_argument("--config", default="config.json")

    calibrate_history = subparsers.add_parser("calibrate-history", help="Calibrate probabilities using locally stored historical candles.")
    calibrate_history.add_argument("--provider", default="bybit")
    calibrate_history.add_argument("--symbol", default="BTCUSDT")
    calibrate_history.add_argument("--interval", default="1h")
    calibrate_history.add_argument("--lookback", type=int, default=80)
    calibrate_history.add_argument("--horizon", type=int, default=6)
    calibrate_history.add_argument("--move-threshold-pct", type=float, default=0.35)
    calibrate_history.add_argument("--config", default="config.json")

    market_context = subparsers.add_parser("market-context", help="Fetch broader market context from CoinMarketCap.")
    market_context.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,LTCUSDT")
    market_context.add_argument("--config", default="config.json")

    train_ml = subparsers.add_parser("train-ml", help="Train a local ML model from stored historical candles.")
    train_ml.add_argument("--provider", default="bybit")
    train_ml.add_argument("--symbol", default="BTCUSDT")
    train_ml.add_argument("--interval", default="1h")
    train_ml.add_argument("--lookback", type=int, default=80)
    train_ml.add_argument("--horizon", type=int, default=6)
    train_ml.add_argument("--move-threshold-pct", type=float, default=0.35)
    train_ml.add_argument("--label-mode", choices=["future_close", "tp_sl_path"], default="tp_sl_path")
    train_ml.add_argument("--take-profit-pct", type=float, default=None)
    train_ml.add_argument("--stop-loss-pct", type=float, default=None)
    train_ml.add_argument("--epochs", type=int, default=180)
    train_ml.add_argument("--learning-rate", type=float, default=0.035)
    train_ml.add_argument("--config", default="config.json")

    predict_ml = subparsers.add_parser("predict-ml", help="Predict the next scenario using a trained local ML model.")
    predict_ml.add_argument("--symbol", default="BTCUSDT")
    predict_ml.add_argument("--interval", default="1h")
    predict_ml.add_argument("--provider", default="bybit")
    predict_ml.add_argument("--config", default="config.json")

    auto_train = subparsers.add_parser("auto-train", help="Run the full automatic learning pipeline.")
    auto_train.add_argument("--provider", choices=["bybit", "binance"], default="bybit")
    auto_train.add_argument("--config", default="config.json")

    scan_universe = subparsers.add_parser("scan-universe", help="Select the best Bybit Spot USDT symbols for ML training.")
    scan_universe.add_argument("--top", type=int, default=None)
    scan_universe.add_argument("--config", default="config.json")

    train_daemon = subparsers.add_parser("train-daemon", help="Run continuous ML training cycles forever.")
    train_daemon.add_argument("--provider", choices=["bybit", "binance"], default="bybit")
    train_daemon.add_argument("--sleep-minutes", type=int, default=30)
    train_daemon.add_argument("--no-universe-scan", action="store_true")
    train_daemon.add_argument("--config", default="config.json")

    edge_monitor = subparsers.add_parser("edge-monitor-daemon", help="Run monitor-only Edge Filter event collection forever.")
    edge_monitor.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    edge_monitor.add_argument("--interval", default="1h")
    edge_monitor.add_argument("--exchange", choices=["bybit", "binance"], default=None)
    edge_monitor.add_argument("--limit", type=int, default=240)
    edge_monitor.add_argument("--sleep-minutes", type=int, default=5)
    edge_monitor.add_argument("--top-ev-symbols", type=int, default=8)
    edge_monitor.add_argument("--config", default="config.json")

    edge_backtest = subparsers.add_parser("edge-backtest", help="Backtest monitor-only Edge Filter behavior on stored candles.")
    edge_backtest.add_argument("--provider", default="bybit")
    edge_backtest.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,ASTERUSDT,CCUSDT,XPLUSDT")
    edge_backtest.add_argument("--intervals", default="1h,4h")
    edge_backtest.add_argument("--lookback", type=int, default=80)
    edge_backtest.add_argument("--max-events", type=int, default=500)
    edge_backtest.add_argument("--stride", type=int, default=6)
    edge_backtest.add_argument("--config", default="config.json")

    edge_outcomes = subparsers.add_parser("edge-outcomes", help="Analyze realized outcomes for logged Edge Filter BUY events.")
    edge_outcomes.add_argument("--exchange", choices=["bybit", "binance"], default=None)
    edge_outcomes.add_argument("--hours", type=int, default=24)
    edge_outcomes.add_argument("--limit", type=int, default=1000)
    edge_outcomes.add_argument("--config", default="config.json")

    blocked_winners = subparsers.add_parser("blocked-winners-audit", help="Find profitable rejected BUY pockets for Edge Filter tuning.")
    blocked_winners.add_argument("--exchange", choices=["bybit", "binance"], default=None)
    blocked_winners.add_argument("--hours", type=int, default=24)
    blocked_winners.add_argument("--limit", type=int, default=1000)
    blocked_winners.add_argument("--min-evaluated", type=int, default=3)
    blocked_winners.add_argument("--config", default="config.json")

    recovery_watch = subparsers.add_parser("recovery-watch", help="Inspect whether profitable recovery pockets are actionable now.")
    recovery_watch.add_argument("--exchange", choices=["bybit", "binance"], default=None)
    recovery_watch.add_argument("--limit", type=int, default=240)
    recovery_watch.add_argument("--max-pockets", type=int, default=12)
    recovery_watch.add_argument("--config", default="config.json")

    recovery_shadow = subparsers.add_parser("recovery-shadow", help="Open/evaluate monitor-only shadow trades from recovery-watch.")
    recovery_shadow.add_argument("--exchange", choices=["bybit", "binance"], default=None)
    recovery_shadow.add_argument("--limit", type=int, default=1000)
    recovery_shadow.add_argument("--no-open-from-watch", action="store_true")
    recovery_shadow.add_argument("--config", default="config.json")

    paper_sim = subparsers.add_parser("paper-simulator", help="Run monitor-only paper simulation from Edge Filter events.")
    paper_sim.add_argument("--exchange", choices=["bybit", "binance"], default=None)
    paper_sim.add_argument("--limit", type=int, default=1000)
    paper_sim.add_argument("--source-hours", type=int, default=6)
    paper_sim.add_argument("--max-new", type=int, default=12)
    paper_sim.add_argument("--min-rejected-score", type=int, default=15)
    paper_sim.add_argument("--config", default="config.json")

    signal_quality = subparsers.add_parser("signal-quality", help="Backtest raw BUY signal quality by symbol, strategy, and regime.")
    signal_quality.add_argument("--provider", default="bybit")
    signal_quality.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,CCUSDT,XPLUSDT")
    signal_quality.add_argument("--intervals", default="1h,4h")
    signal_quality.add_argument("--lookback", type=int, default=80)
    signal_quality.add_argument("--max-events", type=int, default=700)
    signal_quality.add_argument("--stride", type=int, default=6)
    signal_quality.add_argument("--config", default="config.json")

    learning_monitor = subparsers.add_parser("learning-monitor", help="Inspect ML training health without changing trading behavior.")
    learning_monitor.add_argument("--config", default="config.json")

    source_check = subparsers.add_parser("source-check", help="Check Bybit, Binance, and order book data sources.")
    source_check.add_argument("--symbol", default="BTCUSDT")
    source_check.add_argument("--interval", default="1h")
    source_check.add_argument("--config", default="config.json")

    args = parser.parse_args()

    if args.command == "journal":
        config = load_config(args.config)
        print(json.dumps({"summary": summarize_learning(config), "recent": load_recent(config, args.limit)}, indent=2))
        return

    if args.command == "portfolio":
        config = load_config(args.config)
        portfolio = load_portfolio(config)
        print(json.dumps(
            {
                "cash": portfolio.cash,
                "positions": portfolio.positions,
                "cost_basis": portfolio.cost_basis,
                "peak_prices": portfolio.peak_prices,
                "last_prices": portfolio.last_prices,
                "equity": portfolio.equity,
            },
            indent=2,
        ))
        return

    if args.command == "learn-weekly":
        config = load_config(args.config)
        print(json.dumps(update_weekly_weights(config), indent=2))
        return

    if args.command == "backtest":
        config = load_config(args.config)
        exchange = args.exchange or config.exchange
        candles = fetch_market_klines(exchange, args.symbol, args.interval, args.limit, config.bybit_market_testnet)
        report = backtest_probabilities(candles, args.lookback, args.horizon, args.move_threshold_pct)
        print(json.dumps(report, indent=2))
        return

    if args.command == "calibrate":
        config = load_config(args.config)
        exchange = args.exchange or config.exchange
        candles = fetch_market_klines(exchange, args.symbol, args.interval, args.limit, config.bybit_market_testnet)
        report = backtest_probabilities(candles, args.lookback, args.horizon, args.move_threshold_pct)
        calibration = save_calibration(config, args.symbol, args.interval, report)
        print(json.dumps({"calibration": calibration.__dict__, "backtest": report}, indent=2))
        return

    if args.command == "import-history":
        config = load_config(args.config)
        try:
            if args.provider == "coinapi":
                candles = fetch_coinapi_ohlcv(args.symbol, args.interval, args.limit, args.source_exchange)
            else:
                candles = fetch_historical_market_klines(args.provider, args.symbol, args.interval, args.limit, config.bybit_market_testnet)
            path = save_candles(config, args.provider, args.symbol, args.interval, candles)
            print(json.dumps({"saved": str(path), "candles": len(candles), "symbol": args.symbol.upper(), "interval": args.interval}, indent=2))
        except Exception as exc:
            print(json.dumps({"error": str(exc), "provider": args.provider, "symbol": args.symbol.upper()}, indent=2))
        return

    if args.command == "backtest-history":
        config = load_config(args.config)
        candles = load_candles(config, args.provider, args.symbol, args.interval)
        report = backtest_probabilities(candles, args.lookback, args.horizon, args.move_threshold_pct)
        print(json.dumps(report, indent=2))
        return

    if args.command == "calibrate-history":
        config = load_config(args.config)
        candles = load_candles(config, args.provider, args.symbol, args.interval)
        report = backtest_probabilities(candles, args.lookback, args.horizon, args.move_threshold_pct)
        calibration = save_calibration(config, args.symbol, args.interval, report)
        print(json.dumps({"calibration": calibration.__dict__, "backtest": report}, indent=2))
        return

    if args.command == "market-context":
        symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
        print(json.dumps(fetch_coinmarketcap_quotes(symbols), indent=2))
        return

    if args.command == "train-ml":
        config = load_config(args.config)
        candles = load_candles(config, args.provider, args.symbol, args.interval)
        take_profit_pct = args.take_profit_pct if args.take_profit_pct is not None else args.move_threshold_pct
        stop_loss_pct = args.stop_loss_pct if args.stop_loss_pct is not None else args.move_threshold_pct
        dataset = build_dataset(
            candles,
            args.lookback,
            args.horizon,
            args.move_threshold_pct,
            label_mode=args.label_mode,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
        )
        model, metrics = train_model(
            dataset,
            lookback=args.lookback,
            horizon=args.horizon,
            move_threshold_pct=args.move_threshold_pct,
            label_mode=args.label_mode,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        model_path = f"{config.ml_model_dir}/{args.provider}_{args.symbol.upper()}_{args.interval}_scenario.json"
        save_model(model, model_path)
        print(json.dumps({"model_path": model_path, "metrics": metrics}, indent=2))
        return

    if args.command == "predict-ml":
        config = load_config(args.config)
        model_path = f"{config.ml_model_dir}/{args.provider}_{args.symbol.upper()}_{args.interval}_scenario.json"
        model = load_model(model_path)
        candles = fetch_market_klines(config.exchange, args.symbol, args.interval, max(240, model.lookback), config.bybit_market_testnet)
        print(json.dumps(predict_model(model, candles), indent=2))
        return

    if args.command == "auto-train":
        config = load_config(args.config)
        print(json.dumps(run_auto_training(config, args.provider), indent=2))
        return

    if args.command == "scan-universe":
        config = load_config(args.config)
        print(json.dumps(scan_bybit_universe(config, args.top), indent=2))
        return

    if args.command == "train-daemon":
        config = load_config(args.config)
        run_training_daemon(
            config,
            provider=args.provider,
            sleep_minutes=max(5, args.sleep_minutes),
            scan_universe=not args.no_universe_scan,
        )
        return

    if args.command == "edge-monitor-daemon":
        config = load_config(args.config)
        symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
        run_edge_monitor_daemon(
            config,
            symbols=symbols,
            interval=args.interval,
            exchange=args.exchange or config.exchange,
            limit=args.limit,
            sleep_minutes=max(1, args.sleep_minutes),
            top_ev_symbols=max(0, args.top_ev_symbols),
        )
        return

    if args.command == "edge-backtest":
        config = load_config(args.config)
        symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
        intervals = [interval.strip() for interval in args.intervals.split(",") if interval.strip()]
        report = backtest_edge_filter(
            config,
            provider=args.provider,
            symbols=symbols,
            intervals=intervals,
            lookback=args.lookback,
            max_events_per_model=args.max_events,
            stride=max(1, args.stride),
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "edge-outcomes":
        config = load_config(args.config)
        report = analyze_edge_outcomes(
            config,
            exchange=args.exchange or config.exchange,
            hours=max(1, args.hours),
            limit=max(100, args.limit),
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "blocked-winners-audit":
        config = load_config(args.config)
        report = build_blocked_winners_audit(
            config,
            exchange=args.exchange or config.exchange,
            hours=max(1, args.hours),
            limit=max(100, args.limit),
            min_evaluated=max(1, args.min_evaluated),
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "recovery-watch":
        config = load_config(args.config)
        report = build_recovery_watch_report(
            config,
            exchange=args.exchange or config.exchange,
            limit=max(100, args.limit),
            max_pockets=max(1, args.max_pockets),
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "recovery-shadow":
        config = load_config(args.config)
        report = update_recovery_shadow_trades(
            config,
            exchange=args.exchange or config.exchange,
            limit=max(100, args.limit),
            open_from_watch=not args.no_open_from_watch,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "paper-simulator":
        config = load_config(args.config)
        report = update_paper_simulator(
            config,
            exchange=args.exchange or config.exchange,
            limit=max(100, args.limit),
            source_hours=max(1, args.source_hours),
            max_new=max(0, args.max_new),
            min_rejected_score=max(0, args.min_rejected_score),
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "signal-quality":
        config = load_config(args.config)
        symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
        intervals = [interval.strip() for interval in args.intervals.split(",") if interval.strip()]
        report = build_signal_quality_report(
            config,
            provider=args.provider,
            symbols=symbols,
            intervals=intervals,
            lookback=args.lookback,
            max_events_per_model=args.max_events,
            stride=max(1, args.stride),
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "learning-monitor":
        config = load_config(args.config)
        report = build_learning_monitor_report(config)
        path = save_learning_monitor_report(config, report)
        print(json.dumps({"saved": str(path), "report": report}, indent=2))
        return

    if args.command == "source-check":
        config = load_config(args.config)
        print(json.dumps(_source_check(config, args.symbol, args.interval), indent=2))
        return

    if args.command == "exit-check":
        config = load_config(args.config)
        exchange = args.exchange or config.exchange
        candles = fetch_market_klines(exchange, args.symbol, args.interval, args.limit, config.bybit_market_testnet)
        portfolio = load_portfolio(config)
        exit_decision = evaluate_exit(args.symbol, candles, portfolio, config)
        execution = None
        if args.execute_paper and exit_decision.action == "SELL":
            from trading_agent.models import Action, Decision, StrategyName
            from datetime import datetime, timezone

            sell_decision = Decision(
                action=Action.SELL,
                confidence=exit_decision.confidence,
                selected_strategy=StrategyName.MEAN_REVERSION,
                market_state="Exit engine sell.",
                reasoning=exit_decision.reason,
                risk_level=exit_decision.risk_level,
                learning_note="Exit engine does not modify strategy weights directly.",
                symbol=exit_decision.symbol,
                price=exit_decision.current_price,
                timestamp=datetime.now(timezone.utc),
            )
            state = analyze_market(candles)
            execution = paper_execute(sell_decision, portfolio, state, config)
            append_decision(config, sell_decision, {"exchange": exchange, "paper_execution": execution, "exit_engine": exit_decision.to_dict(), "equity": portfolio.equity})
        save_portfolio(config, portfolio)
        print(json.dumps({"exit_decision": exit_decision.to_dict(), "paper_execution": execution}, indent=2))
        return

    config = load_config(args.config)
    exchange = args.exchange or config.exchange
    candles = fetch_market_klines(exchange, args.symbol, args.interval, args.limit, config.bybit_market_testnet)
    portfolio = load_portfolio(config)
    context = RiskContext(
        daily_pnl_pct=portfolio.daily_pnl_pct,
        consecutive_losses=portfolio.consecutive_losses,
        has_position=portfolio.has_position(args.symbol),
    )
    decision = decide(args.symbol, candles, config, context, args.interval)
    print(_format_decision(decision))

    if args.command == "trade":
        state = analyze_market(candles)
        execution = paper_execute(decision, portfolio, state, config)
        live_execution = None
        if args.live:
            quote_amount = args.live_quote_amount or min(config.max_live_order_quote, execution.get("cost", 0.0) or config.max_live_order_quote)
            try:
                live_execution = place_spot_market_order(decision, quote_amount, config)
            except LiveTradingBlocked as exc:
                live_execution = {"executed": False, "blocked": True, "reason": str(exc)}
            except Exception as exc:
                live_execution = {"executed": False, "blocked": False, "error": str(exc)}
        save_portfolio(config, portfolio)
        append_decision(
            config,
            decision,
            {
                "exchange": exchange,
                "paper_execution": execution,
                "live_execution": live_execution,
                "equity": portfolio.equity,
            },
        )
        print("\nPAPER EXECUTION:")
        print(json.dumps(execution, indent=2))
        if live_execution is not None:
            print("\nLIVE EXECUTION:")
            print(json.dumps(live_execution, indent=2))


def _format_decision(decision) -> str:
    payload = asdict(decision)
    return "\n".join(
        [
            f"1) ACTION: {payload['action'].value}",
            f"2) CONFIDENCE: {payload['confidence']}%",
            f"3) STRATEGY USED: {payload['selected_strategy'].value}",
            f"4) MARKET STATE: {payload['market_state']}",
            f"5) REASONING: {payload['reasoning']}",
            f"6) RISK LEVEL: {payload['risk_level']}",
            f"7) LEARNING NOTE: {payload['learning_note']}",
            f"\nSYMBOL: {payload['symbol']}",
            f"PRICE: {payload['price']:.8f}",
            f"TIME: {payload['timestamp'].isoformat()}",
        ]
    )


def _source_check(config, symbol: str, interval: str) -> dict:
    report = {"symbol": symbol.upper(), "interval": interval, "sources": {}}
    for provider in ("bybit", "binance"):
        try:
            candles = fetch_market_klines(provider, symbol, interval, 80, config.bybit_market_testnet)
            report["sources"][provider] = {
                "status": "ok",
                "candles": len(candles),
                "first": candles[0].open_time.isoformat() if candles else None,
                "last": candles[-1].open_time.isoformat() if candles else None,
                "last_close": candles[-1].close if candles else None,
            }
        except Exception as exc:
            report["sources"][provider] = {"status": "error", "error": str(exc)}

    try:
        snapshot = fetch_bybit_order_book(symbol, limit=50, testnet=config.bybit_market_testnet)
        report["sources"]["bybit_order_book"] = {"status": "ok", **analyze_order_book(snapshot, config.max_live_order_quote).to_dict()}
    except Exception as exc:
        report["sources"]["bybit_order_book"] = {"status": "error", "error": str(exc)}

    return report


if __name__ == "__main__":
    main()
