# Spot Crypto Trading Agent

A risk-first crypto trading agent for spot markets. It analyzes market state, selects one predefined strategy, and returns a structured decision:

1. ACTION: BUY / SELL / NO TRADE
2. CONFIDENCE: 0-100%
3. SELECTED STRATEGY
4. MARKET STATE
5. REASONING
6. RISK LEVEL

The default mode is paper trading. Live trading is intentionally not implemented in this first version because capital preservation and testing come first.

The agent also estimates scenario probabilities for bullish, bearish, and sideways outcomes. These probabilities do not predict the future with certainty; they act as an additional confirmation layer before risk rules allow BUY or SELL.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m trading_agent.cli analyze --symbol BTCUSDT --interval 1h
```

By default the app uses Bybit Spot through `pybit`. You can still use Binance public candles:

```powershell
python -m trading_agent.cli analyze --exchange binance --symbol BTCUSDT --interval 1h
```

## Paper Trade

```powershell
python -m trading_agent.cli trade --symbol BTCUSDT --interval 1h
```

## Review Journal

```powershell
python -m trading_agent.cli journal
```

## Historical Learning

Import public historical candles in batches and train a local scenario model:

```powershell
python -m trading_agent.cli import-history --config config.demo.json --provider bybit --symbol LTCUSDT --interval 1h --limit 2500
python -m trading_agent.cli train-ml --config config.demo.json --provider bybit --symbol LTCUSDT --interval 1h
python -m trading_agent.cli predict-ml --config config.demo.json --provider bybit --symbol LTCUSDT --interval 1h
```

Run continuous local training cycles:

```powershell
python -m trading_agent.cli train-daemon --config config.demo.json --provider bybit --sleep-minutes 30
```

Create `data/models/training_daemon.stop` to stop the daemon after its current sleep/check cycle.

## Bybit API Keys

Use environment variables. Do not put keys inside source code.

```powershell
$env:BYBIT_API_KEY="your_key"
$env:BYBIT_API_SECRET="your_secret"
```

The CLI also loads a local `.env` file if it exists. `.env` is ignored by Git.

Live orders are blocked unless all of these are true:

- `live_trading_enabled` is `true` in `config.json`.
- `ENABLE_LIVE_TRADING=true` exists in the shell.
- You pass `--live`.
- The decision is BUY or SELL after risk checks.
- The quote amount is below `max_live_order_quote`.

Example:

```powershell
python -m trading_agent.cli trade --exchange bybit --symbol BTCUSDT --interval 1h --live --live-quote-amount 10
```

Demo/testnet config is available in `config.demo.json`:

```powershell
$env:ENABLE_LIVE_TRADING="true"
python -m trading_agent.cli trade --config config.demo.json --exchange bybit --symbol BTCUSDT --interval 1h --live --live-quote-amount 5
```

## Safety Defaults

- Spot only.
- No leverage, futures, or margin.
- No trading in unclear conditions.
- Daily loss limit: 3%.
- Risk per trade: 1%.
- Stop trading after 3 consecutive losses.
- Minimum confidence to buy: 70%.

Settings live in `config.example.json`. Copy it to `config.json` if you want custom values.
