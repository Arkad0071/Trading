# Trading Bot

This project contains a cryptocurrency trading bot with a Telegram interface. It uses the [CCXT](https://github.com/ccxt/ccxt) library to communicate with the Bybit exchange and includes tools for data collection, indicator calculations, backtesting and automated order execution. Machine learning models (an LSTM network) can be trained to generate trading signals, and the bot features basic risk management utilities and position tracking via SQLite.

## Setup

1. Install Python 3 and create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and provide your own values (see variables below).
4. Run tests to make sure everything works:
   ```bash
   pytest
   ```
5. Start the Telegram bot:
   ```bash
   python telegram_bot/bot.py
   ```

The backtesting engine can be executed directly using `python backtesting/backtesting.py`.

## Environment Variables

The application expects the following variables in the `.env` file:

| Variable | Description |
|----------|-------------|
| `TELEGRAM_TOKEN` | Telegram bot token issued by BotFather. |
| `BYBIT_API_KEY` | API key for your Bybit account. |
| `BYBIT_API_SECRET` | Secret key for the Bybit API. |
| `COMMISSION_RATE` | Trading commission rate (default `0.001`). |
| `BYBIT_MARGIN_MODE` | Margin mode for orders (`cross` or `isolated`). |
| `BYBIT_LEVERAGE` | Leverage value to use when placing orders. |

