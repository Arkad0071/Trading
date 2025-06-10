import logging
from data.data_manager import get_candlestick_data
from indicators.indicators import calculate_indicators
from backtesting.backtesting import Backtester
from backtesting.strategies import STRATEGIES

logger = logging.getLogger(__name__)


def evaluate(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 500):
    df = get_candlestick_data(symbol=symbol, timeframe=timeframe, limit=limit, private=False)
    if df.empty:
        logger.error("Нет данных для оценки")
        return

    df = calculate_indicators(df)

    results = []
    for name, strategy in STRATEGIES.items():
        df_copy = strategy(df.copy())
        bt = Backtester(initial_balance=10000)
        bt.run_backtest(df_copy, signal_column="signal")
        results.append((name, bt.balance))

    results.sort(key=lambda x: x[1], reverse=True)
    print("=== Strategy results ===")
    for name, bal in results:
        print(f"{name:12s} -> final balance: {bal:.2f}")
    if results:
        print(f"Best strategy: {results[0][0]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluate()
