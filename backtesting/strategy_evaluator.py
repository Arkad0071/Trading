import logging
from data.data_manager import get_candlestick_data
from indicators.indicators import calculate_indicators
from backtesting.backtesting import Backtester
from backtesting.strategies import STRATEGIES, generate_all_indicator_strategies

logger = logging.getLogger(__name__)


def evaluate(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 500,
    include_combos: bool = False,
):
    df = get_candlestick_data(symbol=symbol, timeframe=timeframe, limit=limit, private=False)
    if df.empty:
        logger.error("Нет данных для оценки")
        return

    df = calculate_indicators(df)

    strategies = STRATEGIES.copy()
    if include_combos:
        strategies.update(generate_all_indicator_strategies())

    results = []
    for name, strategy in strategies.items():
        df_copy = strategy(df.copy())
        bt = Backtester(initial_balance=100000)
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-combos",
        action="store_true",
        help="Evaluate every combination of indicators",
    )
    args = parser.parse_args()

    evaluate(include_combos=args.all_combos)
