import logging
from data.data_manager import fetch_last_two_years_to_csv


logging.basicConfig(level=logging.INFO, format="%(message)s")

if __name__ == "__main__":
    fetch_last_two_years_to_csv(symbol="BTC/USDT", timeframe="1h", private=True)
