# data/data_manager.py
import ccxt
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os
import logging
from dotenv import load_dotenv
from utils.bybit_client import get_public_client, get_private_client



load_dotenv()
logger = logging.getLogger(__name__)

def init_exchange(private: bool = False):
    """Return a CCXT Bybit client.

    Parameters
    ----------
    private : bool, optional
        If ``True`` an authenticated client is created using API keys.
        Otherwise a public client is returned. Defaults to ``False``.
    """

    if private:
        exchange = get_private_client()
    else:
        exchange = get_public_client()
    return exchange


def get_candlestick_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since: int | None = None,
    limit: int | None = None,
    private: bool = False,
):
    """Получает данные OHLCV и возвращает их в виде DataFrame."""
    exchange = init_exchange(private=private)
    all_rows = []
    fetch_since = since
    try:
        while True:
            logger.info(f"Запрашиваю OHLCV для {symbol} ({timeframe}), since={fetch_since}")
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=500)
            if not batch:
                break
            all_rows.extend(batch)
            if limit and len(all_rows) >= limit:
                all_rows = all_rows[:limit]
                break
            if len(batch) < 500:
                break
            fetch_since = batch[-1][0] + 1
        df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['start_at'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"Получено {len(df)} строк данных для {symbol}.")
        return df
    except Exception:
        logger.exception(f"Ошибка при получении OHLCV для {symbol}:")
        return pd.DataFrame()

def save_to_db(df, symbol, timeframe, db_path="market_data.db"):
    """
    Сохраняет DataFrame в SQLite базу данных.
    """
    conn = sqlite3.connect(db_path)
    df['symbol'] = symbol
    df['timeframe'] = timeframe
    df.to_sql("ohlcv", conn, if_exists="append", index=False)
    conn.close()
    logger.info(f"Сохранено {len(df)} записей для {symbol} в базу данных {db_path}.")


def fetch_last_two_years_to_csv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    csv_path: str = "data/btc_usdt_1h_2y.csv",
    private: bool = False,
):
    """Скачать последние 2 года данных и сохранить их в CSV."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=730)
    since_ms = int(start_time.timestamp() * 1000)
    df = get_candlestick_data(symbol, timeframe, since=since_ms, private=private)
    if not df.empty:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Исторические данные сохранены в {csv_path}")
    return df


def load_ohlcv_from_csv(csv_path: str) -> pd.DataFrame:
    """Загрузить OHLCV данные из CSV-файла."""
    if not os.path.exists(csv_path):
        logger.error(f"Файл {csv_path} не найден")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "start_at" in df.columns:
        df["start_at"] = pd.to_datetime(df["start_at"])
    return df
