# data/data_manager.py
import ccxt
import pandas as pd
import sqlite3
from datetime import datetime
import os
import logging
from dotenv import load_dotenv
from utils.config import BYBIT_API_KEY, BYBIT_API_SECRET



load_dotenv()
logger = logging.getLogger(__name__)

def init_exchange():
    """
    Публичный клиент CCXT для fetch_ohlcv,
    без автоматического вызова fetch_currencies().
    """
    exchange = ccxt.bybit({
        'enableRateLimit': True,
    })
    # Отключаем fetch_currencies, чтобы не вызывалось privateGetV5AssetCoinQueryInfo
    exchange.options['fetchCurrencies'] = False
    return exchange


def get_candlestick_data(symbol="BTC/USDT", timeframe="1h", since=None, limit=None):
    """Получает данные OHLCV и возвращает их в виде DataFrame."""
    exchange = init_exchange()
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
