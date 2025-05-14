# data/data_manager.py
import ccxt
import pandas as pd
import sqlite3
from datetime import datetime
import os
import logging
from dotenv import load_dotenv
from utils.config import MARGIN_MODE, LEVERAGE


load_dotenv()
logger = logging.getLogger(__name__)

def init_exchange():
    """
    Инициализирует биржу Bybit через ccxt, используя ключи из .env.
    """
    return ccxt.bybit({
        'apiKey': os.getenv("BYBIT_API_KEY"),
        'secret': os.getenv("BYBIT_API_SECRET"),
        'enableRateLimit': True,
    })

def get_candlestick_data(symbol="BTC/USDT", timeframe="1h", since=None):
    """
    Получает данные OHLCV и возвращает их в виде DataFrame.
    """
    exchange = init_exchange()
    try:
        logger.info(f"Запрашиваю OHLCV для {symbol} ({timeframe}), since={since}")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['start_at'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"Получено {len(df)} строк данных для {symbol}.")
        return df
    except Exception as e:
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
