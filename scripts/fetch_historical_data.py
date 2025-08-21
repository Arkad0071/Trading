#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для загрузки исторических данных BTC/USDT с Bybit
Таймфреймы: 15m, 1h, 4h
Период: последний год
"""

import os
import sys
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_bybit_client
from data.data_manager import save_ohlcv_to_csv

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_historical_data(symbol='BTC/USDT', timeframes=['15m', '1h', '4h'], days_back=365):
    """
    Загружает исторические данные с Bybit
    
    Args:
        symbol: Торговая пара
        timeframes: Список таймфреймов
        days_back: Количество дней назад
    """
    
    try:
        # Получаем клиент Bybit
        exchange = get_bybit_client()
        logger.info(f"Подключение к Bybit установлено")
        
        # Рассчитываем время начала
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        logger.info(f"Загружаю данные для {symbol}")
        logger.info(f"Период: {start_time.strftime('%Y-%m-%d')} - {end_time.strftime('%Y-%m-%d')}")
        logger.info(f"Таймфреймы: {timeframes}")
        
        for timeframe in timeframes:
            logger.info(f"\nЗагружаю данные для таймфрейма {timeframe}...")
            
            # Загружаем данные
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=int(start_time.timestamp() * 1000),
                limit=10000  # Максимальный лимит
            )
            
            if not ohlcv:
                logger.warning(f"Нет данных для таймфрейма {timeframe}")
                continue
            
            # Конвертируем в DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['start_at'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Убираем дубликаты и сортируем
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            logger.info(f"Загружено {len(df)} записей для {timeframe}")
            logger.info(f"Период данных: {df['start_at'].min()} - {df['start_at'].max()}")
            
            # Сохраняем в CSV
            filename = f"data/btc_usdt_{timeframe.replace('m', 'min').replace('h', 'h')}_1y.csv"
            save_ohlcv_to_csv(df, filename)
            
            # Небольшая пауза между запросами
            time.sleep(1)
        
        logger.info("\n✅ Все данные успешно загружены!")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        return False

def main():
    """Главная функция"""
    print("=" * 60)
    print("ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ BTC/USDT")
    print("=" * 60)
    
    # Создаем папку data если её нет
    os.makedirs('data', exist_ok=True)
    
    # Загружаем данные
    success = fetch_historical_data()
    
    if success:
        print("\n🎉 Данные успешно загружены!")
        print("\nСледующие шаги:")
        print("1. Запустите: python scripts/test_all_strategies.py")
        print("2. Запустите: python scripts/auto_optimizer.py")
    else:
        print("\n❌ Ошибка при загрузке данных!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
