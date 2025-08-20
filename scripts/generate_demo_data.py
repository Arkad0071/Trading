#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для генерации демо-данных BTC для тестирования стратегий
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_demo_btc_data(periods=8760, start_date='2022-01-01'):
    """
    Генерирует демо-данные BTC за 1 год (8760 часов)
    
    Args:
        periods: Количество периодов (часов)
        start_date: Начальная дата
        
    Returns:
        DataFrame с демо-данными OHLCV
    """
    print(f"Генерирую {periods} часов демо-данных BTC начиная с {start_date}")
    
    # Начальная цена BTC
    start_price = 50000
    
    # Генерируем временные метки
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    timestamps = [start_dt + timedelta(hours=i) for i in range(periods)]
    
    # Генерируем цены с реалистичной волатильностью
    np.random.seed(42)  # Для воспроизводимости
    
    # Базовый тренд (рост с коррекциями)
    trend = np.linspace(0, 0.3, periods)  # 30% рост за год
    trend += np.sin(np.linspace(0, 4*np.pi, periods)) * 0.1  # Циклические колебания
    
    # Волатильность
    volatility = 0.02  # 2% в час
    
    # Генерируем цены
    prices = [start_price]
    for i in range(1, periods):
        # Случайное изменение цены
        change = np.random.normal(trend[i] / periods, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Минимальная цена $1000
    
    # Создаем OHLCV данные
    data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        # Генерируем high, low, open, close
        volatility_range = price * volatility
        
        if i == 0:
            open_price = price
        else:
            open_price = prices[i-1]
        
        close_price = price
        
        # High и low с реалистичными границами
        high = max(open_price, close_price) + np.random.uniform(0, volatility_range)
        low = min(open_price, close_price) - np.random.uniform(0, volatility_range)
        
        # Объем (коррелирует с волатильностью)
        base_volume = 1000000  # 1M BTC
        volume_multiplier = 1 + abs(change) * 10  # Больше объема при больших изменениях
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5)
        
        data.append({
            'timestamp': int(ts.timestamp() * 1000),  # Unix timestamp в миллисекундах
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2),
            'start_at': ts.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df = pd.DataFrame(data)
    print(f"Сгенерировано {len(df)} записей")
    print(f"Период: {df['start_at'].iloc[0]} - {df['start_at'].iloc[-1]}")
    print(f"Цена: ${df['close'].iloc[0]:.2f} - ${df['close'].iloc[-1]:.2f}")
    
    return df

def main():
    """Главная функция"""
    print("Генерация демо-данных BTC для тестирования стратегий")
    print("=" * 60)
    
    # Создаем директорию data если её нет
    os.makedirs('data', exist_ok=True)
    
    # Генерируем данные
    df = generate_demo_btc_data(periods=8760, start_date='2022-01-01')
    
    # Сохраняем в CSV
    output_path = 'data/btc_usdt_1h_2y.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nДемо-данные сохранены в {output_path}")
    print(f"Размер файла: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    # Показываем первые и последние записи
    print("\nПервые 5 записей:")
    print(df.head().to_string(index=False))
    
    print("\nПоследние 5 записей:")
    print(df.tail().to_string(index=False))
    
    print("\nСтатистика данных:")
    print(f"Средняя цена: ${df['close'].mean():.2f}")
    print(f"Максимальная цена: ${df['close'].max():.2f}")
    print(f"Минимальная цена: ${df['close'].min():.2f}")
    print(f"Средний объем: {df['volume'].mean():.0f}")
    
    print("\nГенерация завершена успешно!")

if __name__ == "__main__":
    main()
