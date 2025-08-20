#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой тест системы для проверки работоспособности
"""

import sys
import os
import pandas as pd
import numpy as np

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Тестирует базовую функциональность"""
    print("Тестирование базовой функциональности...")
    
    try:
        # 1. Загружаем данные
        from data.data_manager import load_ohlcv_from_csv
        df = load_ohlcv_from_csv("data/btc_usdt_1h_2y.csv")
        print(f"✓ Данные загружены: {len(df)} записей")
        
        # 2. Рассчитываем базовые индикаторы
        from indicators.indicators import (
            calculate_rsi, calculate_macd, calculate_atr, 
            calculate_sma, calculate_ema, calculate_bollinger_bands
        )
        
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_atr(df)
        df = calculate_sma(df)
        df = calculate_ema(df)  # EMA_20
        df = calculate_ema(df, period=50)  # EMA_50 для стратегии
        df = calculate_bollinger_bands(df)
        print("✓ Базовые индикаторы рассчитаны")
        
        # 3. Проверяем наличие колонок
        required_columns = ['RSI', 'MACD', 'MACD_signal', 'ATR', 'SMA_20', 'EMA_20', 'EMA_50', 'BB_upper', 'BB_lower']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"✗ Отсутствуют колонки: {missing_columns}")
            return False
        else:
            print("✓ Все базовые индикаторы присутствуют")
        
        # 4. Тестируем простую стратегию
        from strategies.enhanced_strategies import adaptive_momentum_strategy
        
        df_with_signals = adaptive_momentum_strategy(df.copy())
        
        if 'signal' in df_with_signals.columns:
            print("✓ Стратегия применена успешно")
            signal_counts = df_with_signals['signal'].value_counts()
            print(f"  Сигналы: {dict(signal_counts)}")
        else:
            print("✗ Стратегия не создала сигналы")
            return False
        
        # 5. Тестируем бэктестер
        from backtesting.enhanced_backtester import EnhancedBacktester
        
        backtester = EnhancedBacktester(initial_balance=10000, commission_rate=0.001)
        metrics = backtester.run_backtest(df_with_signals, strategy_name="Test Strategy")
        
        if metrics:
            print("✓ Бэктест выполнен успешно")
            print(f"  Финальный баланс: ${metrics.get('Final Balance', 0):.2f}")
            print(f"  Общая прибыль: ${metrics.get('Total Profit', 0):.2f}")
        else:
            print("✗ Бэктест не выполнен")
            return False
        
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        return True
        
    except Exception as e:
        print(f"✗ Ошибка при тестировании: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization():
    """Тестирует визуализацию"""
    print("\nТестирование визуализации...")
    
    try:
        from visualization.strategy_visualizer import StrategyVisualizer
        
        # Загружаем данные с индикаторами
        from data.data_manager import load_ohlcv_from_csv
        from indicators.indicators import calculate_rsi, calculate_macd, calculate_atr
        
        df = load_ohlcv_from_csv("data/btc_usdt_1h_2y.csv")
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_atr(df)
        
        # Создаем визуализатор
        visualizer = StrategyVisualizer(df, "Test Strategy")
        print("✓ Визуализатор создан")
        
        # Проверяем методы
        if hasattr(visualizer, 'plot_strategy_overview'):
            print("✓ Метод plot_strategy_overview доступен")
        else:
            print("✗ Метод plot_strategy_overview недоступен")
            return False
        
        print("✓ Визуализация работает")
        return True
        
    except Exception as e:
        print(f"✗ Ошибка при тестировании визуализации: {str(e)}")
        return False

def main():
    """Главная функция"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ТОРГОВОЙ СИСТЕМЫ")
    print("=" * 60)
    
    # Тест 1: Базовая функциональность
    test1_passed = test_basic_functionality()
    
    # Тест 2: Визуализация
    test2_passed = test_visualization()
    
    # Итоговый результат
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ:")
    print("=" * 60)
    
    if test1_passed:
        print("✓ Базовая функциональность: ПРОЙДЕНА")
    else:
        print("✗ Базовая функциональность: ПРОВАЛЕНА")
    
    if test2_passed:
        print("✓ Визуализация: ПРОЙДЕНА")
    else:
        print("✗ Визуализация: ПРОВАЛЕНА")
    
    if test1_passed and test2_passed:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! СИСТЕМА ГОТОВА К РАБОТЕ!")
        print("\nСледующие шаги:")
        print("1. Запустите: python scripts/test_all_strategies.py")
        print("2. Просмотрите отчеты в папке reports/")
        print("3. Анализируйте результаты и оптимизируйте стратегии")
    else:
        print("\n❌ ЕСТЬ ПРОБЛЕМЫ! Нужно исправить ошибки перед продолжением.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
