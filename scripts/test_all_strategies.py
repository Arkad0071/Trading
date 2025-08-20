#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт для тестирования всех стратегий на исторических данных
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импорты из нашего проекта
from data.data_manager import load_ohlcv_from_csv
from indicators.indicators import (
    calculate_rsi, calculate_macd, calculate_atr, calculate_sma, 
    calculate_ema, calculate_bollinger_bands, calculate_stochastic
)
from indicators.enhanced_indicators import calculate_all_enhanced_indicators
from strategies.enhanced_strategies import get_all_enhanced_strategies
from backtesting.enhanced_backtester import EnhancedBacktester, StrategyComparator
from visualization.strategy_visualizer import StrategyVisualizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_testing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_and_prepare_data(csv_path="data/btc_usdt_1h_2y.csv"):
    """
    Загружает и подготавливает данные для тестирования
    
    Args:
        csv_path: Путь к CSV файлу с данными
        
    Returns:
        DataFrame с подготовленными данными
    """
    logger.info(f"Загружаю данные из {csv_path}")
    
    # Загружаем данные
    df = load_ohlcv_from_csv(csv_path)
    
    if df.empty:
        logger.error("Не удалось загрузить данные")
        return None
    
    logger.info(f"Загружено {len(df)} записей")
    
    # Преобразуем временные метки
    if 'start_at' in df.columns:
        df['timestamp'] = pd.to_datetime(df['start_at'])
        df.set_index('timestamp', inplace=True)
    
    # Сортируем по времени
    df.sort_index(inplace=True)
    
    # Убираем дубликаты
    df = df[~df.index.duplicated(keep='first')]
    
    logger.info(f"Данные подготовлены. Период: {df.index[0]} - {df.index[-1]}")
    
    return df

def calculate_all_indicators(df):
    """
    Рассчитывает все базовые и расширенные индикаторы
    
    Args:
        df: DataFrame с данными OHLCV
        
    Returns:
        DataFrame с рассчитанными индикаторами
    """
    logger.info("Начинаю расчет всех индикаторов...")
    
    try:
        # Базовые индикаторы
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_atr(df)
        df = calculate_sma(df)
        df = calculate_ema(df)
        df = calculate_bollinger_bands(df)
        df = calculate_stochastic(df)
        
        logger.info("Базовые индикаторы рассчитаны")
        
        # Расширенные индикаторы
        df = calculate_all_enhanced_indicators(df)
        
        logger.info("Все индикаторы рассчитаны успешно")
        
        # Убираем строки с NaN значениями
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)
        
        logger.info(f"Убрано {initial_len - final_len} строк с NaN. Осталось: {final_len}")
        
        return df
        
    except Exception as e:
        logger.error(f"Ошибка при расчете индикаторов: {str(e)}")
        return None

def test_single_strategy(df, strategy_name, strategy_func):
    """
    Тестирует одну стратегию
    
    Args:
        df: DataFrame с данными и индикаторами
        strategy_name: Название стратегии
        strategy_func: Функция стратегии
        
    Returns:
        Словарь с результатами
    """
    logger.info(f"Тестирую стратегию: {strategy_name}")
    
    try:
        # Применяем стратегию
        df_with_signals = strategy_func(df.copy())
        
        # Проверяем наличие сигналов
        if 'signal' not in df_with_signals.columns:
            logger.warning(f"Стратегия {strategy_name} не создала колонку 'signal'")
            return None
        
        # Запускаем бэктест
        backtester = EnhancedBacktester(initial_balance=10000, commission_rate=0.001)
        metrics = backtester.run_backtest(df_with_signals, strategy_name=strategy_name)
        
        # Получаем данные о сделках
        trades_df = backtester.get_trades_dataframe()
        
        # Создаем визуализатор
        visualizer = StrategyVisualizer(df_with_signals, strategy_name)
        
        # Создаем отчет
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"reports/{strategy_name}_{timestamp}"
        
        try:
            visualizer.create_strategy_report(trades_df, metrics, report_dir)
            logger.info(f"Отчет по стратегии {strategy_name} создан в {report_dir}")
        except Exception as e:
            logger.warning(f"Не удалось создать отчет для {strategy_name}: {str(e)}")
        
        return {
            'metrics': metrics,
            'trades': trades_df,
            'backtester': backtester,
            'visualizer': visualizer
        }
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании стратегии {strategy_name}: {str(e)}")
        return None

def test_all_strategies(df):
    """
    Тестирует все доступные стратегии
    
    Args:
        df: DataFrame с данными и индикаторами
        
    Returns:
        Словарь с результатами всех стратегий
    """
    logger.info("Начинаю тестирование всех стратегий...")
    
    # Получаем все стратегии
    strategies = get_all_enhanced_strategies()
    
    # Создаем компаратор
    comparator = StrategyComparator(initial_balance=10000, commission_rate=0.001)
    
    # Тестируем каждую стратегию
    results = {}
    
    for strategy_name, strategy_func in strategies.items():
        logger.info(f"Тестирую стратегию: {strategy_name}")
        
        result = test_single_strategy(df, strategy_name, strategy_func)
        
        if result:
            results[strategy_name] = result
            logger.info(f"Стратегия {strategy_name} протестирована успешно")
        else:
            logger.warning(f"Стратегия {strategy_name} не прошла тестирование")
    
    logger.info(f"Тестирование завершено. Успешно протестировано {len(results)} стратегий")
    
    return results

def create_summary_report(results):
    """
    Создает сводный отчет по всем стратегиям
    
    Args:
        results: Словарь с результатами стратегий
    """
    if not results:
        logger.warning("Нет результатов для создания отчета")
        return
    
    logger.info("Создаю сводный отчет...")
    
    # Создаем DataFrame с результатами
    summary_data = []
    
    for strategy_name, result in results.items():
        metrics = result['metrics']
        
        summary_data.append({
            'Strategy': strategy_name,
            'Total Return (%)': metrics.get('Total Return (%)', 0),
            'Win Rate (%)': metrics.get('Win Rate (%)', 0),
            'Total Trades': metrics.get('Total Trades', 0),
            'Profit Factor': metrics.get('Profit Factor', 0),
            'Max Drawdown (%)': metrics.get('Max Drawdown (%)', 0),
            'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
            'Sortino Ratio': metrics.get('Sortino Ratio', 0),
            'Calmar Ratio': metrics.get('Calmar Ratio', 0),
            'Final Balance': metrics.get('Final Balance', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Total Return (%)', ascending=False)
    
    # Сохраняем сводку
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"reports/strategy_summary_{timestamp}.csv"
    
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"Сводный отчет сохранен в {summary_path}")
    
    # Выводим топ-5 стратегий
    print("\n" + "="*80)
    print("ТОП-5 СТРАТЕГИЙ ПО ДОХОДНОСТИ:")
    print("="*80)
    print(summary_df.head().to_string(index=False))
    print("="*80)
    
    return summary_df

def main():
    """
    Главная функция
    """
    logger.info("Запуск тестирования всех стратегий")
    
    try:
        # 1. Загружаем данные
        df = load_and_prepare_data()
        if df is None:
            logger.error("Не удалось загрузить данные. Завершение работы.")
            return
        
        # 2. Рассчитываем индикаторы
        df = calculate_all_indicators(df)
        if df is None:
            logger.error("Не удалось рассчитать индикаторы. Завершение работы.")
            return
        
        logger.info(f"Данные подготовлены. Размер: {df.shape}")
        
        # 3. Тестируем все стратегии
        results = test_all_strategies(df)
        
        if not results:
            logger.error("Не удалось протестировать ни одной стратегии")
            return
        
        # 4. Создаем сводный отчет
        summary_df = create_summary_report(results)
        
        # 5. Создаем компаратор для визуализации
        comparator = StrategyComparator(initial_balance=10000, commission_rate=0.001)
        
        # Подготавливаем данные для компаратора
        strategies_dict = {}
        for strategy_name, result in results.items():
            # Создаем функцию-обертку для компаратора
            def create_strategy_wrapper(strat_name, strat_result):
                def wrapper(df_copy):
                    return df_copy.copy()  # Возвращаем копию с уже примененными сигналами
                return wrapper
            
            strategies_dict[strategy_name] = create_strategy_wrapper(strategy_name, result)
        
        # 6. Создаем графики сравнения
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = f"reports/comparison_{timestamp}"
        os.makedirs(comparison_dir, exist_ok=True)
        
        # График сравнения по доходности
        comparator.plot_comparison(
            metric='Total Return (%)', 
            save_path=f"{comparison_dir}/return_comparison.png"
        )
        
        # График сравнения по win rate
        comparator.plot_comparison(
            metric='Win Rate (%)', 
            save_path=f"{comparison_dir}/winrate_comparison.png"
        )
        
        # График сравнения по Sharpe ratio
        comparator.plot_comparison(
            metric='Sharpe Ratio', 
            save_path=f"{comparison_dir}/sharpe_comparison.png"
        )
        
        logger.info("Тестирование завершено успешно!")
        logger.info(f"Все отчеты сохранены в директории reports/")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    main()
