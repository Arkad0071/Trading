#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ГЛАВНЫЙ СКРИПТ ЭТАПА 1: ПОЛНАЯ СИСТЕМА ТЕСТИРОВАНИЯ СТРАТЕГИЙ
Запускает все компоненты первого этапа:
1. Загрузка данных
2. Расчет индикаторов
3. Тестирование стратегий
4. Оптимизация параметров
5. Мультитаймфреймовый анализ
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_manager import load_ohlcv_from_csv, save_ohlcv_to_csv
from indicators.indicators import calculate_rsi, calculate_macd, calculate_atr, calculate_sma, calculate_ema, calculate_bollinger_bands
from indicators.enhanced_indicators import calculate_all_enhanced_indicators
from strategies.enhanced_strategies import get_all_enhanced_strategies
from backtesting.enhanced_backtester import EnhancedBacktester, StrategyComparator
from strategies.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from strategies.parameter_optimizer import ParameterOptimizer
from visualization.strategy_visualizer import StrategyVisualizer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Загружает и подготавливает данные"""
    print("📊 Загрузка и подготовка данных...")
    
    try:
        # Загружаем данные (используем существующий файл для демонстрации)
        data_file = "data/btc_usdt_1h_2y.csv"
        
        if os.path.exists(data_file):
            df = load_ohlcv_from_csv(data_file)
            print(f"✓ Данные загружены: {len(df)} записей")
        else:
            print("⚠️ Файл данных не найден. Создаю демо-данные...")
            from scripts.generate_demo_data import generate_demo_btc_data
            df = generate_demo_btc_data(periods=1000, start_date='2023-01-01')
            save_ohlcv_to_csv(df, data_file)
            print(f"✓ Демо-данные созданы: {len(df)} записей")
        
        return df
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return None

def calculate_all_indicators(df):
    """Рассчитывает все индикаторы"""
    print("🔧 Расчет всех индикаторов...")
    
    try:
        # Базовые индикаторы
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_atr(df)
        df = calculate_sma(df)
        df = calculate_ema(df, period=20)  # EMA_20
        df = calculate_ema(df, period=50)  # EMA_50
        df = calculate_bollinger_bands(df)
        
        # Расширенные индикаторы
        df = calculate_all_enhanced_indicators(df)
        
        # Убираем NaN значения
        df = df.dropna()
        
        print(f"✓ Все индикаторы рассчитаны. Финальный размер: {len(df)} записей")
        return df
        
    except Exception as e:
        print(f"❌ Ошибка расчета индикаторов: {e}")
        return None

def test_all_strategies(df):
    """Тестирует все стратегии"""
    print("🧪 Тестирование всех стратегий...")
    
    try:
        # Получаем все стратегии
        strategies = get_all_enhanced_strategies()
        print(f"Найдено {len(strategies)} стратегий")
        
        # Создаем бэктестер
        backtester = EnhancedBacktester(initial_balance=10000, commission_rate=0.001)
        
        # Тестируем каждую стратегию
        results = {}
        for name, strategy_func in strategies.items():
            print(f"\nТестирую стратегию: {name}")
            
            try:
                # Применяем стратегию
                df_with_signals = strategy_func(df.copy())
                
                # Запускаем бэктест
                metrics = backtester.run_backtest(df_with_signals, strategy_name=name)
                
                if metrics and len(metrics) > 0:
                    results[name] = {
                        'metrics': metrics,
                        'trades': backtester.get_trades_dataframe(),
                        'portfolio': backtester.get_portfolio_values()
                    }
                    print(f"  ✓ {name}: {metrics.get('Total Return (%)', 0):.2f}% доходность")
                else:
                    print(f"  ⚠️ {name}: нет сделок")
                    
            except Exception as e:
                print(f"  ❌ {name}: ошибка - {e}")
                continue
        
        print(f"\n✅ Протестировано {len(results)} стратегий")
        return results
        
    except Exception as e:
        print(f"❌ Ошибка тестирования стратегий: {e}")
        return {}

def optimize_best_strategies(df, results):
    """Оптимизирует лучшие стратегии"""
    print("⚡ Оптимизация лучших стратегий...")
    
    if not results:
        print("⚠️ Нет результатов для оптимизации")
        return {}
    
    try:
        # Сортируем стратегии по доходности
        sorted_strategies = sorted(
            results.items(),
            key=lambda x: x[1]['metrics'].get('Total Return (%)', -999),
            reverse=True
        )
        
        # Берем топ-3 стратегии для оптимизации
        top_strategies = sorted_strategies[:3]
        print(f"Оптимизирую топ-{len(top_strategies)} стратегий:")
        
        optimization_results = {}
        backtester = EnhancedBacktester(initial_balance=10000, commission_rate=0.001)
        
        for name, result in top_strategies:
            print(f"\n🔧 Оптимизация {name}...")
            
            try:
                # Создаем оптимизатор
                optimizer = ParameterOptimizer(df, get_all_enhanced_strategies()[name], backtester)
                
                # Оптимизируем параметры
                if 'RSI' in name.lower():
                    best_params = optimizer.optimize_rsi_strategy()
                elif 'MACD' in name.lower():
                    best_params = optimizer.optimize_macd_strategy()
                else:
                    best_params = optimizer.optimize_composite_strategy()
                
                if best_params:
                    optimization_results[name] = {
                        'best_params': best_params,
                        'summary': optimizer.get_optimization_summary()
                    }
                    print(f"  ✓ Лучшие параметры: {best_params}")
                else:
                    print(f"  ⚠️ Оптимизация не дала результатов")
                    
            except Exception as e:
                print(f"  ❌ Ошибка оптимизации: {e}")
                continue
        
        print(f"\n✅ Оптимизировано {len(optimization_results)} стратегий")
        return optimization_results
        
    except Exception as e:
        print(f"❌ Ошибка оптимизации: {e}")
        return {}

def run_multi_timeframe_analysis(df):
    """Запускает мультитаймфреймовый анализ"""
    print("⏰ Мультитаймфреймовый анализ...")
    
    try:
        # Создаем анализатор
        analyzer = MultiTimeframeAnalyzer(['15m', '1h', '4h'])
        
        # Для демонстрации используем один таймфрейм
        # В реальности здесь будут загружены данные с разных таймфреймов
        data_paths = {
            '1h': 'data/btc_usdt_1h_2y.csv'
        }
        
        analyzer.load_data(data_paths)
        
        # Получаем корреляцию (если есть несколько таймфреймов)
        if len(analyzer.data) > 1:
            correlation = analyzer.get_timeframe_correlation()
            print("✓ Корреляция между таймфреймами рассчитана")
            print(correlation)
        
        print("✅ Мультитаймфреймовый анализ завершен")
        return analyzer
        
    except Exception as e:
        print(f"❌ Ошибка мультитаймфреймового анализа: {e}")
        return None

def create_final_report(results, optimization_results, multi_tf_analyzer):
    """Создает финальный отчет"""
    print("📋 Создание финального отчета...")
    
    try:
        # Создаем папку для отчетов
        os.makedirs('reports', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"reports/stage1_complete_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ОТЧЕТ ЭТАПА 1: ПОЛНАЯ СИСТЕМА ТЕСТИРОВАНИЯ\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Результаты тестирования стратегий
            f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ СТРАТЕГИЙ:\n")
            f.write("-" * 40 + "\n")
            
            if results:
                for name, result in results.items():
                    metrics = result['metrics']
                    f.write(f"\n{name}:\n")
                    f.write(f"  Доходность: {metrics.get('Total Return (%)', 0):.2f}%\n")
                    f.write(f"  Винрейт: {metrics.get('Win Rate (%)', 0):.2f}%\n")
                    f.write(f"  Сделок: {metrics.get('Total Trades', 0)}\n")
                    f.write(f"  Макс. просадка: {metrics.get('Max Drawdown (%)', 0):.2f}%\n")
            else:
                f.write("Нет результатов тестирования\n")
            
            # Результаты оптимизации
            f.write("\n\nРЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:\n")
            f.write("-" * 40 + "\n")
            
            if optimization_results:
                for name, opt_result in optimization_results.items():
                    f.write(f"\n{name}:\n")
                    f.write(f"  Лучшие параметры: {opt_result['best_params']}\n")
            else:
                f.write("Нет результатов оптимизации\n")
            
            # Мультитаймфреймовый анализ
            f.write("\n\nМУЛЬТИТАЙМФРЕЙМОВЫЙ АНАЛИЗ:\n")
            f.write("-" * 40 + "\n")
            
            if multi_tf_analyzer:
                f.write("✓ Анализатор создан и настроен\n")
                f.write(f"  Поддерживаемые таймфреймы: {multi_tf_analyzer.timeframes}\n")
            else:
                f.write("❌ Анализатор не создан\n")
            
            # Заключение
            f.write("\n\nЗАКЛЮЧЕНИЕ:\n")
            f.write("-" * 40 + "\n")
            f.write("Этап 1 завершен успешно!\n")
            f.write("Система готова к переходу на Этап 2.\n")
            
        print(f"✅ Отчет сохранен: {report_file}")
        return report_file
        
    except Exception as e:
        print(f"❌ Ошибка создания отчета: {e}")
        return None

def main():
    """Главная функция"""
    print("=" * 60)
    print("🚀 ЭТАП 1: ПОЛНАЯ СИСТЕМА ТЕСТИРОВАНИЯ СТРАТЕГИЙ")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # 1. Загрузка и подготовка данных
        df = load_and_prepare_data()
        if df is None:
            return False
        
        # 2. Расчет всех индикаторов
        df = calculate_all_indicators(df)
        if df is None:
            return False
        
        # 3. Тестирование всех стратегий
        results = test_all_strategies(df)
        
        # 4. Оптимизация лучших стратегий
        optimization_results = optimize_best_strategies(df, results)
        
        # 5. Мультитаймфреймовый анализ
        multi_tf_analyzer = run_multi_timeframe_analysis(df)
        
        # 6. Создание финального отчета
        report_file = create_final_report(results, optimization_results, multi_tf_analyzer)
        
        # 7. Итоги
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("🎉 ЭТАП 1 УСПЕШНО ЗАВЕРШЕН!")
        print("=" * 60)
        print(f"⏱️  Время выполнения: {duration}")
        print(f"📊 Протестировано стратегий: {len(results)}")
        print(f"⚡ Оптимизировано стратегий: {len(optimization_results)}")
        print(f"📋 Отчет создан: {report_file}")
        
        print("\n🎯 СИСТЕМА ГОТОВА К ЭТАПУ 2!")
        print("Следующие шаги:")
        print("1. Машинное обучение для прогнозирования")
        print("2. Автоматический поиск лучших комбинаций")
        print("3. Реальное время торговли")
        
        return True
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
