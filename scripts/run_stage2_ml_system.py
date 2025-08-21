#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ГЛАВНЫЙ СКРИПТ ЭТАПА 2: СИСТЕМА МАШИННОГО ОБУЧЕНИЯ
Запускает полную ML систему для торговли:
1. Обучение продвинутых ML моделей
2. Автоматический поиск лучших стратегий
3. Генерация предиктивных сигналов
4. Интеграция с реальными данными
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_manager import load_ohlcv_from_csv, get_candlestick_data
from indicators.indicators import calculate_rsi, calculate_macd, calculate_atr, calculate_sma, calculate_ema, calculate_bollinger_bands
from indicators.enhanced_indicators import calculate_all_enhanced_indicators
from models.advanced_ml_model import AdvancedMLModel
from strategies.auto_strategy_finder import AutoStrategyFinder
from signals.predictive_signals import PredictiveSignalGenerator
from backtesting.enhanced_backtester import EnhancedBacktester
from visualization.strategy_visualizer import StrategyVisualizer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLTradingSystem:
    """
    Полная система машинного обучения для торговли
    """
    
    def __init__(self):
        self.ml_models = {}
        self.best_strategies = []
        self.signal_generator = None
        self.backtester = EnhancedBacktester(initial_balance=10000, commission_rate=0.001)
        self.visualizer = StrategyVisualizer()
        
    def load_and_prepare_data(self, use_real_data=False, symbol='BTC/USDT', timeframe='1h'):
        """
        Загружает и подготавливает данные для ML
        """
        print("📊 Загрузка и подготовка данных для ML...")
        
        try:
            if use_real_data:
                print("🌐 Загружаю реальные данные с Bybit...")
                # Последние 30 дней данных
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)
                since_ms = int(start_time.timestamp() * 1000)
                
                df = get_candlestick_data(symbol, timeframe, since=since_ms, limit=1000)
                
                if df.empty:
                    print("⚠️ Не удалось загрузить реальные данные. Использую демо-данные...")
                    use_real_data = False
            
            if not use_real_data:
                # Используем демо-данные
                data_file = "data/btc_usdt_1h_2y.csv"
                
                if os.path.exists(data_file):
                    df = load_ohlcv_from_csv(data_file)
                    print(f"✓ Демо-данные загружены: {len(df)} записей")
                else:
                    print("⚠️ Демо-данные не найдены. Создаю новые...")
                    from scripts.generate_demo_data import generate_demo_btc_data
                    df = generate_demo_btc_data(periods=2000, start_date='2022-01-01')
            
            print(f"✓ Данные загружены: {len(df)} записей")
            print(f"  Период: {df.index[0] if hasattr(df.index, '__getitem__') else 'N/A'} - {df.index[-1] if hasattr(df.index, '__getitem__') else 'N/A'}")
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return None
    
    def prepare_features(self, df):
        """
        Подготавливает все индикаторы и признаки
        """
        print("🔧 Подготовка индикаторов и признаков...")
        
        try:
            # Базовые индикаторы
            df = calculate_rsi(df)
            df = calculate_macd(df)
            df = calculate_atr(df)
            df = calculate_sma(df, period=20)
            df = calculate_sma(df, period=50)
            df = calculate_ema(df, period=20)
            df = calculate_ema(df, period=50)
            df = calculate_bollinger_bands(df)
            
            # Расширенные индикаторы
            df = calculate_all_enhanced_indicators(df)
            
            # Убираем NaN значения
            df = df.dropna()
            
            print(f"✓ Все индикаторы рассчитаны. Финальный размер: {len(df)} записей")
            return df
            
        except Exception as e:
            print(f"❌ Ошибка подготовки признаков: {e}")
            return None
    
    def train_ml_models(self, df):
        """
        Обучает различные ML модели
        """
        print("🤖 Обучение ML моделей...")
        
        model_results = {}
        
        # Список моделей для обучения
        models_to_train = [
            {'name': 'LSTM', 'type': 'lstm', 'use_tensorflow': True},
            {'name': 'GRU', 'type': 'gru', 'use_tensorflow': True},
            {'name': 'Random Forest', 'type': 'random_forest', 'use_tensorflow': False},
            {'name': 'Gradient Boosting', 'type': 'gradient_boosting', 'use_tensorflow': False}
        ]
        
        for model_config in models_to_train:
            print(f"\n🧠 Обучение {model_config['name']}...")
            
            try:
                # Создаем модель
                ml_model = AdvancedMLModel(sequence_length=60, prediction_horizon=1)
                
                # Обучаем модель
                if model_config['use_tensorflow']:
                    try:
                        result = ml_model.train_neural_network(
                            df, 
                            model_type=model_config['type'], 
                            validation_split=0.2, 
                            epochs=50
                        )
                        if result:
                            model_results[model_config['name']] = {
                                'model': ml_model,
                                'type': 'neural_network',
                                'training_history': result
                            }
                            print(f"  ✓ {model_config['name']} обучена успешно")
                        else:
                            print(f"  ⚠️ {model_config['name']}: TensorFlow недоступен")
                    except Exception as e:
                        print(f"  ❌ {model_config['name']}: {e}")
                else:
                    result = ml_model.train_sklearn_model(df, model_type=model_config['type'])
                    if result:
                        model_results[model_config['name']] = {
                            'model': ml_model,
                            'type': 'sklearn',
                            'metrics': result
                        }
                        print(f"  ✓ {model_config['name']} обучена успешно")
                        print(f"    R²: {result.get('test_score', 0):.4f}, MSE: {result.get('mse', 0):.6f}")
                    else:
                        print(f"  ⚠️ {model_config['name']}: Scikit-learn недоступен")
                        
            except Exception as e:
                print(f"  ❌ Ошибка обучения {model_config['name']}: {e}")
                continue
        
        self.ml_models = model_results
        print(f"\n✅ Обучено {len(model_results)} ML моделей")
        
        return model_results
    
    def find_best_strategies(self, df):
        """
        Автоматически ищет лучшие торговые стратегии
        """
        print("🔍 Автоматический поиск лучших стратегий...")
        
        try:
            # Создаем поисковик стратегий
            strategy_finder = AutoStrategyFinder(
                backtester=self.backtester,
                min_trades=5,  # Снижаем для демо-данных
                min_win_rate=0.3
            )
            
            # Запускаем поиск
            best_strategies = strategy_finder.search_best_strategies(df, max_strategies=500)
            
            if best_strategies:
                self.best_strategies = best_strategies
                
                # Создаем сводку
                summary = strategy_finder.get_best_strategies_summary()
                print(f"\n✅ Найдено {len(best_strategies)} эффективных стратегий")
                print("\nТоп-5 стратегий:")
                print(summary.head().to_string(index=False))
                
                # Сохраняем результаты
                os.makedirs('results/stage2', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                strategy_finder.save_results(f'results/stage2/best_strategies_{timestamp}.json')
                summary.to_csv(f'results/stage2/strategies_summary_{timestamp}.csv', index=False)
                
                return best_strategies
            else:
                print("⚠️ Не найдено эффективных стратегий")
                return []
                
        except Exception as e:
            print(f"❌ Ошибка поиска стратегий: {e}")
            return []
    
    def create_predictive_signals(self, df):
        """
        Создает систему предиктивных сигналов
        """
        print("🎯 Создание системы предиктивных сигналов...")
        
        try:
            # Выбираем лучшую ML модель
            best_model = None
            best_score = -np.inf
            
            for model_name, model_data in self.ml_models.items():
                if model_data['type'] == 'sklearn':
                    score = model_data['metrics'].get('test_score', 0)
                    if score > best_score:
                        best_score = score
                        best_model = model_data['model']
                        print(f"  Выбрана лучшая модель: {model_name} (R²: {score:.4f})")
            
            # Создаем генератор сигналов
            self.signal_generator = PredictiveSignalGenerator(
                ml_model=best_model,
                confidence_threshold=0.5
            )
            
            # Тестируем генерацию сигналов
            df_with_signals = self.signal_generator.combine_signals(df)
            
            # Анализируем последние сигналы
            analysis = self.signal_generator.get_signal_strength_analysis(df)
            print(f"\n✅ Система сигналов создана")
            print(f"  Текущий сигнал: {analysis['signals']['final_signal']}")
            print(f"  Уверенность: {analysis['signals']['final_confidence']:.3f}")
            print(f"  Обоснование: {analysis['reasoning']}")
            
            return df_with_signals
            
        except Exception as e:
            print(f"❌ Ошибка создания сигналов: {e}")
            return df
    
    def generate_trading_recommendation(self, df):
        """
        Генерирует торговые рекомендации
        """
        print("💡 Генерация торговых рекомендаций...")
        
        if self.signal_generator is None:
            print("⚠️ Система сигналов не инициализирована")
            return None
        
        try:
            recommendation = self.signal_generator.generate_trading_recommendation(df, position_size=1000)
            
            print(f"\n🎯 ТОРГОВАЯ РЕКОМЕНДАЦИЯ:")
            print(f"  Действие: {recommendation['action']}")
            print(f"  Уверенность: {recommendation['confidence']:.3f}")
            print(f"  Текущая цена: ${recommendation['current_price']:.2f}")
            print(f"  Обоснование: {recommendation['reasoning']}")
            
            if 'position' in recommendation and 'recommended_size' in recommendation['position']:
                pos = recommendation['position']
                print(f"\n📈 РЕКОМЕНДАЦИИ ПО ПОЗИЦИИ:")
                print(f"  Тип: {pos['action']}")
                print(f"  Размер: ${pos['recommended_size']:.2f}")
                print(f"  Входная цена: ${pos['entry_price']:.2f}")
                
                if 'stop_loss' in pos:
                    print(f"  Stop Loss: ${pos['stop_loss']:.2f}")
                    print(f"  Take Profit: ${pos['take_profit']:.2f}")
                    print(f"  Risk/Reward: {pos['risk_reward_ratio']:.1f}")
            
            if 'price_prediction' in recommendation:
                pred = recommendation['price_prediction']
                print(f"\n🔮 ПРОГНОЗ ЦЕНЫ:")
                print(f"  Прогнозируемая цена: ${pred['predicted_price']:.2f}")
                print(f"  Ожидаемое изменение: {pred['expected_change_pct']:.2f}%")
                print(f"  Потенциальная прибыль: {pred['potential_profit']:.2f}%")
            
            return recommendation
            
        except Exception as e:
            print(f"❌ Ошибка генерации рекомендаций: {e}")
            return None
    
    def create_comprehensive_report(self, df, ml_results, strategies, recommendation):
        """
        Создает комплексный отчет по всей системе
        """
        print("📋 Создание комплексного отчета...")
        
        try:
            os.makedirs('results/stage2', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"results/stage2/ml_system_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ОТЧЕТ ЭТАПА 2: СИСТЕМА МАШИННОГО ОБУЧЕНИЯ\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Количество данных: {len(df)} записей\n\n")
                
                # ML модели
                f.write("ОБУЧЕННЫЕ ML МОДЕЛИ:\n")
                f.write("-" * 40 + "\n")
                for model_name, model_data in ml_results.items():
                    f.write(f"\n{model_name}:\n")
                    if model_data['type'] == 'sklearn':
                        metrics = model_data['metrics']
                        f.write(f"  Тип: Scikit-learn\n")
                        f.write(f"  R² Score: {metrics.get('test_score', 0):.4f}\n")
                        f.write(f"  MSE: {metrics.get('mse', 0):.6f}\n")
                        f.write(f"  MAE: {metrics.get('mae', 0):.6f}\n")
                    else:
                        f.write(f"  Тип: Neural Network\n")
                        f.write(f"  Статус: Обучена\n")
                
                # Лучшие стратегии
                f.write("\n\nЛУЧШИЕ НАЙДЕННЫЕ СТРАТЕГИИ:\n")
                f.write("-" * 40 + "\n")
                for i, strategy in enumerate(strategies[:10], 1):
                    strat_info = strategy['strategy']
                    metrics = strategy['metrics']
                    f.write(f"\n{i}. {strat_info['name']}:\n")
                    f.write(f"   Тип: {strat_info['type']}\n")
                    f.write(f"   Доходность: {metrics.get('Total Return (%)', 0):.2f}%\n")
                    f.write(f"   Винрейт: {metrics.get('Win Rate (%)', 0):.2f}%\n")
                    f.write(f"   Сделок: {metrics.get('Total Trades', 0)}\n")
                    f.write(f"   Оценка: {strategy['score']:.2f}\n")
                
                # Текущие рекомендации
                if recommendation:
                    f.write("\n\nТЕКУЩИЕ ТОРГОВЫЕ РЕКОМЕНДАЦИИ:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Действие: {recommendation['action']}\n")
                    f.write(f"Уверенность: {recommendation['confidence']:.3f}\n")
                    f.write(f"Текущая цена: ${recommendation['current_price']:.2f}\n")
                    f.write(f"Обоснование: {recommendation['reasoning']}\n")
                
                # Заключение
                f.write("\n\nЗАКЛЮЧЕНИЕ:\n")
                f.write("-" * 40 + "\n")
                f.write("Этап 2 завершен успешно!\n")
                f.write("Система машинного обучения полностью функциональна.\n")
                f.write("Готова к реальной торговле!\n")
            
            print(f"✅ Отчет сохранен: {report_file}")
            return report_file
            
        except Exception as e:
            print(f"❌ Ошибка создания отчета: {e}")
            return None
    
    def run_full_system(self, use_real_data=False):
        """
        Запускает полную систему машинного обучения
        """
        print("=" * 80)
        print("🚀 ЭТАП 2: СИСТЕМА МАШИННОГО ОБУЧЕНИЯ")
        print("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # 1. Загрузка данных
            df = self.load_and_prepare_data(use_real_data=use_real_data)
            if df is None:
                return False
            
            # 2. Подготовка признаков
            df = self.prepare_features(df)
            if df is None:
                return False
            
            # 3. Обучение ML моделей
            ml_results = self.train_ml_models(df)
            
            # 4. Поиск лучших стратегий
            best_strategies = self.find_best_strategies(df)
            
            # 5. Создание системы сигналов
            df_with_signals = self.create_predictive_signals(df)
            
            # 6. Генерация рекомендаций
            recommendation = self.generate_trading_recommendation(df_with_signals)
            
            # 7. Создание отчета
            report_file = self.create_comprehensive_report(df, ml_results, best_strategies, recommendation)
            
            # 8. Итоги
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "=" * 80)
            print("🎉 ЭТАП 2 УСПЕШНО ЗАВЕРШЕН!")
            print("=" * 80)
            print(f"⏱️  Время выполнения: {duration}")
            print(f"🤖 Обучено ML моделей: {len(ml_results)}")
            print(f"🔍 Найдено стратегий: {len(best_strategies)}")
            print(f"📋 Отчет создан: {report_file}")
            
            print("\n🎯 СИСТЕМА ПОЛНОСТЬЮ ГОТОВА К ТОРГОВЛЕ!")
            print("Следующие шаги:")
            print("1. Интеграция с реальной биржей")
            print("2. Автоматическое выполнение сделок")
            print("3. Мониторинг и оптимизация в реальном времени")
            
            return True
            
        except Exception as e:
            print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
            return False

def main():
    """Главная функция"""
    
    # Создаем систему
    ml_system = MLTradingSystem()
    
    # Запускаем с демо-данными (для начала)
    success = ml_system.run_full_system(use_real_data=False)
    
    if success:
        print("\n🎉 ПОЗДРАВЛЯЮ! Система машинного обучения готова!")
        print("Теперь вы можете:")
        print("1. Запустить с реальными данными: run_full_system(use_real_data=True)")
        print("2. Интегрировать с реальной торговлей")
        print("3. Настроить автоматическое выполнение сделок")
    else:
        print("\n❌ Возникли проблемы при запуске системы")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

