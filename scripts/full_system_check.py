#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОЛНАЯ ПРОВЕРКА СИСТЕМЫ ПЕРЕД РАЗВИТИЕМ ML
Проверяет все компоненты и дает рекомендации
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemChecker:
    """Полная проверка системы"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_total = 0
        self.issues = []
        self.recommendations = []
    
    def check_config(self):
        """Проверка конфигурации"""
        print("🔧 ПРОВЕРКА КОНФИГУРАЦИИ:")
        self.checks_total += 1
        
        try:
            from utils.config import TOKEN, BYBIT_API_KEY, BYBIT_API_SECRET
            
            if TOKEN:
                print(f"✅ Telegram токен: {TOKEN[:10]}...{TOKEN[-10:]}")
            else:
                print("❌ Telegram токен не найден")
                self.issues.append("TELEGRAM_TOKEN не задан в .env")
            
            if BYBIT_API_KEY and BYBIT_API_SECRET:
                print(f"✅ Bybit API ключи: {BYBIT_API_KEY[:8]}...{BYBIT_API_KEY[-8:]}")
                self.checks_passed += 1
            else:
                print("❌ Bybit API ключи не найдены")
                self.issues.append("BYBIT_API_KEY или BYBIT_API_SECRET не заданы в .env")
                
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации: {e}")
            self.issues.append(f"Ошибка конфигурации: {e}")
    
    def check_dependencies(self):
        """Проверка зависимостей"""
        print("\n📦 ПРОВЕРКА ЗАВИСИМОСТЕЙ:")
        
        required_packages = {
            'pandas': 'pandas',
            'numpy': 'numpy', 
            'tensorflow': 'tensorflow',
            'ccxt': 'ccxt',
            'python-telegram-bot': 'telegram',
            'scikit-learn': 'sklearn',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn'
        }
        
        for package_name, import_name in required_packages.items():
            self.checks_total += 1
            try:
                __import__(import_name)
                print(f"✅ {package_name}")
                self.checks_passed += 1
            except ImportError:
                print(f"❌ {package_name} не установлен")
                self.issues.append(f"{package_name} не установлен")
    
    def check_data_access(self):
        """Проверка доступа к данным"""
        print("\n📊 ПРОВЕРКА ДОСТУПА К ДАННЫМ:")
        self.checks_total += 1
        
        try:
            from data.data_manager import get_candlestick_data
            
            # Пробуем получить тестовые данные
            df = get_candlestick_data("BTC/USDT", "1h", limit=10)
            
            if not df.empty:
                print(f"✅ Данные получены: {len(df)} записей")
                print(f"   Колонки: {list(df.columns)}")
                self.checks_passed += 1
            else:
                print("❌ Данные не получены")
                self.issues.append("Не удалось получить данные с Bybit")
                
        except Exception as e:
            print(f"❌ Ошибка получения данных: {e}")
            self.issues.append(f"Ошибка данных: {e}")
    
    def check_indicators(self):
        """Проверка индикаторов"""
        print("\n📈 ПРОВЕРКА ИНДИКАТОРОВ:")
        
        # Создаем тестовые данные
        test_data = pd.DataFrame({
            'close': [100 + i + np.random.randn() * 2 for i in range(100)],
            'high': [102 + i + np.random.randn() * 2 for i in range(100)],
            'low': [98 + i + np.random.randn() * 2 for i in range(100)],
            'volume': [1000 + np.random.randn() * 100 for _ in range(100)]
        })
        
        # Проверяем базовые индикаторы
        basic_indicators = [
            ('RSI', 'indicators.indicators', 'calculate_rsi'),
            ('MACD', 'indicators.indicators', 'calculate_macd'),
            ('ATR', 'indicators.indicators', 'calculate_atr'),
            ('SMA', 'indicators.indicators', 'calculate_sma'),
            ('EMA', 'indicators.indicators', 'calculate_ema'),
            ('Bollinger Bands', 'indicators.indicators', 'calculate_bollinger_bands')
        ]
        
        for name, module, func in basic_indicators:
            self.checks_total += 1
            try:
                module_obj = __import__(module, fromlist=[func])
                func_obj = getattr(module_obj, func)
                result = func_obj(test_data.copy())
                
                if not result.empty:
                    print(f"✅ {name}")
                    self.checks_passed += 1
                else:
                    print(f"❌ {name} - пустой результат")
                    self.issues.append(f"Индикатор {name} возвращает пустые данные")
                    
            except Exception as e:
                print(f"❌ {name} - ошибка: {e}")
                self.issues.append(f"Ошибка индикатора {name}: {e}")
        
        # Проверяем расширенные индикаторы
        self.checks_total += 1
        try:
            from indicators.enhanced_indicators import calculate_all_enhanced_indicators
            result = calculate_all_enhanced_indicators(test_data.copy())
            
            if not result.empty:
                print("✅ Расширенные индикаторы")
                added_cols = len(result.columns) - len(test_data.columns)
                print(f"   Добавлено колонок: {added_cols}")
                self.checks_passed += 1
            else:
                print("❌ Расширенные индикаторы - пустой результат")
                self.issues.append("Расширенные индикаторы не работают")
                
        except Exception as e:
            print(f"❌ Расширенные индикаторы - ошибка: {e}")
            self.issues.append(f"Ошибка расширенных индикаторов: {e}")
    
    def check_ml_models(self):
        """Проверка ML моделей"""
        print("\n🤖 ПРОВЕРКА ML МОДЕЛЕЙ:")
        
        # Проверяем базовую LSTM
        self.checks_total += 1
        try:
            from models.lstm_model import LSTMModel
            model = LSTMModel(sequence_length=10, num_features=5)
            
            # Тестовые данные
            X_test = np.random.randn(5, 10, 5)
            prediction = model.predict(X_test)
            
            if prediction is not None and len(prediction) == 5:
                print("✅ Базовая LSTM модель")
                self.checks_passed += 1
            else:
                print("❌ Базовая LSTM модель - неверный результат")
                self.issues.append("LSTM модель работает некорректно")
                
        except Exception as e:
            print(f"❌ Базовая LSTM модель - ошибка: {e}")
            self.issues.append(f"Ошибка LSTM модели: {e}")
        
        # Проверяем продвинутые модели
        self.checks_total += 1
        try:
            from models.advanced_ml_model import AdvancedMLModel
            advanced_model = AdvancedMLModel(sequence_length=10, prediction_horizon=1)
            
            print("✅ Продвинутые ML модели импортированы")
            self.checks_passed += 1
            
        except Exception as e:
            print(f"❌ Продвинутые ML модели - ошибка: {e}")
            self.issues.append(f"Ошибка продвинутых ML моделей: {e}")
    
    def check_trading_components(self):
        """Проверка торговых компонентов"""
        print("\n💼 ПРОВЕРКА ТОРГОВЫХ КОМПОНЕНТОВ:")
        
        # Проверяем risk manager
        self.checks_total += 1
        try:
            from trading.risk_manager import calculate_position_size, calculate_sl_tp_levels
            
            # Тестируем расчет позиции
            pos_size = calculate_position_size(
                balance=1000,
                entry_price=50000,
                stop_loss_pct=2.0,
                risk_pct=1.0
            )
            
            sl, tp = calculate_sl_tp_levels(
                entry_price=50000,
                stop_loss_pct=2.0,
                tp_ratio=2.0
            )
            
            if pos_size > 0 and sl > 0 and tp > 0:
                print(f"✅ Risk Manager (pos: {pos_size:.6f}, SL: {sl:.2f}, TP: {tp:.2f})")
                self.checks_passed += 1
            else:
                print("❌ Risk Manager - неверные расчеты")
                self.issues.append("Risk Manager работает некорректно")
                
        except Exception as e:
            print(f"❌ Risk Manager - ошибка: {e}")
            self.issues.append(f"Ошибка Risk Manager: {e}")
        
        # Проверяем executor
        self.checks_total += 1
        try:
            from trading.executor import init_trading_client
            print("✅ Trading Executor импортирован")
            self.checks_passed += 1
            
        except Exception as e:
            print(f"❌ Trading Executor - ошибка: {e}")
            self.issues.append(f"Ошибка Trading Executor: {e}")
    
    def check_backtesting(self):
        """Проверка бэктестинга"""
        print("\n📊 ПРОВЕРКА БЭКТЕСТИНГА:")
        
        # Проверяем базовый бэктестер
        self.checks_total += 1
        try:
            from backtesting.backtesting import Backtester
            
            # Создаем тестовые данные с сигналами
            test_data = pd.DataFrame({
                'close': [100 + i for i in range(20)],
                'signal': ['BUY', 'HOLD'] * 10
            })
            
            backtester = Backtester(initial_balance=1000, commission_rate=0.001)
            trades = backtester.run_backtest(test_data, signal_column='signal')
            
            print(f"✅ Базовый бэктестер (сделок: {len(trades)})")
            self.checks_passed += 1
            
        except Exception as e:
            print(f"❌ Базовый бэктестер - ошибка: {e}")
            self.issues.append(f"Ошибка базового бэктестера: {e}")
        
        # Проверяем расширенный бэктестер
        self.checks_total += 1
        try:
            from backtesting.enhanced_backtester import EnhancedBacktester
            print("✅ Расширенный бэктестер импортирован")
            self.checks_passed += 1
            
        except Exception as e:
            print(f"❌ Расширенный бэктестер - ошибка: {e}")
            self.issues.append(f"Ошибка расширенного бэктестера: {e}")
    
    def check_telegram_bot(self):
        """Проверка Telegram бота"""
        print("\n🤖 ПРОВЕРКА TELEGRAM БОТА:")
        
        self.checks_total += 1
        try:
            from telegram_bot.bot import main
            print("✅ Telegram бот импортирован")
            self.checks_passed += 1
            
        except Exception as e:
            print(f"❌ Telegram бот - ошибка: {e}")
            self.issues.append(f"Ошибка Telegram бота: {e}")
    
    def analyze_current_architecture(self):
        """Анализ текущей архитектуры"""
        print("\n🏗️ АНАЛИЗ АРХИТЕКТУРЫ:")
        
        print("📂 Структура проекта:")
        components = {
            'indicators/': ['Базовые и расширенные индикаторы'],
            'models/': ['LSTM модель', 'Продвинутые ML модели', 'Предобработка'],
            'trading/': ['Risk Manager', 'Executor'],
            'backtesting/': ['Базовый и расширенный бэктестеры'],
            'telegram_bot/': ['Telegram бот с автоторговлей'],
            'data/': ['Менеджер данных'],
            'scripts/': ['Различные скрипты для тестирования']
        }
        
        for folder, description in components.items():
            print(f"  {folder}: {', '.join(description)}")
    
    def generate_recommendations(self):
        """Генерирует рекомендации"""
        print("\n💡 РЕКОМЕНДАЦИИ ДЛЯ РАЗВИТИЯ ML:")
        
        recommendations = [
            "🤖 МАШИННОЕ ОБУЧЕНИЕ:",
            "  • Интегрировать AdvancedMLModel в Telegram бота",
            "  • Добавить автоматический поиск лучших комбинаций индикаторов", 
            "  • Создать систему автообучения модели на новых данных",
            "  • Реализовать ансамблевые методы (комбинирование моделей)",
            "",
            "📊 ДАННЫЕ:",
            "  • Загружать данные разных таймфреймов (1m, 5m, 15m, 1h, 4h, 1d)",
            "  • Добавить больше криптовалют для диверсификации",
            "  • Создать систему кэширования данных",
            "",
            "🔧 ОПТИМИЗАЦИЯ:",
            "  • Автоматическая настройка плеч",
            "  • Динамические стоп-лоссы на основе волатильности",
            "  • Адаптивные размеры позиций",
            "  • Мультитаймфреймовый анализ",
            "",
            "⚡ АВТОМАТИЗАЦИЯ:",
            "  • Непрерывный поиск лучших стратегий",
            "  • Автоматическое переключение между стратегиями",
            "  • Система алертов при изменении рыночных условий"
        ]
        
        for rec in recommendations:
            print(rec)
    
    def run_full_check(self):
        """Запускает полную проверку"""
        print("🔍 ПОЛНАЯ ПРОВЕРКА ТОРГОВОЙ СИСТЕМЫ")
        print("=" * 60)
        
        # Запускаем все проверки
        self.check_config()
        self.check_dependencies()
        self.check_data_access()
        self.check_indicators()
        self.check_ml_models()
        self.check_trading_components()
        self.check_backtesting()
        self.check_telegram_bot()
        
        # Анализируем архитектуру
        self.analyze_current_architecture()
        
        # Результаты
        print("\n" + "=" * 60)
        print("📋 РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
        print("=" * 60)
        
        success_rate = (self.checks_passed / self.checks_total * 100) if self.checks_total > 0 else 0
        print(f"✅ Успешно: {self.checks_passed}/{self.checks_total} ({success_rate:.1f}%)")
        
        if self.issues:
            print(f"\n❌ ПРОБЛЕМЫ ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        # Рекомендации
        self.generate_recommendations()
        
        # Итоговая оценка
        print("\n" + "=" * 60)
        if success_rate >= 80:
            print("🎉 СИСТЕМА ГОТОВА К РАЗВИТИЮ ML!")
            print("Можно переходить к созданию продвинутой ML системы для бота")
        elif success_rate >= 60:
            print("⚠️ СИСТЕМА ЧАСТИЧНО ГОТОВА")
            print("Исправьте основные проблемы перед развитием ML")
        else:
            print("❌ СИСТЕМА ТРЕБУЕТ СЕРЬЕЗНЫХ ИСПРАВЛЕНИЙ")
            print("Сначала устраните все критические ошибки")
        
        print("=" * 60)
        
        return success_rate >= 80

def main():
    """Главная функция"""
    checker = SystemChecker()
    is_ready = checker.run_full_check()
    
    return 0 if is_ready else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
