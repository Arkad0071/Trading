#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ГЛАВНЫЙ СКРИПТ ЭТАПА 3: РЕАЛЬНАЯ ТОРГОВАЯ СИСТЕМА
Запускает полную систему реальной торговли:
1. Загружает лучшие ML модели из Этапа 2
2. Подключается к реальной бирже Bybit
3. Торгует в реальном времени с ML прогнозами
4. Мониторинг и веб-дашборд
5. Уведомления и управление рисками
"""

import os
import sys
import asyncio
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import time

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.advanced_ml_model import AdvancedMLModel
from signals.predictive_signals import PredictiveSignalGenerator
from trading.live_trader import LiveTrader
from monitoring.real_time_monitor import RealTimeMonitor, console_notification
from dashboard.web_dashboard import TradingDashboard
from utils.config import get_bybit_credentials

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveTradingSystem:
    """
    Полная система реальной торговли
    """
    
    def __init__(self, use_sandbox=True, initial_balance=1000):
        """
        Инициализация системы реальной торговли
        
        Args:
            use_sandbox: Использовать тестовую среду (рекомендуется для начала)
            initial_balance: Начальный баланс для торговли
        """
        self.use_sandbox = use_sandbox
        self.initial_balance = initial_balance
        
        # Компоненты системы
        self.ml_model = None
        self.signal_generator = None
        self.trader = None
        self.monitor = None
        self.dashboard = None
        
        # Потоки для асинхронной работы
        self.trading_thread = None
        self.monitoring_thread = None
        self.dashboard_thread = None
        
        # Состояние системы
        self.is_running = False
        self.system_start_time = None
        
    def load_ml_model(self, model_path: str = None) -> bool:
        """
        Загружает лучшую ML модель из Этапа 2
        """
        print("🤖 Загрузка ML модели...")
        
        try:
            # Создаем новую модель (в реальной системе здесь была бы загрузка из файла)
            self.ml_model = AdvancedMLModel(sequence_length=60, prediction_horizon=1)
            
            # TODO: Загрузить сохраненную модель
            # if model_path and os.path.exists(model_path):
            #     self.ml_model.load_model(model_path)
            #     print(f"✅ ML модель загружена: {model_path}")
            # else:
            
            print("⚠️ Используется новая модель (в продакшене загружайте обученную)")
            print("✅ ML модель инициализирована")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки ML модели: {e}")
            return False
    
    def initialize_signal_generator(self) -> bool:
        """
        Инициализирует генератор торговых сигналов
        """
        print("🎯 Инициализация генератора сигналов...")
        
        try:
            self.signal_generator = PredictiveSignalGenerator(
                ml_model=self.ml_model,
                confidence_threshold=0.6  # Повышенный порог для реальной торговли
            )
            
            print("✅ Генератор сигналов готов")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации генератора сигналов: {e}")
            return False
    
    def initialize_trader(self) -> bool:
        """
        Инициализирует торговый модуль
        """
        print("💼 Инициализация торгового модуля...")
        
        try:
            # Получаем конфигурацию биржи
            exchange_config = {
                'api_key': os.getenv('BYBIT_API_KEY', ''),
                'secret': os.getenv('BYBIT_SECRET', ''),
                'sandbox': self.use_sandbox
            }
            
            # Проверяем наличие API ключей
            if not exchange_config['api_key'] or not exchange_config['secret']:
                print("⚠️ API ключи не найдены. Работа в демо-режиме.")
                print("Для реальной торговли установите переменные окружения:")
                print("  BYBIT_API_KEY=your_api_key")
                print("  BYBIT_SECRET=your_secret")
                
                # В демо-режиме используем фиктивную конфигурацию
                exchange_config = {
                    'api_key': 'demo_key',
                    'secret': 'demo_secret',
                    'sandbox': True
                }
            
            # Создаем трейдер
            self.trader = LiveTrader(
                exchange_config=exchange_config,
                initial_balance=self.initial_balance
            )
            
            # Устанавливаем параметры торговли (консервативные для начала)
            self.trader.max_position_size = 0.05    # Максимум 5% от баланса на сделку
            self.trader.stop_loss_pct = 0.015       # 1.5% стоп-лосс
            self.trader.take_profit_pct = 0.03      # 3% тейк-профит
            self.trader.min_confidence = 0.7        # Высокий порог уверенности
            
            # Ограничиваем торговлю одним символом для начала
            self.trader.trading_symbols = ['BTC/USDT']
            
            # Подключаем ML систему к трейдеру
            self.trader.set_ml_system(self.ml_model, self.signal_generator)
            
            # Инициализируем подключение к бирже (если не демо-режим)
            if exchange_config['api_key'] != 'demo_key':
                success = self.trader.initialize_exchange()
                if not success:
                    print("⚠️ Не удалось подключиться к бирже. Работа в демо-режиме.")
            else:
                print("📊 Работа в демо-режиме (без реальных сделок)")
            
            print("✅ Торговый модуль готов")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации торгового модуля: {e}")
            return False
    
    def initialize_monitoring(self) -> bool:
        """
        Инициализирует систему мониторинга
        """
        print("🔍 Инициализация системы мониторинга...")
        
        try:
            # Создаем монитор
            self.monitor = RealTimeMonitor(
                trader=self.trader,
                notification_callbacks=[console_notification]
            )
            
            # Настраиваем параметры мониторинга (консервативные)
            self.monitor.max_drawdown_alert = 0.03   # Уведомление при просадке > 3%
            self.monitor.max_daily_loss = 0.05       # Максимальная дневная потеря 5%
            self.monitor.max_open_positions = 2      # Максимум 2 позиции одновременно
            self.monitor.monitoring_interval = 30    # Проверка каждые 30 секунд
            
            print("✅ Система мониторинга готова")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации мониторинга: {e}")
            return False
    
    def initialize_dashboard(self) -> bool:
        """
        Инициализирует веб-дашборд
        """
        print("🌐 Инициализация веб-дашборда...")
        
        try:
            self.dashboard = TradingDashboard(
                trader=self.trader,
                monitor=self.monitor,
                port=5000
            )
            
            print("✅ Веб-дашборд готов (http://localhost:5000)")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации дашборда: {e}")
            return False
    
    def start_trading_thread(self):
        """
        Запускает торговлю в отдельном потоке
        """
        def trading_worker():
            try:
                logger.info("🚀 Запуск торгового потока")
                
                # Эмуляция торгового цикла для демо-режима
                if not self.trader.exchange:
                    logger.info("📊 Демо-режим: эмуляция торговли")
                    
                    while self.is_running:
                        try:
                            # Эмулируем анализ рынка
                            logger.info("📊 Анализ рынка (демо-режим)")
                            
                            # Эмулируем генерацию сигнала
                            if self.signal_generator:
                                # Создаем фиктивные данные для демонстрации
                                demo_data = pd.DataFrame({
                                    'close': [50000 + np.random.randn() * 1000 for _ in range(100)],
                                    'volume': [1000 + np.random.randn() * 100 for _ in range(100)]
                                })
                                
                                # Добавляем базовые индикаторы
                                demo_data['RSI'] = 50 + np.random.randn(100) * 10
                                demo_data['MACD'] = np.random.randn(100) * 5
                                demo_data['MACD_signal'] = np.random.randn(100) * 3
                                
                                # Генерируем сигнал
                                try:
                                    df_with_signals = self.signal_generator.generate_technical_signals(demo_data)
                                    
                                    if len(df_with_signals) > 0:
                                        last_signal = df_with_signals['tech_signal'].iloc[-1]
                                        last_strength = df_with_signals['tech_strength'].iloc[-1]
                                        
                                        logger.info(f"Сигнал: {last_signal} (сила: {last_strength:.3f})")
                                except Exception as signal_error:
                                    logger.warning(f"Ошибка генерации сигнала: {signal_error}")
                            
                            # Пауза между циклами
                            time.sleep(60)  # Проверка каждую минуту в демо-режиме
                            
                        except Exception as cycle_error:
                            logger.error(f"Ошибка в торговом цикле: {cycle_error}")
                            time.sleep(30)
                else:
                    # Реальная торговля
                    self.trader.start_trading()
                    
            except Exception as e:
                logger.error(f"❌ Критическая ошибка в торговом потоке: {e}")
        
        self.trading_thread = threading.Thread(target=trading_worker, daemon=True)
        self.trading_thread.start()
    
    def start_monitoring_thread(self):
        """
        Запускает мониторинг в отдельном потоке
        """
        def monitoring_worker():
            try:
                logger.info("🔍 Запуск мониторинга")
                self.monitor.start_monitoring()
            except Exception as e:
                logger.error(f"❌ Ошибка в потоке мониторинга: {e}")
        
        if self.monitor:
            self.monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
            self.monitoring_thread.start()
    
    def start_dashboard_thread(self):
        """
        Запускает веб-дашборд в отдельном потоке
        """
        def dashboard_worker():
            try:
                logger.info("🌐 Запуск веб-дашборда")
                if self.dashboard:
                    self.dashboard.run(debug=False, host='0.0.0.0')
            except Exception as e:
                logger.error(f"❌ Ошибка в потоке дашборда: {e}")
        
        if self.dashboard:
            self.dashboard_thread = threading.Thread(target=dashboard_worker, daemon=True)
            self.dashboard_thread.start()
    
    def print_system_info(self):
        """
        Выводит информацию о системе
        """
        print("\n" + "=" * 80)
        print("🤖 СИСТЕМА РЕАЛЬНОЙ ТОРГОВЛИ ЗАПУЩЕНА")
        print("=" * 80)
        print(f"⏰ Время запуска: {self.system_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Начальный баланс: ${self.initial_balance}")
        print(f"🔒 Режим: {'Sandbox (Тестовый)' if self.use_sandbox else 'LIVE (Реальный)'}")
        print(f"📊 Торговые символы: {self.trader.trading_symbols if self.trader else 'Не задано'}")
        print(f"🎯 Мин. уверенность: {self.trader.min_confidence if self.trader else 'Не задано'}")
        print(f"💼 Макс. размер позиции: {self.trader.max_position_size * 100 if self.trader else 'Не задано'}%")
        print(f"🛡️ Stop Loss: {self.trader.stop_loss_pct * 100 if self.trader else 'Не задано'}%")
        print(f"🎯 Take Profit: {self.trader.take_profit_pct * 100 if self.trader else 'Не задано'}%")
        print("\n🌐 ВЕБ-ДАШБОРД: http://localhost:5000")
        print("📊 Мониторинг: Активен")
        print("🔔 Уведомления: Консоль")
        print("\n⚠️ ДЛЯ ОСТАНОВКИ НАЖМИТЕ Ctrl+C")
        print("=" * 80 + "\n")
    
    def run_system(self) -> bool:
        """
        Запускает полную систему реальной торговли
        """
        self.system_start_time = datetime.now()
        
        print("🚀 ЗАПУСК СИСТЕМЫ РЕАЛЬНОЙ ТОРГОВЛИ")
        print("=" * 50)
        
        try:
            # 1. Загружаем ML модель
            if not self.load_ml_model():
                return False
            
            # 2. Инициализируем генератор сигналов
            if not self.initialize_signal_generator():
                return False
            
            # 3. Инициализируем трейдер
            if not self.initialize_trader():
                return False
            
            # 4. Инициализируем мониторинг
            if not self.initialize_monitoring():
                return False
            
            # 5. Инициализируем веб-дашборд
            if not self.initialize_dashboard():
                return False
            
            # 6. Создаем директорию для логов
            os.makedirs('logs', exist_ok=True)
            
            # 7. Запускаем все компоненты
            self.is_running = True
            
            # Запускаем торговлю
            self.start_trading_thread()
            
            # Запускаем мониторинг
            self.start_monitoring_thread()
            
            # Запускаем веб-дашборд
            self.start_dashboard_thread()
            
            # Выводим информацию о системе
            self.print_system_info()
            
            # Основной цикл
            try:
                while self.is_running:
                    time.sleep(10)
                    
                    # Проверяем состояние потоков
                    if self.trading_thread and not self.trading_thread.is_alive():
                        logger.warning("⚠️ Торговый поток остановлен")
                        
                    if self.monitoring_thread and not self.monitoring_thread.is_alive():
                        logger.warning("⚠️ Поток мониторинга остановлен")
                    
                    # Выводим краткую статистику
                    if self.trader and hasattr(self.trader, 'stats'):
                        stats = self.trader.stats
                        total_trades = stats.get('total_trades', 0)
                        total_profit = stats.get('total_profit', 0)
                        
                        if total_trades > 0:
                            logger.info(f"📈 Статистика: {total_trades} сделок, ${total_profit:.2f} прибыль")
                
            except KeyboardInterrupt:
                print("\n🛑 Получен сигнал остановки...")
                self.stop_system()
            
            return True
            
        except Exception as e:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
            self.stop_system()
            return False
    
    def stop_system(self):
        """
        Останавливает всю систему
        """
        print("\n🛑 ОСТАНОВКА СИСТЕМЫ...")
        
        self.is_running = False
        
        # Останавливаем торговлю
        if self.trader:
            self.trader.stop_trading()
            print("✅ Торговля остановлена")
        
        # Останавливаем мониторинг
        if self.monitor:
            self.monitor.stop_monitoring()
            print("✅ Мониторинг остановлен")
        
        # Сохраняем данные
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.trader:
                self.trader.save_trading_log(f"logs/trading_log_{timestamp}.json")
            
            if self.monitor:
                self.monitor.save_monitoring_data(f"logs/monitoring_data_{timestamp}.json")
            
            print("✅ Данные сохранены")
            
        except Exception as e:
            print(f"⚠️ Ошибка сохранения данных: {e}")
        
        # Выводим финальную статистику
        if self.trader and hasattr(self.trader, 'stats'):
            stats = self.trader.stats
            uptime = datetime.now() - self.system_start_time
            
            print(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
            print(f"⏰ Время работы: {uptime}")
            print(f"📈 Всего сделок: {stats.get('total_trades', 0)}")
            print(f"💰 Общая прибыль: ${stats.get('total_profit', 0):.2f}")
            print(f"🏆 Прибыльных сделок: {stats.get('winning_trades', 0)}")
        
        print("\n🏁 СИСТЕМА ПОЛНОСТЬЮ ОСТАНОВЛЕНА")

def main():
    """Главная функция"""
    
    print("🤖 ДОБРО ПОЖАЛОВАТЬ В СИСТЕМУ РЕАЛЬНОЙ ТОРГОВЛИ!")
    print("=" * 60)
    
    # Запрашиваем параметры у пользователя
    print("\n⚙️ НАСТРОЙКА СИСТЕМЫ:")
    
    use_sandbox = input("Использовать тестовую среду? (y/n) [y]: ").lower()
    use_sandbox = use_sandbox != 'n'
    
    initial_balance = input("Начальный баланс для торговли [$1000]: ")
    try:
        initial_balance = float(initial_balance) if initial_balance else 1000.0
    except ValueError:
        initial_balance = 1000.0
    
    print(f"\n✅ Настройки:")
    print(f"  Тестовая среда: {'Да' if use_sandbox else 'НЕТ (РЕАЛЬНАЯ ТОРГОВЛЯ!)'}")
    print(f"  Начальный баланс: ${initial_balance}")
    
    if not use_sandbox:
        print("\n🚨 ВНИМАНИЕ! ВЫ ВЫБРАЛИ РЕАЛЬНУЮ ТОРГОВЛЮ!")
        print("🚨 УБЕДИТЕСЬ, ЧТО У ВАС ЕСТЬ API КЛЮЧИ И ВЫ ГОТОВЫ К РИСКАМ!")
        confirm = input("Продолжить? (yes/no): ").lower()
        if confirm != 'yes':
            print("❌ Отменено пользователем")
            return 1
    
    print("\n🚀 ЗАПУСК СИСТЕМЫ...")
    
    # Создаем и запускаем систему
    trading_system = LiveTradingSystem(
        use_sandbox=use_sandbox,
        initial_balance=initial_balance
    )
    
    success = trading_system.run_system()
    
    if success:
        print("\n🎉 СИСТЕМА ЗАВЕРШИЛА РАБОТУ УСПЕШНО!")
        return 0
    else:
        print("\n❌ СИСТЕМА ЗАВЕРШИЛАСЬ С ОШИБКАМИ!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
