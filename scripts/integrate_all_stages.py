#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ИНТЕГРАЦИЯ ВСЕХ ЭТАПОВ В ЕДИНУЮ СИСТЕМУ
Объединяет Telegram бота с ML системой и живой торговлей
"""

import os
import sys
import asyncio
import threading
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импорты из всех этапов
from telegram_bot.bot import main as telegram_main
from models.advanced_ml_model import AdvancedMLModel
from signals.predictive_signals import PredictiveSignalGenerator
from trading.live_trader import LiveTrader
from monitoring.real_time_monitor import RealTimeMonitor, console_notification
from dashboard.web_dashboard import TradingDashboard

logger = logging.getLogger(__name__)

class IntegratedTradingSystem:
    """
    Интегрированная система, объединяющая все этапы
    """
    
    def __init__(self):
        # Компоненты системы
        self.ml_model = None
        self.signal_generator = None
        self.trader = None
        self.monitor = None
        self.dashboard = None
        
        # Потоки
        self.telegram_thread = None
        self.trading_thread = None
        self.monitoring_thread = None
        self.dashboard_thread = None
        
        self.is_running = False
    
    def initialize_ml_system(self):
        """Инициализирует ML систему из Этапа 2"""
        print("🤖 Инициализация ML системы...")
        
        try:
            # Создаем продвинутую ML модель
            self.ml_model = AdvancedMLModel(sequence_length=60, prediction_horizon=1)
            
            # Создаем генератор сигналов
            self.signal_generator = PredictiveSignalGenerator(
                ml_model=self.ml_model,
                confidence_threshold=0.6
            )
            
            print("✅ ML система инициализирована")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации ML системы: {e}")
            return False
    
    def initialize_trading_system(self):
        """Инициализирует торговую систему из Этапа 3"""
        print("💼 Инициализация торговой системы...")
        
        try:
            # Конфигурация для торговли
            exchange_config = {
                'api_key': os.getenv('BYBIT_API_KEY', ''),
                'secret': os.getenv('BYBIT_API_SECRET', ''),
                'sandbox': True  # Начинаем с sandbox
            }
            
            # Создаем трейдера
            self.trader = LiveTrader(
                exchange_config=exchange_config,
                initial_balance=1000
            )
            
            # Настройки (консервативные)
            self.trader.max_position_size = 0.02  # 2% от баланса
            self.trader.stop_loss_pct = 0.015     # 1.5% стоп-лосс
            self.trader.take_profit_pct = 0.03    # 3% тейк-профит
            self.trader.min_confidence = 0.7      # Высокий порог уверенности
            
            # Подключаем ML систему
            self.trader.set_ml_system(self.ml_model, self.signal_generator)
            
            print("✅ Торговая система инициализирована")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации торговой системы: {e}")
            return False
    
    def initialize_monitoring(self):
        """Инициализирует мониторинг"""
        print("🔍 Инициализация мониторинга...")
        
        try:
            self.monitor = RealTimeMonitor(
                trader=self.trader,
                notification_callbacks=[console_notification]
            )
            
            # Настройки мониторинга
            self.monitor.max_drawdown_alert = 0.03
            self.monitor.max_daily_loss = 0.05
            self.monitor.monitoring_interval = 30
            
            print("✅ Мониторинг инициализирован")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации мониторинга: {e}")
            return False
    
    def initialize_dashboard(self):
        """Инициализирует веб-дашборд"""
        print("🌐 Инициализация веб-дашборда...")
        
        try:
            self.dashboard = TradingDashboard(
                trader=self.trader,
                monitor=self.monitor,
                port=5000
            )
            
            print("✅ Веб-дашборд инициализирован")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации дашборда: {e}")
            return False
    
    def update_telegram_bot(self):
        """Обновляет Telegram бота для использования новых систем"""
        print("📱 Обновление Telegram бота...")
        
        # TODO: Здесь нужно модифицировать telegram_bot/bot.py
        # чтобы он использовал новые ML модели и торговую систему
        
        print("⚠️ Telegram бот использует старую систему")
        print("   Для полной интеграции нужно обновить telegram_bot/bot.py")
        return True
    
    def start_telegram_bot(self):
        """Запускает Telegram бота в отдельном потоке"""
        def telegram_worker():
            try:
                logger.info("🤖 Запуск Telegram бота")
                telegram_main()
            except Exception as e:
                logger.error(f"❌ Ошибка в Telegram боте: {e}")
        
        self.telegram_thread = threading.Thread(target=telegram_worker, daemon=True)
        self.telegram_thread.start()
    
    def start_trading_system(self):
        """Запускает торговую систему в отдельном потоке"""
        def trading_worker():
            try:
                logger.info("💼 Запуск торговой системы")
                if self.trader:
                    self.trader.start_trading()
            except Exception as e:
                logger.error(f"❌ Ошибка в торговой системе: {e}")
        
        self.trading_thread = threading.Thread(target=trading_worker, daemon=True)
        self.trading_thread.start()
    
    def start_monitoring(self):
        """Запускает мониторинг в отдельном потоке"""
        def monitoring_worker():
            try:
                logger.info("🔍 Запуск мониторинга")
                if self.monitor:
                    self.monitor.start_monitoring()
            except Exception as e:
                logger.error(f"❌ Ошибка в мониторинге: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self.monitoring_thread.start()
    
    def start_dashboard(self):
        """Запускает веб-дашборд в отдельном потоке"""
        def dashboard_worker():
            try:
                logger.info("🌐 Запуск веб-дашборда")
                if self.dashboard:
                    self.dashboard.run(debug=False, host='0.0.0.0')
            except Exception as e:
                logger.error(f"❌ Ошибка в дашборде: {e}")
        
        self.dashboard_thread = threading.Thread(target=dashboard_worker, daemon=True)
        self.dashboard_thread.start()
    
    def run_integrated_system(self):
        """Запускает полную интегрированную систему"""
        print("🚀 ЗАПУСК ИНТЕГРИРОВАННОЙ ТОРГОВОЙ СИСТЕМЫ")
        print("=" * 60)
        
        try:
            # 1. Инициализация всех компонентов
            if not self.initialize_ml_system():
                return False
            
            if not self.initialize_trading_system():
                return False
            
            if not self.initialize_monitoring():
                return False
            
            if not self.initialize_dashboard():
                return False
            
            if not self.update_telegram_bot():
                return False
            
            # 2. Запуск всех сервисов
            self.is_running = True
            
            print("\n🔥 Запуск всех сервисов:")
            
            # Telegram бот (основной)
            self.start_telegram_bot()
            print("✅ Telegram бот запущен")
            
            # Торговая система (пока в демо-режиме)
            # self.start_trading_system()
            # print("✅ Торговая система запущена")
            
            # Мониторинг
            self.start_monitoring()
            print("✅ Мониторинг запущен")
            
            # Веб-дашборд
            self.start_dashboard()
            print("✅ Веб-дашборд запущен")
            
            # 3. Информация о системе
            print("\n" + "=" * 60)
            print("🎉 ИНТЕГРИРОВАННАЯ СИСТЕМА ЗАПУЩЕНА!")
            print("=" * 60)
            print("📱 Telegram бот: Работает")
            print("🤖 ML система: Подключена")
            print("💼 Торговая система: Готова (демо-режим)")
            print("🔍 Мониторинг: Активен")
            print("🌐 Веб-дашборд: http://localhost:5000")
            print("\n⚡ Все этапы объединены в единую систему!")
            print("📞 Используйте Telegram бота для управления")
            print("\n⚠️ ДЛЯ ОСТАНОВКИ НАЖМИТЕ Ctrl+C")
            print("=" * 60)
            
            # 4. Основной цикл
            try:
                while self.is_running:
                    import time
                    time.sleep(10)
                    
                    # Проверяем состояние потоков
                    if self.telegram_thread and not self.telegram_thread.is_alive():
                        logger.warning("⚠️ Telegram бот остановлен")
                    
                    if self.monitoring_thread and not self.monitoring_thread.is_alive():
                        logger.warning("⚠️ Мониторинг остановлен")
                        
            except KeyboardInterrupt:
                print("\n🛑 Получен сигнал остановки...")
                self.stop_system()
            
            return True
            
        except Exception as e:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
            self.stop_system()
            return False
    
    def stop_system(self):
        """Останавливает всю систему"""
        print("\n🛑 ОСТАНОВКА ИНТЕГРИРОВАННОЙ СИСТЕМЫ...")
        
        self.is_running = False
        
        # Останавливаем все компоненты
        if self.trader:
            self.trader.stop_trading()
            print("✅ Торговая система остановлена")
        
        if self.monitor:
            self.monitor.stop_monitoring()
            print("✅ Мониторинг остановлен")
        
        print("🏁 СИСТЕМА ПОЛНОСТЬЮ ОСТАНОВЛЕНА")

def main():
    """Главная функция"""
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🤖 ИНТЕГРИРОВАННАЯ ТОРГОВАЯ СИСТЕМА")
    print("Объединяет все этапы в единое решение")
    print("=" * 60)
    
    # Создаем и запускаем систему
    integrated_system = IntegratedTradingSystem()
    success = integrated_system.run_integrated_system()
    
    if success:
        print("\n✅ СИСТЕМА ЗАВЕРШИЛА РАБОТУ УСПЕШНО!")
        return 0
    else:
        print("\n❌ СИСТЕМА ЗАВЕРШИЛАСЬ С ОШИБКАМИ!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
