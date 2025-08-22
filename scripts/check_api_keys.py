#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки API ключей и подключения к Bybit
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import BYBIT_API_KEY, BYBIT_API_SECRET, TOKEN
from utils.bybit_client import get_private_client, get_public_client

def check_telegram_token():
    """Проверяет токен Telegram бота"""
    print("🤖 ПРОВЕРКА TELEGRAM БОТА:")
    
    if not TOKEN:
        print("❌ TELEGRAM_TOKEN не найден в переменных окружения")
        print("   Создайте файл .env и добавьте: TELEGRAM_TOKEN=your_token")
        return False
    
    print(f"✅ Токен найден: {TOKEN[:10]}...{TOKEN[-10:]}")
    
    # Можно добавить проверку валидности токена через API
    return True

def check_bybit_keys():
    """Проверяет API ключи Bybit"""
    print("\n💼 ПРОВЕРКА BYBIT API:")
    
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        print("❌ API ключи Bybit не найдены")
        print("   Создайте файл .env и добавьте:")
        print("   BYBIT_API_KEY=your_key")
        print("   BYBIT_API_SECRET=your_secret")
        return False
    
    print(f"✅ API Key: {BYBIT_API_KEY[:8]}...{BYBIT_API_KEY[-8:]}")
    print(f"✅ Secret: {BYBIT_API_SECRET[:8]}...{BYBIT_API_SECRET[-8:]}")
    
    return True

def test_bybit_connection():
    """Тестирует подключение к Bybit"""
    print("\n🌐 ТЕСТИРОВАНИЕ ПОДКЛЮЧЕНИЯ К BYBIT:")
    
    try:
        # Тест публичного API
        print("📊 Тест публичного API...")
        public_client = get_public_client()
        ticker = public_client.fetch_ticker('BTC/USDT')
        print(f"✅ Публичный API работает. BTC/USDT: ${ticker['last']:.2f}")
        
        # Тест приватного API (если есть ключи)
        if BYBIT_API_KEY and BYBIT_API_SECRET:
            print("🔐 Тест приватного API...")
            private_client = get_private_client()
            
            # Проверяем баланс
            balance = private_client.fetch_balance()
            print("✅ Приватный API работает")
            
            # Показываем основные балансы
            main_currencies = ['USDT', 'BTC', 'ETH']
            for currency in main_currencies:
                if currency in balance:
                    free = balance[currency].get('free', 0)
                    if free > 0:
                        print(f"   {currency}: {free}")
            
            return True
        else:
            print("⚠️ API ключи не найдены, приватный API не тестируется")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def check_dependencies():
    """Проверяет установленные зависимости"""
    print("\n📦 ПРОВЕРКА ЗАВИСИМОСТЕЙ:")
    
    required_packages = [
        'ccxt', 'python-telegram-bot', 'pandas', 'numpy', 
        'tensorflow', 'scikit-learn', 'matplotlib', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - НЕ УСТАНОВЛЕН")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Установите недостающие пакеты:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Основная функция проверки"""
    print("🔍 ПРОВЕРКА СИСТЕМЫ ТОРГОВОГО БОТА")
    print("=" * 50)
    
    # Проверяем все компоненты
    telegram_ok = check_telegram_token()
    bybit_keys_ok = check_bybit_keys()
    dependencies_ok = check_dependencies()
    bybit_connection_ok = test_bybit_connection()
    
    print("\n" + "=" * 50)
    print("📋 ИТОГИ ПРОВЕРКИ:")
    print("=" * 50)
    
    print(f"🤖 Telegram токен: {'✅ ОК' if telegram_ok else '❌ ПРОБЛЕМА'}")
    print(f"🔑 Bybit API ключи: {'✅ ОК' if bybit_keys_ok else '❌ ПРОБЛЕМА'}")
    print(f"📦 Зависимости: {'✅ ОК' if dependencies_ok else '❌ ПРОБЛЕМА'}")
    print(f"🌐 Подключение Bybit: {'✅ ОК' if bybit_connection_ok else '❌ ПРОБЛЕМА'}")
    
    if all([telegram_ok, bybit_keys_ok, dependencies_ok, bybit_connection_ok]):
        print("\n🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ! СИСТЕМА ГОТОВА К РАБОТЕ!")
        return True
    else:
        print("\n⚠️ ЕСТЬ ПРОБЛЕМЫ. ИСПРАВЬТЕ ИХ ПЕРЕД ЗАПУСКОМ ТОРГОВЛИ.")
        
        # Инструкции по исправлению
        if not telegram_ok or not bybit_keys_ok:
            print("\n🔧 КАК ИСПРАВИТЬ:")
            print("1. Создайте файл .env в корне проекта")
            print("2. Добавьте в него:")
            print("   TELEGRAM_TOKEN=ваш_токен_бота")
            print("   BYBIT_API_KEY=ваш_api_ключ")
            print("   BYBIT_API_SECRET=ваш_секретный_ключ")
            print("\n📚 Где получить:")
            print("   • Telegram токен: @BotFather")
            print("   • Bybit API: https://www.bybit.com/app/user/api-management")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
