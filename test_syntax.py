#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка синтаксиса всех модулей
"""

import sys
import os

def test_import(module_name, import_path):
    """Тестирует импорт модуля"""
    try:
        __import__(import_path)
        print(f"✅ {module_name}: OK")
        return True
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        return False

def main():
    """Главная функция"""
    print("🔍 ПРОВЕРКА СИНТАКСИСА ВСЕХ МОДУЛЕЙ")
    print("=" * 50)
    
    # Список модулей для проверки
    modules_to_test = [
        ("Базовые индикаторы", "indicators.indicators"),
        ("Расширенные индикаторы", "indicators.enhanced_indicators"),
        ("Базовые стратегии", "backtesting.strategies"),
        ("Расширенные стратегии", "strategies.enhanced_strategies"),
        ("Улучшенный бэктестер", "backtesting.enhanced_backtester"),
        ("Визуализация", "visualization.strategy_visualizer"),
        ("Мультитаймфреймовый анализатор", "strategies.multi_timeframe_analyzer"),
        ("Оптимизатор параметров", "strategies.parameter_optimizer"),
        ("Менеджер данных", "data.data_manager")
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module_name, import_path in modules_to_test:
        if test_import(module_name, import_path):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТ: {success_count}/{total_count} модулей прошли проверку")
    
    if success_count == total_count:
        print("🎉 ВСЕ МОДУЛИ РАБОТАЮТ! Можете запускать первый этап!")
        return True
    else:
        print("❌ ЕСТЬ ПРОБЛЕМЫ! Нужно исправить ошибки.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
