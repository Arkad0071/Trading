#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ТЕСТ ML МОЗГА
Простая проверка работы продвинутого ML мозга
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram_bot.enhanced_ml_brain import ml_brain, get_enhanced_prediction, run_brain_training

def test_basic_functionality():
    """Тестирует базовую функциональность"""
    print("🧠 ТЕСТ ML МОЗГА")
    print("=" * 40)
    
    try:
        # 1. Тест получения прогноза
        print("1️⃣ Тест получения прогноза...")
        prediction = get_enhanced_prediction()
        
        print(f"✅ Прогноз получен:")
        print(f"   Сигнал: {prediction['signal']}")
        print(f"   Уверенность: {prediction['confidence']:.1%}")
        print(f"   Плечо: {prediction['recommended_leverage']}x")
        print(f"   Обоснование: {prediction['reasoning']}")
        
        # 2. Тест загрузки данных
        print("\n2️⃣ Тест загрузки данных...")
        data = ml_brain.load_all_bitcoin_data()
        print(f"✅ Загружено таймфреймов: {len(data)}")
        for tf, df in data.items():
            print(f"   {tf}: {len(df)} записей")
        
        # 3. Тест индикаторов (если есть данные)
        if data:
            print("\n3️⃣ Тест расчета индикаторов...")
            first_data = list(data.values())[0]
            df_with_indicators = ml_brain.calculate_all_indicators(first_data.copy())
            
            original_cols = len(first_data.columns)
            new_cols = len(df_with_indicators.columns)
            added_cols = new_cols - original_cols
            
            print(f"✅ Индикаторы рассчитаны:")
            print(f"   Было колонок: {original_cols}")
            print(f"   Стало колонок: {new_cols}")
            print(f"   Добавлено: {added_cols}")
        
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА ТЕСТА: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training():
    """Тестирует обучение ML мозга"""
    print("\n🏋️ ТЕСТ ОБУЧЕНИЯ ML МОЗГА")
    print("=" * 40)
    
    try:
        print("⚠️ Это может занять несколько минут...")
        results = run_brain_training()
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
        print(f"   Статус: {results.get('status', 'unknown')}")
        print(f"   Данные загружены: {results.get('data_loaded', False)}")
        print(f"   Индикаторы рассчитаны: {results.get('indicators_calculated', False)}")
        print(f"   Найдено стратегий: {results.get('strategies_found', 0)}")
        print(f"   Обучено моделей: {results.get('models_trained', 0)}")
        
        if results.get('best_parameters'):
            params = results['best_parameters']
            print(f"\n🎯 ЛУЧШИЕ ПАРАМЕТРЫ:")
            print(f"   Плечо: {params.get('leverage', 1)}x")
            print(f"   Stop Loss: {params.get('stop_loss_pct', 2.0)}%")
            print(f"   Take Profit: {params.get('take_profit_ratio', 2.0)}x")
        
        if results.get('status') == 'completed':
            print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            return True
        else:
            print(f"\n⚠️ Обучение завершилось со статусом: {results.get('status')}")
            return False
            
    except Exception as e:
        print(f"\n❌ ОШИБКА ОБУЧЕНИЯ: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Главная функция теста"""
    print("🚀 ЗАПУСК ТЕСТОВ ML МОЗГА")
    print("=" * 50)
    
    # Базовые тесты
    basic_ok = test_basic_functionality()
    
    if basic_ok:
        print("\n" + "=" * 50)
        choice = input("Запустить полное обучение? (y/N): ").lower()
        
        if choice == 'y':
            training_ok = test_training()
            
            if training_ok:
                print("\n🎉 ВСЕ ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ!")
                print("\nТеперь можно:")
                print("1. Запустить Telegram бота")
                print("2. Использовать команду /brain_status")
                print("3. Использовать команду /enhanced_predict")
                return 0
            else:
                print("\n⚠️ Обучение завершилось с ошибками")
                return 1
        else:
            print("\n✅ Базовые тесты пройдены. Обучение пропущено.")
            return 0
    else:
        print("\n❌ Базовые тесты провалены")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
