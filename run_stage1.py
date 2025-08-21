#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой скрипт для запуска первого этапа
"""

import subprocess
import sys
import os

def main():
    """Запускает первый этап"""
    script_path = "scripts/run_stage1_complete.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Файл {script_path} не найден!")
        return False
    
    print("🚀 Запускаю первый этап...")
    print(f"📁 Скрипт: {script_path}")
    
    try:
        # Запускаем скрипт
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              encoding='utf-8')
        
        # Выводим результат
        if result.stdout:
            print("📤 ВЫВОД:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ ОШИБКИ:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Первый этап завершен успешно!")
            return True
        else:
            print(f"❌ Первый этап завершился с ошибкой (код: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
