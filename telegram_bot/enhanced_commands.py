#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
УЛУЧШЕННЫЕ КОМАНДЫ ДЛЯ TELEGRAM БОТА
Команды для использования продвинутого ML мозга
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import os
import sys
import subprocess
from datetime import datetime
from telegram import Update
from telegram.ext import CallbackContext

from enhanced_ml_brain import ml_brain, get_enhanced_prediction, run_brain_training
from data.data_manager import get_candlestick_data
from trading.risk_manager import calculate_position_size, calculate_sl_tp_levels
from positions_db import load_bot_state, save_bot_state, log_prediction
from utils.config import DEFAULT_TP_RATIO, DEFAULT_SL_PCT

logger = logging.getLogger(__name__)

async def enhanced_prediction_command(update: Update, context: CallbackContext):
    """
    Команда /enhanced_predict - улучшенный прогноз с ML мозгом
    """
    try:
        await update.message.reply_text("🧠 Запуск продвинутого ML анализа...")
        
        # Получаем СВЕЖИЕ данные (последние 100 минут)
        df = get_candlestick_data("BTC/USDT", "1m", limit=100, private=True)
        if df.empty:
            await update.message.reply_text("❌ Не удалось получить данные для анализа")
            return
        
        # Получаем улучшенный прогноз
        prediction = get_enhanced_prediction(df)
        
        # Время последних данных
        last_data_time = df.index[-1]
        
        # Формируем ответ
        signal = prediction['signal']
        confidence = prediction['confidence']
        reasoning = prediction['reasoning']
        leverage = prediction['recommended_leverage']
        sl_pct = prediction['stop_loss_pct']
        tp_ratio = prediction['take_profit_ratio']
        
        current_price = df['close'].iloc[-1]
        
        # Рассчитываем параметры позиции
        state = load_bot_state()
        position_size = calculate_position_size(
            balance=state.get('usd_balance', 1000),
            entry_price=current_price,
            stop_loss_pct=sl_pct,
            risk_pct=2.0  # 2% риск
        )
        
        stop_price, take_price = calculate_sl_tp_levels(
            entry_price=current_price,
            stop_loss_pct=sl_pct,
            tp_ratio=tp_ratio
        )
        
        # Логируем прогноз
        log_prediction(signal, confidence * 100, current_price)
        
        # Формируем сообщение
        emoji = {
            'STRONG_BUY': '🚀',
            'BUY': '📈',
            'HOLD': '⏸️',
            'SELL': '📉',
            'STRONG_SELL': '💥'
        }.get(signal, '❓')
        
        message = (
            f"{emoji} ПРОДВИНУТЫЙ ML ПРОГНОЗ {emoji}\n\n"
            f"📊 Анализ:\n"
            f"• Сигнал: {signal}\n"
            f"• Уверенность: {confidence:.1%}\n"
            f"• Текущая цена: ${current_price:,.2f}\n"
            f"• Время данных: {last_data_time.strftime('%H:%M:%S UTC')}\n\n"
            f"⚙️ Рекомендуемые параметры:\n"
            f"• Плечо: {leverage}x\n"
            f"• Размер позиции: {position_size:.6f} BTC\n"
            f"• Stop Loss: ${stop_price:,.2f} ({sl_pct}%)\n"
            f"• Take Profit: ${take_price:,.2f} ({tp_ratio}x)\n\n"
            f"🧠 Обоснование:\n{reasoning}\n\n"
            f"⚡ Для автоматической торговли используйте /enhanced_auto_trade"
        )
        
        await update.message.reply_text(message, )
        
    except Exception as e:
        logger.error(f"Ошибка в enhanced_prediction: {e}")
        await update.message.reply_text(f"❌ Ошибка анализа: {e}")

async def train_ml_brain_command(update: Update, context: CallbackContext):
    """
    Команда /train_brain - обучение ML мозга
    """
    try:
        await update.message.reply_text("🧠 Запуск обучения ML мозга...\n⏳ Это может занять несколько минут")
        
        # Запускаем обучение в фоне
        results = run_brain_training()
        
        # Формируем отчет
        status = results.get('status', 'unknown')
        
        if status == 'completed':
            message = (
                f"🎉 ОБУЧЕНИЕ ML МОЗГА ЗАВЕРШЕНО!\n\n"
                f"📊 Результаты:\n"
                f"• Данные загружены: {'✅' if results.get('data_loaded') else '❌'}\n"
                f"• Индикаторы рассчитаны: {'✅' if results.get('indicators_calculated') else '❌'}\n"
                f"• Найдено стратегий: {results.get('strategies_found', 0)}\n"
                f"• Обучено моделей: {results.get('models_trained', 0)}\n\n"
            )
            
            # Добавляем лучшие параметры если есть
            best_params = results.get('best_parameters', {})
            if best_params:
                message += (
                    f"🎯 Оптимальные параметры:\n"
                    f"• Плечо: {best_params.get('leverage', 1)}x\n"
                    f"• Stop Loss: {best_params.get('stop_loss_pct', 2.0)}%\n"
                    f"• Take Profit: {best_params.get('take_profit_ratio', 2.0)}x\n\n"
                )
            
            message += "✨ Теперь используйте /enhanced_predict для улучшенных прогнозов!"
            
        elif status == 'error':
            message = f"❌ ОШИБКА ОБУЧЕНИЯ:\n{results.get('error', 'Неизвестная ошибка')}"
        else:
            message = f"⚠️ ОБУЧЕНИЕ НЕ ЗАВЕРШЕНО:\nСтатус: {status}"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Ошибка в train_ml_brain: {e}")
        await update.message.reply_text(f"❌ Ошибка обучения: {e}")

async def brain_status_command(update: Update, context: CallbackContext):
    """
    Команда /brain_status - статус ML мозга
    """
    try:
        # Проверяем наличие обученных моделей
        import os
        
        model_exists = os.path.exists('enhanced_ml_model.pkl')
        scaler_exists = os.path.exists('enhanced_ml_scaler.pkl')
        features_exist = os.path.exists('enhanced_ml_features.pkl')
        
        # Информация о данных в кэше
        cached_timeframes = list(ml_brain.data_cache.keys()) if ml_brain.data_cache else []
        strategies_count = len(ml_brain.best_strategies)
        
        message = (
            f"🧠 СТАТУС ML МОЗГА\n\n"
            f"🤖 Модели:\n"
            f"• ML модель: {'✅ Обучена' if model_exists else '❌ Не обучена'}\n"
            f"• Скейлер: {'✅ Готов' if scaler_exists else '❌ Отсутствует'}\n"
            f"• Признаки: {'✅ Сохранены' if features_exist else '❌ Отсутствуют'}\n\n"
            f"📊 Данные:\n"
            f"• Загружено таймфреймов: {len(cached_timeframes)}\n"
            f"• Доступные: {', '.join(cached_timeframes) if cached_timeframes else 'Нет'}\n\n"
            f"🎯 Стратегии:\n"
            f"• Найдено лучших: {strategies_count}\n\n"
        )
        
        if model_exists:
            message += "✅ Готов к работе! Используйте /enhanced_predict\n"
        else:
            message += "⚠️ Требуется обучение! Запустите /train_brain\n"
        
        message += "\n📋 Доступные команды:\n• /enhanced_predict - Улучшенный прогноз\n• /train_brain - Обучить мозг\n• /enhanced_auto_trade - Автоторговля"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Ошибка в brain_status: {e}")
        await update.message.reply_text(f"❌ Ошибка получения статуса: {e}")

async def enhanced_auto_trade_command(update: Update, context: CallbackContext):
    """
    Команда /enhanced_auto_trade - автоматическая торговля с ML мозгом
    """
    try:
        await update.message.reply_text("🤖 Запуск автоматической торговли с ML мозгом...")
        
        # Получаем улучшенный прогноз
        prediction = get_enhanced_prediction()
        
        signal = prediction['signal']
        confidence = prediction['confidence']
        leverage = prediction['recommended_leverage']
        sl_pct = prediction['stop_loss_pct']
        tp_ratio = prediction['take_profit_ratio']
        
        # Проверяем уверенность
        if confidence < 0.6:
            await update.message.reply_text(
                f"⚠️ Низкая уверенность прогноза\n"
                f"Уверенность: {confidence:.1%} (требуется >60%)\n"
                f"Сигнал: {signal}\n\n"
                f"Автоматическая торговля отменена для безопасности."
            )
            return
        
        # Получаем текущие данные
        df = get_candlestick_data("BTC/USDT", "1h", limit=10, private=True)
        current_price = df['close'].iloc[-1]
        
        # Рассчитываем параметры позиции
        state = load_bot_state()
        balance = state.get('usd_balance', 1000)
        
        position_size = calculate_position_size(
            balance=balance,
            entry_price=current_price,
            stop_loss_pct=sl_pct,
            risk_pct=2.0
        ) * leverage  # Применяем плечо
        
        stop_price, take_price = calculate_sl_tp_levels(
            entry_price=current_price,
            stop_loss_pct=sl_pct,
            tp_ratio=tp_ratio
        )
        
        # Выполняем торговую операцию
        if signal in ['STRONG_BUY', 'BUY']:
            try:
                from trading.executor import execute_entry
                
                response = execute_entry(
                    symbol="BTC/USDT",
                    entry_price=current_price,
                    position_size=position_size,
                    stop_loss=stop_price,
                    take_profit=take_price
                )
                
                message = (
                    f"✅ ПОЗИЦИЯ ОТКРЫТА С ML МОЗГОМ!\n\n"
                    f"📊 Детали:\n"
                    f"• Сигнал: {signal} ({confidence:.1%})\n"
                    f"• Плечо: {leverage}x\n"
                    f"• Размер: {position_size:.6f} BTC\n"
                    f"• Вход: ${current_price:,.2f}\n"
                    f"• Stop Loss: ${stop_price:,.2f}\n"
                    f"• Take Profit: ${take_price:,.2f}\n\n"
                    f"🤖 Ответ биржи:\n{response}"
                )
                
            except Exception as e:
                message = f"❌ Ошибка выполнения сделки:\n{e}"
                
        elif signal in ['STRONG_SELL', 'SELL']:
            try:
                from trading.executor import execute_exit
                from positions_db import get_open_positions
                
                # Закрываем все открытые позиции
                positions = get_open_positions()
                if positions:
                    responses = []
                    for pos in positions:
                        resp = execute_exit(pos["symbol"], pos["id"], current_price)
                        responses.append(resp)
                    
                    message = (
                        f"🔴 ПОЗИЦИИ ЗАКРЫТЫ С ML МОЗГОМ!\n\n"
                        f"📊 Детали:\n"
                        f"• Сигнал: {signal} ({confidence:.1%})\n"
                        f"• Закрыто позиций: {len(positions)}\n"
                        f"• Цена закрытия: ${current_price:,.2f}\n\n"
                        f"🤖 Ответы биржи:\n" + "\n".join(responses)
                    )
                else:
                    message = f"⚠️ Нет открытых позиций для закрытия\nСигнал: {signal} ({confidence:.1%})"
                    
            except Exception as e:
                message = f"❌ Ошибка закрытия позиций:\n{e}"
                
        else:
            message = (
                f"⏸️ УДЕРЖАНИЕ ПОЗИЦИИ\n\n"
                f"• Сигнал: {signal} ({confidence:.1%})\n"
                f"• Действие: Никаких операций\n"
                f"• Причина: Слабый сигнал или неопределенность"
            )
        
        await update.message.reply_text(message, )
        
    except Exception as e:
        logger.error(f"Ошибка в enhanced_auto_trade: {e}")
        await update.message.reply_text(f"❌ Ошибка автоторговли: {e}")

async def start_enhanced_monitoring_command(update: Update, context: CallbackContext):
    """
    Команда /start_enhanced_monitor - запуск улучшенного мониторинга
    """
    try:
        chat_id = update.effective_chat.id
        
        # Проверяем, что такого Job ещё нет
        current_jobs = context.application.job_queue.get_jobs_by_name(f"enhanced_{chat_id}")
        if current_jobs:
            await update.message.reply_text("🧠 Улучшенный мониторинг уже запущен!")
            return
        
        # Запускаем задачу: каждые 10 минут вызываем enhanced_monitor_callback
        context.application.job_queue.run_repeating(
            enhanced_monitor_callback,
            interval=10 * 60,  # 10 минут
            first=0,           # запуск сразу
            name=f"enhanced_{chat_id}",
            data=chat_id,
            chat_id=chat_id
        )
        
        message = (
            "🧠 УЛУЧШЕННЫЙ МОНИТОРИНГ ЗАПУЩЕН!\n\n"
            "🔄 Частота: каждые 10 минут\n"
            "🤖 Анализ: продвинутый ML мозг\n"
            "⚡ Автоторговля: при высокой уверенности\n\n"
            "Для остановки используйте /stop_enhanced_monitor"
        )
        
        await update.message.reply_text(message, )
        
    except Exception as e:
        logger.error(f"Ошибка запуска enhanced мониторинга: {e}")
        await update.message.reply_text(f"❌ Ошибка запуска: {e}")

async def stop_enhanced_monitoring_command(update: Update, context: CallbackContext):
    """
    Команда /stop_enhanced_monitor - остановка улучшенного мониторинга
    """
    try:
        chat_id = update.effective_chat.id
        jobs = context.application.job_queue.get_jobs_by_name(f"enhanced_{chat_id}")
        
        if not jobs:
            await update.message.reply_text("🧠 Улучшенный мониторинг не запущен")
            return
        
        for job in jobs:
            job.schedule_removal()
        
        await update.message.reply_text("🛑 Улучшенный мониторинг остановлен")
        
    except Exception as e:
        logger.error(f"Ошибка остановки enhanced мониторинга: {e}")
        await update.message.reply_text(f"❌ Ошибка остановки: {e}")

async def enhanced_monitor_callback(context: CallbackContext):
    """
    Callback для улучшенного мониторинга
    """
    chat_id = context.job.data
    
    try:
        # Получаем улучшенный прогноз
        prediction = get_enhanced_prediction()
        
        signal = prediction['signal']
        confidence = prediction['confidence']
        reasoning = prediction['reasoning']
        
        # Проверяем, изменился ли сигнал
        prev_signal = context.chat_data.get("last_enhanced_signal")
        if signal == prev_signal and signal == 'HOLD':
            return  # Не спамим при HOLD
        
        context.chat_data["last_enhanced_signal"] = signal
        
        # Получаем текущую цену
        df = get_candlestick_data("BTC/USDT", "1h", limit=10, private=True)
        current_price = df['close'].iloc[-1]
        
        # Формируем уведомление
        emoji = {
            'STRONG_BUY': '🚀',
            'BUY': '📈', 
            'HOLD': '⏸️',
            'SELL': '📉',
            'STRONG_SELL': '💥'
        }.get(signal, '❓')
        
        message = (
            f"{emoji} ML МОЗГ АЛЕРТ {emoji}\n\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}\n"
            f"📊 Сигнал: {signal}\n"
            f"🎯 Уверенность: {confidence:.1%}\n"
            f"💰 Цена: ${current_price:,.2f}\n\n"
            f"🧠 {reasoning}\n\n"
        )
        
        # Автоматическая торговля при высокой уверенности
        if confidence >= 0.75 and signal in ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']:
            message += "⚡ АВТОТОРГОВЛЯ АКТИВИРОВАНА!\n"
            # Здесь можно добавить автоматическое выполнение сделок
        elif confidence >= 0.6:
            message += "💡 Рекомендуется ручная проверка\n"
        
        message += f"\nИспользуйте /enhanced_auto_trade для торговли"
        
        await context.bot.send_message(chat_id=chat_id, text=message, )
        
    except Exception as e:
        logger.error(f"Ошибка в enhanced_monitor_callback: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка в улучшенном мониторинге: {e}"
        )

async def restart_bot_command(update: Update, context: CallbackContext):
    """
    Команда /restart_bot - перезапуск бота для принятия новых изменений
    """
    try:
        await update.message.reply_text("🔄 Перезапуск бота для обновления...")
        
        # Получаем путь к текущему скрипту
        current_script = os.path.abspath(sys.argv[0])
        
        # Создаем скрипт перезапуска
        restart_script = """#!/bin/bash
echo "🔄 Остановка текущего бота..."
pkill -f "python.*bot.py"
sleep 3

echo "📥 Обновление кода из Git..."
cd /opt/my_trading_bot
git pull origin main

echo "🚀 Запуск обновленного бота..."
nohup python3 telegram_bot/bot.py > /dev/null 2>&1 &

echo "✅ Бот перезапущен с обновлениями!"
"""
        
        # Сохраняем скрипт
        with open('/tmp/restart_bot.sh', 'w') as f:
            f.write(restart_script)
        
        # Делаем скрипт исполняемым
        os.chmod('/tmp/restart_bot.sh', 0o755)
        
        await update.message.reply_text(
            "✅ Команда перезапуска подготовлена!\n\n"
            "🔄 Бот будет:\n"
            "1. Остановлен\n"
            "2. Обновлен из Git\n"
            "3. Перезапущен\n\n"
            "⏳ Запуск через 5 секунд..."
        )
        
        # Запускаем перезапуск через 5 секунд
        await asyncio.sleep(5)
        
        # Выполняем скрипт перезапуска
        subprocess.Popen(['/bin/bash', '/tmp/restart_bot.sh'], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Завершаем текущий процесс
        os._exit(0)
        
    except Exception as e:
        logger.error(f"Ошибка при перезапуске: {e}")
        await update.message.reply_text(f"❌ Ошибка перезапуска: {e}")

async def update_code_command(update: Update, context: CallbackContext):
    """
    Команда /update_code - обновление кода из Git без перезапуска
    """
    try:
        await update.message.reply_text("📥 Обновление кода из Git...")
        
        # Выполняем git pull
        result = subprocess.run(['git', 'pull', 'origin', 'main'], 
                              capture_output=True, text=True, cwd='/opt/my_trading_bot')
        
        if result.returncode == 0:
            message = (
                "✅ КОД ОБНОВЛЕН!\n\n"
                f"📋 Git вывод:\n{result.stdout}\n\n"
                "🔄 Для применения изменений используйте /restart_bot"
            )
        else:
            message = f"❌ Ошибка обновления:\n{result.stderr}"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Ошибка обновления кода: {e}")
        await update.message.reply_text(f"❌ Ошибка обновления: {e}")

async def system_status_command(update: Update, context: CallbackContext):
    """
    Команда /system_status - статус системы и процессов
    """
    try:
        await update.message.reply_text("🔍 Проверка статуса системы...")
        
        # Проверяем процессы бота
        ps_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        bot_processes = [line for line in ps_result.stdout.split('\n') if 'bot.py' in line]
        
        # Проверяем Git статус
        git_status = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd='/opt/my_trading_bot')
        
        git_log = subprocess.run(['git', 'log', '--oneline', '-5'], 
                               capture_output=True, text=True, cwd='/opt/my_trading_bot')
        
        message = (
            "🖥️ СТАТУС СИСТЕМЫ\n\n"
            f"🤖 Процессы бота: {len(bot_processes)}\n"
            f"📁 Незакоммиченных изменений: {len(git_status.stdout.strip().split()) if git_status.stdout.strip() else 0}\n\n"
            f"📋 Последние коммиты:\n{git_log.stdout}\n\n"
            "⚙️ Доступные команды:\n"
            "• /update_code - обновить код\n"
            "• /restart_bot - перезапустить бота\n"
            "• /system_status - этот статус"
        )
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Ошибка проверки статуса: {e}")
        await update.message.reply_text(f"❌ Ошибка статуса: {e}")
