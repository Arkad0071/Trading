# telegram_bot/bot.py
import os
import logging
import joblib
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
from utils.config import TOKEN, BYBIT_API_KEY, BYBIT_API_SECRET
from utils.config import DEFAULT_RISK_PCT, DEFAULT_SL_PCT, DEFAULT_TP_RATIO
from data.data_manager import get_candlestick_data
from indicators.indicators import calculate_indicators
from models.preprocess import prepare_features
from models.lstm_model import LSTMModel
from backtesting.backtesting import Backtester
from trading.risk_manager import calculate_position_size, calculate_sl_tp_levels
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from positions_db import load_bot_state, save_bot_state
from positions_db import init_bot_state_table
from positions_db import log_prediction, log_trade
from positions_db import log_trade
from telegram.ext import JobQueue, Job
import io
import matplotlib.pyplot as plt
from collections import Counter


# Импортируем обратные вызовы для TensorFlow (если необходимо)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

logger = logging.getLogger(__name__)


async def status_command(update: Update, context: CallbackContext):
    try:
        await update.message.reply_text("Получаю текущее состояние стратегии...")

        # Получаем исторические данные (например, последние 300 свечей)
        df = get_candlestick_data(symbol="BTC/USDT", timeframe="1h")
        if df.empty:
            await update.message.reply_text("Не удалось получить данные для отображения состояния.")
            return

        # Рассчитываем индикаторы для этих данных
        df = calculate_indicators(df)

        # Генерируем простые сигналы по правилу (например, на основе RSI)
        # Если RSI < 30, то сигнал BUY, если RSI > 70, то SELL, иначе HOLD
        df["signal"] = df["RSI"].apply(lambda rsi: "BUY" if rsi < 30 else ("SELL" if rsi > 70 else "HOLD"))

        # Создаем объект бэктестера и запускаем бэктест
        backtester = Backtester(initial_balance=10000, commission_rate=0.001)
        trades = backtester.run_backtest(df, signal_column="signal")

        # Формируем отчет: итоговый баланс и количество сделок
        report = f"Состояние стратегии:\nИтоговый баланс: {backtester.balance:.2f}\n" \
                 f"Количество сделок: {len(trades)}"
        await update.message.reply_text(report)
    except Exception as e:
        logger.error(f"Ошибка в команде /status: {e}")
        await update.message.reply_text(f"Ошибка при получении статуса: {e}")

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Привет! Я торговый бот. Доступные команды: /train, /chart, /backtest, /status.")

async def start_monitor(update: Update, context: CallbackContext):
    """Запускает фоновую проверку сигнала каждые 15 минут."""
    chat_id = update.effective_chat.id

    # Проверяем, что такого Job ещё нет
    current_jobs = context.application.job_queue.get_jobs_by_name(str(chat_id))
    if current_jobs:
        await update.message.reply_text("Мониторинг уже запущен.")
        return

    # Запускаем задачу: каждые 15 мин вызываем функцию monitor_callback
    context.application.job_queue.run_repeating(
        monitor_callback,
        interval=15 * 60,      # 15 минут
        first=0,               # запуск сразу
        name=str(chat_id),     # имя Job = чат id
        data=chat_id        # передаём chat_id в колбэк
    )
    await update.message.reply_text("Мониторинг запущен: буду присылать сигналы каждые 15 мин.")

async def stop_monitor(update: Update, context: CallbackContext):
    """Останавливает фоновые проверки."""
    chat_id = update.effective_chat.id
    jobs = context.application.job_queue.get_jobs_by_name(str(chat_id))
    if not jobs:
        await update.message.reply_text("Мониторинг не запущен.")
        return
    for job in jobs:
        job.schedule_removal()
    await update.message.reply_text("Мониторинг остановлен.")

async def monitor_callback(context: CallbackContext):
    chat_id = context.job.data

    try:
        # ─── 1) Загрузим состояние ─────────────────────────────────────
        state = load_bot_state()
        usd_balance = state["usd_balance"]

        # ─── 2) Получаем и подготавливаем данные ──────────────────────
        df = get_candlestick_data(symbol="BTC/USDT", timeframe="1h")
        if df.empty:
            return

        df = calculate_indicators(df)
        features = ["close", "volume", "RSI", "MACD", "MACD_signal", "ATR"]
        sequence_length = 50
        X, _, _ = prepare_features(df, features=features, sequence_length=sequence_length)
        if len(X) == 0:
            return

        # ─── 3) Прогнозируем ─────────────────────────────────────────
        scaler = joblib.load("scaler.pkl")
        model = LSTMModel(sequence_length=sequence_length, num_features=len(features))
        model.load_model("lstm_model.h5")

        X_last = X[-1].reshape(1, sequence_length, len(features))
        prob = model.predict(X_last)[0][0] * 100  # в процентах

        # ─── 4) Определяем сигнал ─────────────────────────────────────
        signal = "HOLD"
        if prob > 55:
            signal = "BUY"
        elif prob < 45:
            signal = "SELL"

        log_prediction(signal, prob, entry_price)

        # ─── 5) Проверяем, изменился ли сигнал ────────────────────────
        prev = context.chat_data.get("last_signal")
        if signal == prev:
            return        # если нет изменений — уходим без уведомления
        context.chat_data["last_signal"] = signal

        # ─── 6) Рассчитываем entry_price, объём и уровни SL/TP ─────────
        slippage_rate = 0.01
        if signal == "BUY":
            entry_price = df["close"].iloc[-1] * (1 + slippage_rate)
        else:  # SELL
            entry_price = df["close"].iloc[-1] * (1 - slippage_rate)

        position_size = calculate_position_size(
            balance=usd_balance,
            entry_price=entry_price,
            stop_loss_pct=DEFAULT_SL_PCT,
            risk_pct=DEFAULT_RISK_PCT
        )
        stop_price, take_price = calculate_sl_tp_levels(
            entry_price=entry_price,
            stop_loss_pct=DEFAULT_SL_PCT,
            tp_ratio=DEFAULT_TP_RATIO
        )

        # ─── 7) Формируем и отправляем уведомление ───────────────────
        text = (
            f"⏰ Мониторинг сигнала:\n"
            f"📈 Вероятность роста: {prob:.2f}%\n"
            f"🔔 Сигнал: {signal}\n\n"
            f"• Вход: {entry_price:.2f} USDT\n"
            f"• Объём: {position_size:.6f} BTC\n"
            f"• SL: {stop_price:.2f}, TP: {take_price:.2f}"
        )
        await context.bot.send_message(chat_id=chat_id, text=text)

    except Exception as e:
        logger.error(f"Ошибка в monitor_callback: {e}")


async def chart_command(update: Update, context: CallbackContext):
    try:
        await update.message.reply_text("Запуск прогноза модели...")

        # ─── ЗАГРУЗКА СОСТОЯНИЯ И ПАРАМЕТРОВ ──────────────────────────────
        state = load_bot_state()
        usd_balance      = state["usd_balance"]
        btc_balance      = state["btc_balance"]
        entry_price = state["entry_price"]  # <- инициализируем
        stop_loss_price = state["stop_loss"]  # <- чтобы всегда были переменные
        take_profit_price = state["take_profit"]
        fraction         = state.get("fraction", 0.3)
        risk_per_trade   = state.get("risk_per_trade", 0.02)

        # параметры проскальзывания и комиссии
        slippage_rate    = 0.01   # 1%
        commission_rate  = 0.001  # 0.1%
        # ────────────────────────────────────────────────────────────────


        # 1. Получаем свежие данные
        df = get_candlestick_data(symbol="BTC/USDT", timeframe="1h")
        if df.empty:
            await update.message.reply_text("Не удалось получить данные. Попробуйте позже.")
            return

        # 2. Рассчитываем индикаторы
        df = calculate_indicators(df)

        # 3. Подготавливаем данные для модели
        features = ["close", "volume", "RSI", "MACD", "MACD_signal", "ATR"]
        sequence_length = 50
        X, _, _ = prepare_features(df, features=features, sequence_length=sequence_length)
        if len(X) == 0:
            await update.message.reply_text("Недостаточно данных для прогноза.")
            return

        # Загружаем scaler (предполагается, что он был сохранён в /train)
        if not os.path.exists("scaler.pkl"):
            await update.message.reply_text("Scaler не найден. Сначала выполните /train.")
            return
        scaler = joblib.load("scaler.pkl")

        # Создаем объект модели и загружаем сохранённую модель (предполагается, что файл существует)
        model = LSTMModel(sequence_length=sequence_length, num_features=len(features))
        if not os.path.exists("lstm_model.h5"):
            await update.message.reply_text("Модель не найдена. Сначала выполните /train.")
            return
        model.load_model("lstm_model.h5")

        # 5. Прогнозируем на последней последовательности данных
        X_last = X[-1].reshape(1, sequence_length, len(features))
        prediction = model.predict(X_last)[0][0]  # Получаем вероятность
        probability = prediction * 100  # перевод в проценты

        signal = "HOLD"
        details_message = ""

        # 6. Формируем интерпретацию сигнала
        threshold_buy = 55.0  # 55%
        threshold_sell = 45.0  # 45%
        if probability > threshold_buy:
            signal = "BUY"
            last_close = df["close"].iloc[-1]
            # 1) рассчитываем цену входа с проскальзыванием
            entry_price = last_close * (1 + slippage_rate)

            # 2) рассчитываем объём позиции (BTC) по риску
            position_size = calculate_position_size(
                balance=usd_balance,
                entry_price=entry_price,
                stop_loss_pct=DEFAULT_SL_PCT,
                risk_pct=DEFAULT_RISK_PCT
            )

            # 3) рассчитываем уровни SL и TP
            stop_price, take_price = calculate_sl_tp_levels(
                entry_price=entry_price,
                stop_loss_pct=DEFAULT_SL_PCT,
                tp_ratio=DEFAULT_TP_RATIO
            )

            # 4) проверяем, хватает ли средств (учитываем комиссию)
            cost = position_size * entry_price * (1 + commission_rate)
            if cost > usd_balance or position_size <= 0:
                details_message += "❗ Недостаточно средств или расчёт объёма неверен.\n"
            else:
                # 5) списываем USDT, добавляем BTC и формируем сообщение
                usd_balance -= cost
                btc_balance += position_size

                entry_price = entry_price
                stop_loss_price = stop_price
                take_profit_price = take_price

                details_message += (
                    f"💰 BUY сигнал:\n"
                    f"• Объём: {position_size:.6f} BTC\n"
                    f"• Цена входа: {entry_price:.2f} USDT\n"
                    f"• SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}\n"
                )

        elif probability < threshold_sell:
            signal = "SELL"

        # 7. Формируем итоговое сообщение
        message = (
            f"🔎 Прогноз модели:\n"
            f"📈 Вероятность роста: {probability:.2f}%\n"
            f"Сигнал: {signal}\n\n"
            f"{details_message}"
        )
        await update.message.reply_text(message)
        save_bot_state(
            usd_balance=usd_balance,
            btc_balance=btc_balance,
            entry_price=entry_price,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            fraction=fraction,
            risk_per_trade=risk_per_trade
        )

    except Exception as e:
        logger.error(f"Ошибка в команде /chart: {e}")
        await update.message.reply_text(f"Ошибка при прогнозировании: {e}")

async def train_command(update: Update, context: CallbackContext):
    try:
        await update.message.reply_text("Начинаем обучение модели...")

        # 1. Получаем данные с биржи (или из базы)
        df = get_candlestick_data(symbol="BTC/USDT", timeframe="1h")
        if df.empty:
            await update.message.reply_text("Не удалось получить данные. Попробуйте позже.")
            return

        # 2. Рассчитываем технические индикаторы
        df = calculate_indicators(df)

        # 3. Подготовка данных: выбираем признаки и создаем обучающие последовательности
        features = ["close", "volume", "RSI", "MACD", "MACD_signal", "ATR"]
        sequence_length = 50
        X, y, scaler = prepare_features(df, features=features, sequence_length=sequence_length)
        if len(X) == 0:
            await update.message.reply_text("Недостаточно данных для обучения.")
            return

        # 4. Создаем и обучаем модель
        model = LSTMModel(sequence_length=sequence_length, num_features=len(features))

        # Устанавливаем обратные вызовы: ранняя остановка и сохранение лучшей модели по метрике loss
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True, verbose=1)

        # Обучение модели (параметры epochs и batch_size можно регулировать)
        model.train(X, y, epochs=10, batch_size=32, callbacks=[early_stop, checkpoint])

        # 5. Сохраняем модель и scaler для будущего использования (например, для прогнозирования)
        model.save_model("lstm_model.h5")
        joblib.dump(scaler, "scaler.pkl")

        await update.message.reply_text("Обучение завершено и модель сохранена!")

    except Exception as e:
        logger.error(f"Ошибка в команде /train: {e}")
        await update.message.reply_text(f"Ошибка при обучении: {e}")


# Пока оставляем простую команду /backtest как пример (пока не до конца реализован)
async def backtest_command(update: Update, context: CallbackContext):
    try:
        await update.message.reply_text("Запуск бэктеста стратегии...")

        # Получаем исторические данные (например, последние 300 свечей)
        df = get_candlestick_data(symbol="BTC/USDT", timeframe="1h")
        if df.empty:
            await update.message.reply_text("Не удалось получить исторические данные.")
            return

        # Рассчитываем индикаторы
        df = calculate_indicators(df)

        # Здесь нам нужно сформировать сигналы для торговли.
        # Для простоты рассмотрим такой вариант: используем прогноз модели на каждую свечу.
        # Или можно применить простую стратегию, например:
        # Если RSI < 30 -> BUY, если RSI > 70 -> SELL, иначе HOLD.
        # В данном примере создадим столбец 'signal' с простыми правилами.
        df["signal"] = df["RSI"].apply(lambda rsi: "BUY" if rsi < 30 else ("SELL" if rsi > 70 else "HOLD"))

        # Создаем объект бэктестера
        backtester = Backtester(initial_balance=10000, commission_rate=0.001)
        trades = backtester.run_backtest(df, signal_column="signal")

        # Формируем отчет по бэктестингу
        report = f"Бэктест завершен.\nИтоговый баланс: {backtester.balance:.2f}\nКоличество сделок: {len(trades)}"
        await update.message.reply_text(report)
    except Exception as e:
        logger.error(f"Ошибка в команде /backtest: {e}")
        await update.message.reply_text(f"Ошибка при бэктестинге: {e}")

async def backtest_report(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    await update.message.reply_text("Запускаю полный отчёт по бэктесту…")

    # 1) Получаем данные и сигналы
    df = get_candlestick_data("BTC/USDT", "1h")
    df = calculate_indicators(df)
    df["signal"] = df["RSI"].apply(lambda rsi: "BUY" if rsi < 30 else ("SELL" if rsi > 70 else "HOLD"))

    # 2) Прогоним бэктест
    backtester = Backtester(initial_balance=10000, commission_rate=0.001)
    trades = backtester.run_backtest(df, signal_column="signal")
    # Логируем все сделки из бэктеста
    for t in trades:
        log_trade(
            entry_price=t["entry_price"],
            exit_price=t["exit_price"],
            position_size=t["position_size"],
            profit=t["profit"]
        )

    # 3) Собираем equity-кривую
    equity = [backtester.initial_balance] + [t["balance"] for t in trades]

    # 4) Вычисляем метрики
    #   Max Drawdown
    peak = equity[0]
    drawdowns = []
    for v in equity:
        peak = max(peak, v)
        drawdowns.append((peak - v) / peak)
    max_dd = max(drawdowns) * 100

    #   Profit Factor
    profits = [t["profit"] for t in trades]
    gross_win  = sum(p for p in profits if p > 0)
    gross_loss = -sum(p for p in profits if p < 0)
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # 5) Визуализация кривой
    buf = io.BytesIO()
    plt.figure()
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # 6) Отправляем картинку
    await context.bot.send_photo(chat_id=chat_id, photo=buf)

    # 7) Отправляем текстовый отчёт
    report = (
        f"📊 Бэктест завершён\n"
        f"Итоговый баланс: {backtester.balance:.2f}\n"
        f"Всего сделок:    {len(trades)}\n"
        f"Средняя прибыль: {sum(profits)/len(trades):.2f}\n"
        f"Max Drawdown:    {max_dd:.2f}%\n"
        f"Profit Factor:   {pf:.2f}\n"
    )
    # и ещё разбивка по типам выхода
    exit_counts = Counter(t["exit_type"] for t in trades)
    report += (
        f"SL: {exit_counts.get('SL',0)}, "
        f"TP: {exit_counts.get('TP',0)}, "
        f"SELL: {exit_counts.get('SELL',0)}"
    )
    await update.message.reply_text(report)


def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("train", train_command))
    application.add_handler(CommandHandler("chart", chart_command))
    application.add_handler(CommandHandler("backtest", backtest_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("monitor", start_monitor))
    application.add_handler(CommandHandler("stop_monitor", stop_monitor))
    application.add_handler(CommandHandler("backtest_report", backtest_report))
    logger.info("Telegram-бот запущен.")
    application.run_polling()

init_bot_state_table()

if __name__ == "__main__":
    main()