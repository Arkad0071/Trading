# telegram_bot/bot.py
import os
import logging
import joblib
import numpy as np
from datetime import time
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)

logger = logging.getLogger(__name__)


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
        if prob > 60:
            signal = "STRONG_BUY"
        elif prob > 55:
            signal = "BUY"
        elif prob < 40:
            signal = "STRONG_SELL"
        elif prob < 45:
            signal = "SELL"
        else:
            signal = "HOLD"

        # динамический процент риска по ATR
        atr = df["ATR"].iloc[-1]
        last_close = df["close"].iloc[-1]
        atr_pct = atr / last_close
        risk_pct = min(max(atr_pct * 1.5, 0.005), 0.03)  # от 0.5% до 3%

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

        log_prediction(signal, prob, entry_price)

        position_size = calculate_position_size(
            balance=usd_balance,
            entry_price=entry_price,
            stop_loss_pct=DEFAULT_SL_PCT,
            risk_pct=risk_pct
        )
        stop_price, take_price = calculate_sl_tp_levels(
            entry_price=entry_price,
            stop_loss_pct=DEFAULT_SL_PCT,
            tp_ratio=DEFAULT_TP_RATIO
        )

        # ─── 7) Формируем и отправляем уведомление ───────────────────
        text = (
            f"⏰ Мониторинг сигнала:\n"
            f"📈 Вероятность: {prob:.2f}%\n"
            f"🔔 {signal.replace('_', ' ')}\n"
            f"• Risk: {risk_pct * 100:.2f}%\n"
            f"• Объём: {position_size:.6f} BTC\n"  # ← вот она
            f"• Entry: {entry_price:.2f}\n"
            f"• SL: {stop_price:.2f}\n"
            f"• TP: {take_price:.2f}"
        )

        await context.bot.send_message(chat_id=chat_id, text=text)

    except Exception as e:
        logger.error(f"Ошибка в monitor_callback: {e}")


async def chart_command(update: Update, context: CallbackContext):
    try:
        await update.message.reply_text("🔄 Запуск прогноза модели...")

        # 1) загрузили состояние
        state = load_bot_state()
        usd_balance = state["usd_balance"]
        btc_balance = state["btc_balance"]
        entry_price    = state["entry_price"]
        stop_loss_price= state["stop_loss"]
        take_profit_price = state["take_profit"]
        fraction      = state.get("fraction", 0.3)
        risk_per_trade= state.get("risk_per_trade", DEFAULT_RISK_PCT)

        # параметры
        slippage_rate   = 0.01
        commission_rate = 0.001

        # 2) данные и индикаторы
        df = get_candlestick_data("BTC/USDT", "1h")
        if df.empty:
            return await update.message.reply_text("Нет данных для прогноза.")
        df = calculate_indicators(df)

        # 3) подготовка фич
        features = ["close","volume","RSI","MACD","MACD_signal","ATR"]
        X, _, _ = prepare_features(df, features, sequence_length=50)
        if not len(X):
            return await update.message.reply_text("Недостаточно исторических свечей.")

        # 4) загрузили модель
        if not os.path.exists("scaler.pkl") or not os.path.exists("lstm_model.h5"):
            return await update.message.reply_text("Сначала выполните /train.")
        model = LSTMModel(50, len(features))
        model.load_model("lstm_model.h5")
        prob = model.predict(X[-1].reshape(1,50,len(features)))[0][0] * 100

        # 5) определяем сигналы
        if prob > 60:
            signal = "STRONG_BUY"
        elif prob > 55:
            signal = "BUY"
        elif prob < 40:
            signal = "STRONG_SELL"
        elif prob < 45:
            signal = "SELL"
        else:
            signal = "HOLD"

        details = ""
        # только для покупок делаем расчёт позиции
        if signal in ("BUY", "STRONG_BUY"):
            last = df["close"].iloc[-1]
            entry_price = last * (1 + slippage_rate)
            position_size = calculate_position_size(
                balance=usd_balance,
                entry_price=entry_price,
                stop_loss_pct=DEFAULT_SL_PCT,
                risk_pct=risk_per_trade
            )
            stop_price, take_price = calculate_sl_tp_levels(
                entry_price, DEFAULT_SL_PCT, DEFAULT_TP_RATIO
            )
            cost = position_size * entry_price * (1 + commission_rate)
            if cost > usd_balance or position_size <= 0:
                details = "❗ Недостаточно средств или расчёт объёма неверен.\n"
            else:
                usd_balance -= cost
                btc_balance  += position_size
                stop_loss_price = stop_price
                take_profit_price = take_price
                details = (
                    f"💰 {signal}:\n"
                    f"• Объём: {position_size:.6f} BTC\n"
                    f"• Вход:  {entry_price:.2f} USDT\n"
                    f"• SL:    {stop_price:.2f}\n"
                    f"• TP:    {take_price:.2f}\n"
                )

        # 6) логируем прогноз
        log_prediction(signal, prob, entry_price)
        logger.info(f"[PREDICTION] signal={signal} prob={prob:.2f}% entry={entry_price:.2f}")

        # 7) выводим в чат
        msg = (
            f"🔎 Прогноз модели:\n"
            f"📈 Вероятность роста: {prob:.2f}%\n"
            f"🔔 Сигнал: {signal}\n\n"
            f"{details}"
        )
        await update.message.reply_text(msg)

        # 8) сохраняем новое состояние
        if signal == "BUY":
            new_in_trade = True
        elif signal == "SELL":
            new_in_trade = False
        else:
            new_in_trade = state["in_trade"]

        save_bot_state(
            usd_balance=usd_balance,
            btc_balance=btc_balance,
            entry_price=entry_price,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            fraction=fraction,
            risk_per_trade=risk_per_trade,
            in_trade = new_in_trade
        )

    except Exception as e:
        logger.error(f"Ошибка в /chart: {e}")
        await update.message.reply_text(f"Ошибка: {e}")

async def open_position_cmd(update: Update, context: CallbackContext):
    """Пользователь подтвердил вход по последнему сигналу."""
    state = load_bot_state()
    ep = state["entry_price"]
    sl = state["stop_loss"]
    tp = state["take_profit"]
    if not ep or sl is None or tp is None:
        return await update.message.reply_text("Нет данных по последнему сигналу, сначала /chart")
    # 1) переключаем флаг
    open_position(ep, sl, tp)
    # 2) запускаем задачу проверки каждую минуту
    chat_id = update.effective_chat.id
    # если уже есть, удалим
    for job in context.application.job_queue.get_jobs_by_name(f"trade_{chat_id}"):
        job.schedule_removal()
    context.application.job_queue.run_repeating(
        monitor_trade_callback,
        interval=60,  # каждую минуту
        first=0,
        name=f"trade_{chat_id}",
        data=chat_id
    )
    await update.message.reply_text(f"Позиция открыта по {ep:.2f}. SL={sl:.2f}, TP={tp:.2f}. Мониторинг запущен.")

async def monitor_trade_callback(context: CallbackContext):
    chat_id = context.job.data
    state = load_bot_state()
    if not state.get("in_trade", False):
        return  # вы уже закрыли

    price = get_candlestick_data("BTC/USDT","1m")["close"].iloc[-1]
    sl = state["stop_loss"]
    tp = state["take_profit"]

    if price <= sl:
        await context.bot.send_message(chat_id, f"⚠️ STOP-LOSS hit: {price:.2f} ≤ SL={sl:.2f}")
        log_trade(entry_price=state["entry_price"], exit_price=sl, position_size=..., profit=...)
        close_position()
        context.job.schedule_removal()
    elif price >= tp:
        await context.bot.send_message(chat_id, f"✅ TAKE-PROFIT hit: {price:.2f} ≥ TP={tp:.2f}")
        log_trade(entry_price=state["entry_price"], exit_price=tp, position_size=..., profit=...)
        close_position()
        context.job.schedule_removal()


async def close_position_cmd(update: Update, context: CallbackContext):
    """Пользователь закрыл позицию вручную."""
    chat_id = update.effective_chat.id
    close_position()
    jobs = context.application.job_queue.get_jobs_by_name(f"trade_{chat_id}")
    for j in jobs: j.schedule_removal()
    await update.message.reply_text("Позиция закрыта, мониторинг остановлен.")


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

async def retrain_callback(context: CallbackContext):
    logger.info("=== Начинаю автоматическое переобучение ===")
    # 1) Получаем и готовим данные
    df = get_candlestick_data(symbol="BTC/USDT", timeframe="1h")
    if df.empty:
        logger.error("Retrain: не удалось скачать данные")
        return
    df = calculate_indicators(df)
    features = ["close", "volume", "RSI", "MACD", "MACD_signal", "ATR"]
    seq_len = 50
    X, y, scaler = prepare_features(df, features=features, sequence_length=seq_len)
    if len(X) == 0:
        logger.error("Retrain: недостаточно данных для обучения")
        return

    # 2) Обучаем модель
    model = LSTMModel(sequence_length=seq_len, num_features=len(features))
    # не будем отправлять в чат, только логируем прогресс
    model.train(X, y, epochs=10, batch_size=32)
    model.save_model("lstm_model.h5")
    joblib.dump(scaler, "scaler.pkl")

    logger.info("=== Переобучение завершено ===")


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

        # ————— Логируем каждую симулированную сделку в БД и в системный лог —————
        for i, t in enumerate(trades, start=1):
            log_trade(
                entry_price=t["entry_price"],
                exit_price=t["exit_price"],
                position_size=t["position_size"],
                profit=t["profit"]
            )
            logger.info(
                f"[BACKTEST TRADE #{i}] "
                f"entry={t['entry_price']:.2f} "
                f"exit={t['exit_price']:.2f} "
                f"size={t['position_size']:.6f} "
                f"profit={t['profit']:.2f} "
                f"balance={t['balance']:.2f} "
                f"exit_type={t.get('exit_type', '-')}"
            )
        # ——————————————————————————————————————————————————————————————

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

    logger.info(
        f"[BACKTEST SUMMARY] "
        f"final_balance={backtester.balance:.2f} "
        f"trades={len(trades)} "
        f"max_dd={max_dd:.2f}% "
        f"pf={pf:.2f}"
    )

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
    application.add_handler(CommandHandler("open_position", open_position_cmd))
    application.add_handler(CommandHandler("close_position", close_position_cmd))

    # планируем retrain_callback на 00:00 UTC каждый день
    application.job_queue.run_daily(
        retrain_callback,
        time=time(hour=0, minute=0, second=0),
        name="daily_retrain"
    )

    logger.info("Telegram-бот запущен.")
    application.run_polling()

init_bot_state_table()

if __name__ == "__main__":
    main()