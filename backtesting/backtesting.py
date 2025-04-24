import pandas as pd
import logging
from trading.risk_manager import calculate_position_size, calculate_sl_tp_levels
from utils.config import DEFAULT_RISK_PCT, DEFAULT_SL_PCT, DEFAULT_TP_RATIO

# включаем вывод INFO-сообщений
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, initial_balance=10000, commission_rate=0.001):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.balance = initial_balance
        self.trades = []  # записи по сделкам

    def simulate_trade(self, entry_price, exit_price, position_size, exit_type):
        # Рассчитываем комиссию на вход и выход
        commission = (entry_price * position_size + exit_price * position_size) * self.commission_rate
        # Прибыль с учётом комиссии
        profit = (exit_price - entry_price) * position_size - commission
        self.balance += profit

        trade_info = {
            "entry_price":   entry_price,
            "exit_price":    exit_price,
            "position_size": position_size,
            "profit":        profit,
            "balance":       self.balance,
            "exit_type":     exit_type
        }
        self.trades.append(trade_info)
        logger.info(f"Сделка ({exit_type}) проведена: {trade_info}")
        return trade_info

    def run_backtest(self, df, signal_column="signal"):
        in_position = False
        entry_price = None
        position_size = 0.0
        stop_price = None
        take_price = None

        for _, row in df.iterrows():
            price = row["close"]
            sig = row[signal_column]

            if in_position:
                # Лог уровней и бара
                logger.info(
                    f"Bar low={row['low']:.2f}, SL={stop_price:.2f}; "
                    f"Bar high={row['high']:.2f}, TP={take_price:.2f}"
                )
                # Стоп‑лосс
                if row["low"] <= stop_price:
                    self.simulate_trade(entry_price, stop_price, position_size, exit_type="SL")
                    in_position = False
                    logger.info(f"SL hit at {stop_price:.2f}")
                    continue
                # Тейк‑профит
                if row["high"] >= take_price:
                    self.simulate_trade(entry_price, take_price, position_size, exit_type="TP")
                    in_position = False
                    logger.info(f"TP hit at {take_price:.2f}")
                    continue
                # Закрытие по сигналу SELL
                if sig == "SELL":
                    self.simulate_trade(entry_price, price, position_size, exit_type="SELL")
                    in_position = False
                    logger.info(f"Close by SELL at {price:.2f}")
                    continue

            # Открываем позицию по сигналу BUY
            if sig == "BUY" and not in_position:
                entry_price = price
                position_size = calculate_position_size(
                    balance=self.balance,
                    entry_price=entry_price,
                    stop_loss_pct=DEFAULT_SL_PCT,
                    risk_pct=DEFAULT_RISK_PCT
                )
                stop_price, take_price = calculate_sl_tp_levels(
                    entry_price=entry_price,
                    stop_loss_pct=DEFAULT_SL_PCT,
                    tp_ratio=DEFAULT_TP_RATIO
                )
                in_position = True
                logger.info(
                    f"OPEN ▶ price={entry_price:.2f}, size={position_size:.6f}, "
                    f"SL={stop_price:.2f}, TP={take_price:.2f}"
                )

        logger.info(f"Бэктест завершен. Итоговый баланс: {self.balance:.2f}")
        return self.trades


if __name__ == "__main__":
    from data.data_manager import get_candlestick_data
    from indicators.indicators import calculate_indicators
    from collections import Counter
    import matplotlib.pyplot as plt

    # 1) Загружаем данные
    df = get_candlestick_data(symbol="BTC/USDT", timeframe="1h")
    if df.empty:
        print("Не удалось получить данные для бэктеста")
        exit(1)

    # 2) Считаем индикаторы
    df = calculate_indicators(df)

    # 3) Формируем сигналы по RSI
    df["signal"] = df["RSI"].apply(
        lambda rsi: "BUY" if rsi < 30 else ("SELL" if rsi > 70 else "HOLD")
    )

    # 4) Запускаем бэктест
    backtester = Backtester(initial_balance=10000, commission_rate=0.001)
    trades = backtester.run_backtest(df, signal_column="signal")

    # 5) Итоговая статистика
    print("\n=== Результаты бэктеста ===")
    print(f"Итоговый баланс: {backtester.balance:.2f}")
    print(f"Всего сделок: {len(trades)}")
    if trades:
        avg_profit = sum(t["profit"] for t in trades) / len(trades)
        print(f"Средняя прибыль на сделку: {avg_profit:.2f}")

    # 6) Статистика по типам выхода
    exit_types = [t["exit_type"] for t in trades]
    counts = Counter(exit_types)
    print("\n— По типам выхода —")
    print(f"SL сработал:       {counts.get('SL', 0)} раз(а)")
    print(f"TP сработал:       {counts.get('TP', 0)} раз(а)")
    print(f"Закрыто по SELL:   {counts.get('SELL', 0)} раз(а)")

    # 7) Собираем equity-кривую
    equity = [backtester.initial_balance] + [t["balance"] for t in trades]

    # 8) Дополнительные метрики (опционально)
    peak = equity[0]
    drawdowns = []
    profits = [t["profit"] for t in trades]
    for v in equity:
        peak = max(peak, v)
        drawdowns.append((peak - v) / peak)
    max_dd = max(drawdowns) * 100
    gross_win = sum(p for p in profits if p > 0)
    gross_loss = -sum(p for p in profits if p < 0)
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    print(f"\nMax Drawdown:  {max_dd:.2f}%")
    print(f"Profit Factor: {pf:.2f}")

    # 9) Рисуем equity-кривую
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.show()
