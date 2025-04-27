# positions_db.py
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("market_data.db")

def init_bot_state_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 1) Создаём таблицу (если её ещё нет) с правильным синтаксисом
    c.execute('''
        CREATE TABLE IF NOT EXISTS bot_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            usd_balance    REAL    DEFAULT 0,
            btc_balance    REAL    DEFAULT 0,
            entry_price    REAL    DEFAULT 0,
            stop_loss      REAL    DEFAULT 0,
            take_profit    REAL    DEFAULT 0,
            fraction       REAL    DEFAULT 0.3,
            risk_per_trade REAL    DEFAULT 0.02,
            in_trade       INTEGER DEFAULT 0  -- 0 == False, 1 == True
        )
    ''')
    # 2) Миграция: если мы подхватили уже существующую БД без колонки in_trade, добавим её
    cols = [row[1] for row in c.execute("PRAGMA table_info(bot_state)")]
    if "in_trade" not in cols:
        c.execute("ALTER TABLE bot_state ADD COLUMN in_trade INTEGER DEFAULT 0")

    # 3) Если таблица совсем пустая — вставим дефолтную запись
    c.execute("SELECT COUNT(*) FROM bot_state")
    if c.fetchone()[0] == 0:
        c.execute('''
            INSERT INTO bot_state
                (id, usd_balance, btc_balance, entry_price, stop_loss, take_profit, fraction, risk_per_trade, in_trade)
            VALUES
                (1, 0, 0, 0, 0, 0, 0.3, 0.02, 0)
        ''')

    conn.commit()
    conn.close()



def init_logs_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # таблица для записей прогнозов
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            ts           TEXT,
            signal       TEXT,
            probability  REAL,
            price        REAL
        )
    """ )
    # таблица для записей симулированных/реальных сделок
    c.execute("""
        CREATE TABLE IF NOT EXISTS executed_trades (
            ts             TEXT,
            entry_price    REAL,
            exit_price     REAL,
            position_size  REAL,
            profit         REAL
        )
    """ )
    conn.commit()
    conn.close()


def load_bot_state():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT usd_balance, btc_balance, entry_price, stop_loss, take_profit,
               fraction, risk_per_trade, in_trade
        FROM bot_state WHERE id = 1
    ''')
    row = c.fetchone()
    conn.close()
    return {
        "usd_balance":    row[0],
        "btc_balance":    row[1],
        "entry_price":    row[2],
        "stop_loss":      row[3],
        "take_profit":    row[4],
        "fraction":       row[5],
        "risk_per_trade": row[6],
        "in_trade":       bool(row[7]),
    }

def save_bot_state(
    usd_balance, btc_balance,
    entry_price, stop_loss, take_profit,
    fraction, risk_per_trade, in_trade
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE bot_state SET
          usd_balance    = ?,
          btc_balance    = ?,
          entry_price    = ?,
          stop_loss      = ?,
          take_profit    = ?,
          fraction       = ?,
          risk_per_trade = ?,
          in_trade       = ?
        WHERE id = 1
    ''', (
        usd_balance,
        btc_balance,
        entry_price,
        stop_loss,
        take_profit,
        fraction,
        risk_per_trade,
        1 if in_trade else 0,
    ))
    conn.commit()
    conn.close()


def open_position(entry_price, stop_loss, take_profit):
    state = load_bot_state()
    state.update({
        "in_trade":    True,
        "entry_price": entry_price,
        "stop_loss":   stop_loss,
        "take_profit": take_profit
    })
    save_bot_state(**state)


def close_position():
    state = load_bot_state()
    state["in_trade"] = False
    save_bot_state(**state)


def log_prediction(signal: str, probability: float, price: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (ts, signal, probability, price) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), signal, probability, price)
    )
    conn.commit()
    conn.close()


def log_trade(entry_price: float, exit_price: float, position_size: float, profit: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO executed_trades (ts, entry_price, exit_price, position_size, profit) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), entry_price, exit_price, position_size, profit)
    )
    conn.commit()
    conn.close()

# Инициализация при импорте
init_bot_state_table()
init_logs_table()
