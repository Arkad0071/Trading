import pytest
import pandas as pd
from backtesting.backtesting import Backtester


def test_basic_buy_sell_no_commission():
    df = pd.DataFrame([
        {"open": 100, "high": 101, "low": 99, "close": 100, "signal": "BUY"},
        {"open": 101, "high": 102, "low": 100, "close": 101, "signal": "HOLD"},
        {"open": 102, "high": 103, "low": 99, "close": 101, "signal": "SELL"},
    ])
    bt = Backtester(initial_balance=10000, commission_rate=0)
    trades = bt.run_backtest(df)
    assert len(trades) == 1
    assert trades[0]["exit_type"] == "SELL"
    assert bt.balance == pytest.approx(10050)


def test_close_on_end_of_data():
    df = pd.DataFrame([
        {"open": 100, "high": 101, "low": 99, "close": 100, "signal": "BUY"},
        {"open": 101, "high": 102, "low": 100, "close": 100, "signal": "HOLD"},
    ])
    bt = Backtester(initial_balance=10000, commission_rate=0)
    trades = bt.run_backtest(df)
    assert len(trades) == 1
    assert trades[0]["exit_type"] == "EOD"
    assert bt.balance == pytest.approx(10000)


def test_commission_applied():
    df = pd.DataFrame([
        {"open": 100, "high": 101, "low": 99, "close": 100, "signal": "BUY"},
        {"open": 101, "high": 102, "low": 100, "close": 102, "signal": "SELL"},
    ])
    bt = Backtester(initial_balance=10000, commission_rate=0.001)
    trades = bt.run_backtest(df)
    assert len(trades) == 1
    profit = trades[0]["profit"]
    expected_commission = (100 * 50 + 102 * 50) * 0.001
    expected_profit = (102 - 100) * 50 - expected_commission
    assert profit == pytest.approx(expected_profit)
    assert bt.balance == pytest.approx(10000 + expected_profit)
