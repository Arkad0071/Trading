import pandas as pd
import numpy as np
from indicators.indicators import calculate_indicators


def test_calculate_indicators_adds_columns():
    data = {
        'open':  np.arange(1, 31),
        'high':  np.arange(2, 32),
        'low':   np.arange(0, 30),
        'close': np.arange(1, 31)
    }
    df = pd.DataFrame(data)
    result = calculate_indicators(df)
    expected_cols = [
        'RSI', 'MACD', 'MACD_signal', 'ATR', 'SMA_20', 'EMA_20',
        'BB_upper', 'BB_middle', 'BB_lower', 'STOCH_K', 'STOCH_D'
    ]
    for col in expected_cols:
        assert col in result.columns
