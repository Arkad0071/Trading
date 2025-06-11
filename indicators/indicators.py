# indicators/indicators.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_rsi(df, period=14):
    """
    Рассчитывает Relative Strength Index (RSI) для DataFrame.
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    logger.info("RSI рассчитан.")
    return df

def calculate_macd(df, span_short=12, span_long=26, span_signal=9):
    """
    Рассчитывает MACD и сигнальную линию.
    """
    ema_short = df['close'].ewm(span=span_short, adjust=False).mean()
    ema_long = df['close'].ewm(span=span_long, adjust=False).mean()
    df['MACD'] = ema_short - ema_long
    df['MACD_signal'] = df['MACD'].ewm(span=span_signal, adjust=False).mean()
    logger.info("MACD рассчитан.")
    return df

def calculate_atr(df, period=14):
    """
    Рассчитывает Average True Range (ATR) для оценки волатильности.
    """
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    df['ATR'] = tr.rolling(window=period).mean()
    logger.info("ATR рассчитан.")
    return df

def calculate_sma(df, period=20):
    """Adds Simple Moving Average (SMA)."""
    df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
    logger.info("SMA рассчитана.")
    return df

def calculate_ema(df, period=20):
    """Adds Exponential Moving Average (EMA)."""
    df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    logger.info("EMA рассчитана.")
    return df

def calculate_bollinger_bands(df, period=20, std_factor=2):
    """Adds Bollinger Bands (upper, middle, lower)."""
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    df['BB_middle'] = sma
    df['BB_upper'] = sma + std_factor * std
    df['BB_lower'] = sma - std_factor * std
    logger.info("Bollinger Bands рассчитаны.")
    return df

def calculate_stochastic(df, k_period=14, d_period=3):
    """Adds Stochastic Oscillator %K and %D."""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    df['STOCH_K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['STOCH_D'] = df['STOCH_K'].rolling(window=d_period).mean()
    logger.info("Stochastic рассчитан.")
    return df

def calculate_momentum(df, period=10):
    """Adds Momentum indicator."""
    df['MOM'] = df['close'] - df['close'].shift(period)
    logger.info("Momentum рассчитан.")
    return df

def calculate_roc(df, period=12):
    """Adds Rate of Change (ROC)."""
    df['ROC'] = df['close'].pct_change(periods=period) * 100
    logger.info("ROC рассчитан.")
    return df

def calculate_cci(df, period=20):
    """Adds Commodity Channel Index (CCI)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df['CCI'] = (tp - sma) / (0.015 * mad)
    logger.info("CCI рассчитан.")
    return df

def calculate_obv(df):
    """Adds On-Balance Volume (OBV)."""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV'] = obv
    logger.info("OBV рассчитан.")
    return df

def calculate_williams_r(df, period=14):
    """Adds Williams %R (WILLR)."""
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    df['WILLR'] = -100 * (highest_high - df['close']) / denom
    logger.info("Williams %R рассчитан.")
    return df

def calculate_indicators(df):
    """
    Запускает расчёт всех индикаторов и удаляет строки с NaN.
    """
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_atr(df)
    df = calculate_sma(df)
    df = calculate_ema(df)
    df = calculate_bollinger_bands(df)
    df = calculate_stochastic(df)
    df = calculate_momentum(df)
    df = calculate_roc(df)
    df = calculate_cci(df)
    df = calculate_obv(df)
    df = calculate_williams_r(df)
    df.dropna(inplace=True)
    logger.info("Все индикаторы рассчитаны и начальные NaN значения удалены.")
    return df
