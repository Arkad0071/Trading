# indicators/enhanced_indicators.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_span_b_period=52, displacement=26):
    """
    Рассчитывает индикатор Ichimoku Cloud
    
    Args:
        df: DataFrame с данными OHLCV
        tenkan_period: Период для Tenkan-sen (Conversion Line)
        kijun_period: Период для Kijun-sen (Base Line)
        senkou_span_b_period: Период для Senkou Span B
        displacement: Смещение для Senkou Span A и B
    """
    # Tenkan-sen (Conversion Line)
    high_tenkan = df['high'].rolling(window=tenkan_period).max()
    low_tenkan = df['low'].rolling(window=tenkan_period).min()
    df['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line)
    high_kijun = df['high'].rolling(window=kijun_period).max()
    low_kijun = df['low'].rolling(window=kijun_period).min()
    df['kijun_sen'] = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B)
    high_senkou_b = df['high'].rolling(window=senkou_span_b_period).max()
    low_senkou_b = df['low'].rolling(window=senkou_span_b_period).min()
    df['senkou_span_b'] = ((high_senkou_b + low_senkou_b) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span)
    df['chikou_span'] = df['close'].shift(-displacement)
    
    logger.info("Ichimoku Cloud рассчитан")
    return df

def calculate_williams_r(df, period=14):
    """
    Рассчитывает Williams %R
    
    Args:
        df: DataFrame с данными OHLCV
        period: Период для расчета
    """
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    
    df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    
    logger.info("Williams %R рассчитан")
    return df

def calculate_money_flow_index(df, period=14):
    """
    Рассчитывает Money Flow Index (MFI)
    
    Args:
        df: DataFrame с данными OHLCV
        period: Период для расчета
    """
    # Типичная цена
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Денежный поток
    money_flow = typical_price * df['volume']
    
    # Положительный и отрицательный денежный поток
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    
    # Скользящие средние
    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
    
    # MFI
    money_ratio = positive_mf / negative_mf
    df['mfi'] = 100 - (100 / (1 + money_ratio))
    
    logger.info("Money Flow Index рассчитан")
    return df

def calculate_average_directional_index(df, period=14):
    """
    Рассчитывает Average Directional Index (ADX)
    
    Args:
        df: DataFrame с данными OHLCV
        period: Период для расчета
    """
    # True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Сглаживание
    tr_smooth = true_range.rolling(window=period).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
    
    # Дирекциональные индикаторы
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(window=period).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    logger.info("Average Directional Index рассчитан")
    return df

def calculate_parabolic_sar(df, acceleration=0.02, maximum=0.2):
    """
    Рассчитывает Parabolic SAR
    
    Args:
        df: DataFrame с данными OHLCV
        acceleration: Фактор ускорения
        maximum: Максимальный фактор ускорения
    """
    sar = np.zeros(len(df))
    ep = np.zeros(len(df))
    af = np.zeros(len(df))
    
    # Инициализация
    sar[0] = df['low'].iloc[0]
    ep[0] = df['high'].iloc[0]
    af[0] = acceleration
    
    long_position = True
    
    for i in range(1, len(df)):
        if long_position:
            # Длинная позиция
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            # Проверка на разворот
            if df['low'].iloc[i] < sar[i]:
                long_position = False
                sar[i] = ep[i-1]
                ep[i] = df['low'].iloc[i]
                af[i] = acceleration
            else:
                # Обновление экстремума
                if df['high'].iloc[i] > ep[i-1]:
                    ep[i] = df['high'].iloc[i]
                    af[i] = min(af[i-1] + acceleration, maximum)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:
            # Короткая позиция
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            # Проверка на разворот
            if df['high'].iloc[i] > sar[i]:
                long_position = True
                sar[i] = ep[i-1]
                ep[i] = df['high'].iloc[i]
                af[i] = acceleration
            else:
                # Обновление экстремума
                if df['low'].iloc[i] < ep[i-1]:
                    ep[i] = df['low'].iloc[i]
                    af[i] = min(af[i-1] + acceleration, maximum)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    df['parabolic_sar'] = sar
    df['sar_long'] = np.where(df['close'] > sar, True, False)
    
    logger.info("Parabolic SAR рассчитан")
    return df

def calculate_fibonacci_retracements(df, lookback_period=20):
    """
    Рассчитывает уровни Fibonacci Retracements
    
    Args:
        df: DataFrame с данными OHLCV
        lookback_period: Период для поиска максимума/минимума
    """
    # Находим локальные максимумы и минимумы
    df['local_high'] = df['high'].rolling(window=lookback_period, center=True).max()
    df['local_low'] = df['low'].rolling(window=lookback_period, center=True).min()
    
    # Уровни Fibonacci
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    
    for level in fib_levels:
        if level == 0:
            df[f'fib_{int(level*1000)}'] = df['local_low']
        elif level == 1:
            df[f'fib_{int(level*1000)}'] = df['local_high']
        else:
            df[f'fib_{int(level*1000)}'] = df['local_low'] + level * (df['local_high'] - df['local_low'])
    
    logger.info("Fibonacci Retracements рассчитаны")
    return df

def calculate_volume_profile(df, period=20):
    """
    Рассчитывает Volume Profile
    
    Args:
        df: DataFrame с данными OHLCV
        period: Период для анализа объема
    """
    # Нормализованная цена
    price_range = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
    normalized_price = (df['close'] - df['low'].rolling(window=period).min()) / price_range
    
    # Volume Profile
    df['volume_profile'] = normalized_price * df['volume']
    
    # Скользящее среднее объема
    df['volume_sma'] = df['volume'].rolling(window=period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    logger.info("Volume Profile рассчитан")
    return df

def calculate_order_flow_indicators(df, period=14):
    """
    Рассчитывает индикаторы Order Flow
    
    Args:
        df: DataFrame с данными OHLCV
        period: Период для расчета
    """
    # Buy/Sell Pressure
    buy_pressure = np.where(df['close'] > df['open'], df['volume'], 0)
    sell_pressure = np.where(df['close'] < df['open'], df['volume'], 0)
    
    df['buy_pressure'] = pd.Series(buy_pressure).rolling(window=period).sum()
    df['sell_pressure'] = pd.Series(sell_pressure).rolling(window=period).sum()
    
    # Buy/Sell Ratio
    df['buy_sell_ratio'] = df['buy_pressure'] / (df['buy_pressure'] + df['sell_pressure'])
    
    # Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    # Price vs VWAP
    df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap'] * 100
    
    logger.info("Order Flow индикаторы рассчитаны")
    return df

def calculate_volatility_indicators(df, period=20):
    """
    Рассчитывает дополнительные индикаторы волатильности
    
    Args:
        df: DataFrame с данными OHLCV
        period: Период для расчета
    """
    # Historical Volatility
    returns = df['close'].pct_change()
    df['historical_volatility'] = returns.rolling(window=period).std() * np.sqrt(252) * 100
    
    # Parkinson Volatility (использует high/low)
    df['parkinson_volatility'] = np.sqrt(
        (1 / (4 * np.log(2))) * 
        ((np.log(df['high'] / df['low']) ** 2).rolling(window=period).mean()
    ) * np.sqrt(252) * 100
    
    # Garman-Klass Volatility - исправленная версия
    df['garman_klass_volatility'] = np.sqrt(
        (0.5 * (np.log(df['high'] / df['low']) ** 2) - 
         (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)
        ).rolling(window=period).mean()
    ) * np.sqrt(252) * 100
    
    # Volatility Ratio
    df['volatility_ratio'] = df['historical_volatility'] / df['historical_volatility'].rolling(window=period*2).mean()
    
    logger.info("Дополнительные индикаторы волатильности рассчитаны")
    return df

def calculate_market_regime_indicators(df, short_period=10, long_period=50):
    """
    Рассчитывает индикаторы рыночного режима
    
    Args:
        df: DataFrame с данными OHLCV
        short_period: Короткий период для тренда
        long_period: Длинный период для тренда
    """
    # Trend Strength
    df['trend_strength'] = abs(df['close'] - df['close'].shift(short_period)) / df['close'].shift(short_period) * 100
    
    # Market Regime Classification
    df['market_regime'] = np.where(
        (df['trend_strength'] > df['trend_strength'].rolling(window=long_period).mean()) & 
        (df['close'] > df['close'].rolling(window=long_period).mean()),
        'Strong Uptrend',
        np.where(
            (df['trend_strength'] > df['trend_strength'].rolling(window=long_period).mean()) & 
            (df['close'] < df['close'].rolling(window=long_period).mean()),
            'Strong Downtrend',
            np.where(
                df['trend_strength'] < df['trend_strength'].rolling(window=long_period).mean(),
                'Sideways/Low Volatility',
                'Weak Trend'
            )
        )
    )
    
    # Volatility Regime
    df['volatility_regime'] = np.where(
        df['historical_volatility'] > df['historical_volatility'].rolling(window=long_period).quantile(0.75),
        'High Volatility',
        np.where(
            df['historical_volatility'] < df['historical_volatility'].rolling(window=long_period).quantile(0.25),
            'Low Volatility',
            'Medium Volatility'
        )
    )
    
    logger.info("Индикаторы рыночного режима рассчитаны")
    return df

def calculate_all_enhanced_indicators(df):
    """
    Рассчитывает все расширенные индикаторы
    
    Args:
        df: DataFrame с данными OHLCV
    """
    logger.info("Начинаю расчет всех расширенных индикаторов...")
    
    # Список функций для расчета
    indicator_functions = [
        calculate_ichimoku,
        calculate_williams_r,
        calculate_money_flow_index,
        calculate_average_directional_index,
        calculate_parabolic_sar,
        calculate_fibonacci_retracements,
        calculate_volume_profile,
        calculate_order_flow_indicators,
        calculate_volatility_indicators,
        calculate_market_regime_indicators
    ]
    
    # Применяем каждую функцию
    for func in indicator_functions:
        try:
            df = func(df)
            logger.info(f"Успешно применена функция: {func.__name__}")
        except Exception as e:
            logger.error(f"Ошибка при применении {func.__name__}: {str(e)}")
            continue
    
    logger.info("Все расширенные индикаторы рассчитаны")
    return df
