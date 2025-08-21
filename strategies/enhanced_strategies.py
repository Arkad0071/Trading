# strategies/enhanced_strategies.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def test_simple_strategy(df):
    """
    Очень простая тестовая стратегия для демонстрации
    
    Особенности:
    - Минимальные условия
    - Максимум сигналов для тестирования
    - Только для демонстрации работы системы
    """
    df = df.copy()
    
    # Простые сигналы
    df['signal'] = 'HOLD'
    
    # BUY: RSI < 50 (простое условие)
    buy_signal = df['RSI'] < 50
    
    # SELL: RSI > 50 (простое условие)
    sell_signal = df['RSI'] > 50
    
    df.loc[buy_signal, 'signal'] = 'BUY'
    df.loc[sell_signal, 'signal'] = 'SELL'
    
    logger.info("Тестовая простая стратегия применена")
    return df

def simple_rsi_macd_strategy(df):
    """
    Простая и эффективная стратегия на основе RSI + MACD
    
    Особенности:
    - Простые условия входа
    - Меньше фильтров = больше сигналов
    - Быстрая реакция на изменения
    """
    df = df.copy()
    
    # Базовые сигналы
    df['signal'] = 'HOLD'
    
    # BUY: RSI перепродан + MACD растет
    buy_signal = (
        (df['RSI'] < 35) &  # Перепроданность
        (df['MACD'] > df['MACD_signal']) &  # MACD выше сигнала
        (df['close'] > df['close'].shift(1))  # Цена растет
    )
    
    # SELL: RSI перекуплен + MACD падает
    sell_signal = (
        (df['RSI'] > 65) &  # Перекупленность
        (df['MACD'] < df['MACD_signal']) &  # MACD ниже сигнала
        (df['close'] < df['close'].shift(1))  # Цена падает
    )
    
    df.loc[buy_signal, 'signal'] = 'BUY'
    df.loc[sell_signal, 'signal'] = 'SELL'
    
    logger.info("Простая RSI+MACD стратегия применена")
    return df

def adaptive_momentum_strategy(df):
    """
    Адаптивная стратегия на основе моментума и волатильности
    
    Особенности:
    - Анализ тренда через EMA
    - Фильтрация по волатильности
    - Адаптивные уровни входа
    """
    df = df.copy()
    
    # Анализ тренда
    df['trend'] = np.where(df['EMA_20'] > df['EMA_50'], 'UP', 'DOWN')
    
    # Анализ волатильности
    df['volatility_rank'] = df['ATR'].rolling(20).rank(pct=True)
    
    # Сигналы с учетом тренда и волатильности
    df['signal'] = 'HOLD'
    
    # Сильные сигналы в тренде
    strong_buy = (
        (df['RSI'] < 30) & 
        (df['trend'] == 'UP') & 
        (df['volatility_rank'] > 0.7) &
        (df['close'] > df['EMA_20']) &
        (df['MACD'] > df['MACD_signal'])
    )
    
    strong_sell = (
        (df['RSI'] > 70) & 
        (df['trend'] == 'DOWN') & 
        (df['volatility_rank'] > 0.7) &
        (df['close'] < df['EMA_20']) &
        (df['MACD'] < df['MACD_signal'])
    )
    
    df.loc[strong_buy, 'signal'] = 'BUY'
    df.loc[strong_sell, 'signal'] = 'SELL'
    
    logger.info("Адаптивная momentum стратегия применена")
    return df

def multi_timeframe_strategy(df):
    """
    Мультитаймфреймовая стратегия
    
    Особенности:
    - Анализ на нескольких временных масштабах
    - Подтверждение сигналов на разных таймфреймах
    - Фильтрация ложных сигналов
    """
    df = df.copy()
    
    # Создаем "виртуальные" таймфреймы через агрегацию
    # 4H таймфрейм (группируем по 4 часа)
    df['close_4h'] = df['close'].rolling(window=4).mean()
    df['high_4h'] = df['high'].rolling(window=4).max()
    df['low_4h'] = df['low'].rolling(window=4).min()
    
    # 1D таймфрейм (группируем по 24 часа)
    df['close_1d'] = df['close'].rolling(window=24).mean()
    df['high_1d'] = df['high'].rolling(window=24).max()
    df['low_1d'] = df['low'].rolling(window=24).min()
    
    # Тренды на разных таймфреймах
    df['trend_1h'] = np.where(df['close'] > df['EMA_20'], 'UP', 'DOWN')
    df['trend_4h'] = np.where(df['close_4h'] > df['close_4h'].rolling(20).mean(), 'UP', 'DOWN')
    df['trend_1d'] = np.where(df['close_1d'] > df['close_1d'].rolling(20).mean(), 'UP', 'DOWN')
    
    # Сигналы с подтверждением на нескольких таймфреймах
    df['signal'] = 'HOLD'
    
    # BUY: все таймфреймы показывают восходящий тренд
    buy_signal = (
        (df['trend_1h'] == 'UP') & 
        (df['trend_4h'] == 'UP') & 
        (df['trend_1d'] == 'UP') &
        (df['RSI'] < 40) &
        (df['close'] > df['BB_lower'])
    )
    
    # SELL: все таймфреймы показывают нисходящий тренд
    sell_signal = (
        (df['trend_1h'] == 'DOWN') & 
        (df['trend_4h'] == 'DOWN') & 
        (df['trend_1d'] == 'DOWN') &
        (df['RSI'] > 60) &
        (df['close'] < df['BB_upper'])
    )
    
    df.loc[buy_signal, 'signal'] = 'BUY'
    df.loc[sell_signal, 'signal'] = 'SELL'
    
    logger.info("Мультитаймфреймовая стратегия применена")
    return df

def volume_confirmation_strategy(df):
    """
    Стратегия с подтверждением объема
    
    Особенности:
    - Анализ объема для подтверждения сигналов
    - Buy/Sell pressure анализ
    - VWAP как динамическая поддержка/сопротивление
    """
    df = df.copy()
    
    # Анализ объема
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Buy/Sell pressure
    df['buy_pressure'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['sell_pressure'] = np.where(df['close'] < df['open'], df['volume'], 0)
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    
    # Сигналы с подтверждением объема
    df['signal'] = 'HOLD'
    
    # BUY: цена выше VWAP + высокий объем + buy pressure
    buy_signal = (
        (df['close'] > df['vwap']) &
        (df['volume_ratio'] > 1.5) &
        (df['buy_pressure'] > df['sell_pressure']) &
        (df['RSI'] < 50) &
        (df['close'] > df['EMA_20'])
    )
    
    # SELL: цена ниже VWAP + высокий объем + sell pressure
    sell_signal = (
        (df['close'] < df['vwap']) &
        (df['volume_ratio'] > 1.5) &
        (df['sell_pressure'] > df['buy_pressure']) &
        (df['RSI'] > 50) &
        (df['close'] < df['EMA_20'])
    )
    
    df.loc[buy_signal, 'signal'] = 'BUY'
    df.loc[sell_signal, 'signal'] = 'SELL'
    
    logger.info("Стратегия с подтверждением объема применена")
    return df

def breakout_strategy(df):
    """
    Стратегия прорыва уровней
    
    Особенности:
    - Анализ уровней поддержки/сопротивления
    - Подтверждение прорыва объемом
    - Фильтрация ложных прорывов
    """
    df = df.copy()
    
    # Уровни поддержки и сопротивления
    df['resistance'] = df['high'].rolling(20).max()
    df['support'] = df['low'].rolling(20).min()
    
    # Bollinger Bands как дополнительные уровни
    df['bb_upper_resistance'] = df['BB_upper']
    df['bb_lower_support'] = df['BB_lower']
    
    # Прорыв сопротивления
    breakout_up = (
        (df['close'] > df['resistance'].shift(1)) &
        (df['volume'] > df['volume'].rolling(20).mean() * 1.5) &
        (df['RSI'] < 70) &
        (df['MACD'] > df['MACD_signal'])
    )
    
    # Прорыв поддержки
    breakout_down = (
        (df['close'] < df['support'].shift(1)) &
        (df['volume'] > df['volume'].rolling(20).mean() * 1.5) &
        (df['RSI'] > 30) &
        (df['MACD'] < df['MACD_signal'])
    )
    
    # Сигналы
    df['signal'] = 'HOLD'
    df.loc[breakout_up, 'signal'] = 'BUY'
    df.loc[breakout_down, 'signal'] = 'SELL'
    
    logger.info("Стратегия прорыва применена")
    return df

def mean_reversion_strategy(df):
    """
    Стратегия возврата к среднему
    
    Особенности:
    - Торговля в боковике
    - Использование Bollinger Bands
    - Фильтрация по тренду
    """
    df = df.copy()
    
    # Определение боковика (низкая волатильность + слабый тренд)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    is_sideways = (
        (df['price_range'] < df['price_range'].rolling(50).quantile(0.3)) &
        (df['trend_strength'] < df['trend_strength'].rolling(50).quantile(0.3))
    )
    
    # Сигналы возврата к среднему
    df['signal'] = 'HOLD'
    
    # BUY: цена у нижней границы Bollinger Bands в боковике
    mean_reversion_buy = (
        is_sideways &
        (df['close'] < df['BB_lower'] * 1.01) &
        (df['RSI'] < 35) &
        (df['close'] < df['EMA_20'])
    )
    
    # SELL: цена у верхней границы Bollinger Bands в боковике
    mean_reversion_sell = (
        is_sideways &
        (df['close'] > df['BB_upper'] * 0.99) &
        (df['RSI'] > 65) &
        (df['close'] > df['EMA_20'])
    )
    
    df.loc[mean_reversion_buy, 'signal'] = 'BUY'
    df.loc[mean_reversion_sell, 'signal'] = 'SELL'
    
    logger.info("Стратегия возврата к среднему применена")
    return df

def volatility_regime_strategy(df):
    """
    Стратегия, адаптирующаяся к режиму волатильности
    
    Особенности:
    - Разные параметры для разных режимов волатильности
    - Адаптивные стоп-лоссы
    - Оптимизация под текущие рыночные условия
    """
    df = df.copy()
    
    # Определение режима волатильности
    df['volatility_regime'] = np.where(
        df['ATR'] > df['ATR'].rolling(50).quantile(0.75),
        'High',
        np.where(
            df['ATR'] < df['ATR'].rolling(50).quantile(0.25),
            'Low',
            'Medium'
        )
    )
    
    # Адаптивные параметры RSI
    df['rsi_oversold'] = np.where(
        df['volatility_regime'] == 'High', 25,
        np.where(df['volatility_regime'] == 'Medium', 30, 35)
    )
    
    df['rsi_overbought'] = np.where(
        df['volatility_regime'] == 'High', 75,
        np.where(df['volatility_regime'] == 'Medium', 70, 65)
    )
    
    # Сигналы с адаптивными параметрами
    df['signal'] = 'HOLD'
    
    # BUY: адаптивный RSI + подтверждение тренда
    volatility_buy = (
        (df['RSI'] < df['rsi_oversold']) &
        (df['close'] > df['EMA_20']) &
        (df['MACD'] > df['MACD_signal']) &
        (df['volume'] > df['volume'].rolling(20).mean())
    )
    
    # SELL: адаптивный RSI + подтверждение тренда
    volatility_sell = (
        (df['RSI'] > df['rsi_overbought']) &
        (df['close'] < df['EMA_20']) &
        (df['MACD'] < df['MACD_signal']) &
        (df['volume'] > df['volume'].rolling(20).mean())
    )
    
    df.loc[volatility_buy, 'signal'] = 'BUY'
    df.loc[volatility_sell, 'signal'] = 'SELL'
    
    logger.info("Волатильностная стратегия применена")
    return df

def ichimoku_strategy(df):
    """
    Стратегия на основе Ichimoku Cloud
    
    Особенности:
    - Анализ тренда через Cloud
    - Сигналы пересечения линий
    - Фильтрация по положению цены относительно Cloud
    """
    df = df.copy()
    
    # Проверяем наличие Ichimoku индикаторов
    required_columns = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']
    if not all(col in df.columns for col in required_columns):
        logger.warning("Ichimoku индикаторы не найдены. Пропускаю стратегию.")
        df['signal'] = 'HOLD'
        return df
    
    # Положение цены относительно Cloud
    df['above_cloud'] = (
        (df['close'] > df['senkou_span_a']) & 
        (df['close'] > df['senkou_span_b'])
    )
    
    df['below_cloud'] = (
        (df['close'] < df['senkou_span_a']) & 
        (df['close'] < df['senkou_span_b'])
    )
    
    # Сигналы пересечения
    df['signal'] = 'HOLD'
    
    # BUY: Tenkan-sen пересекает Kijun-sen снизу вверх + цена выше Cloud
    ichimoku_buy = (
        (df['tenkan_sen'] > df['kijun_sen']) &
        (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1)) &
        (df['above_cloud']) &
        (df['close'] > df['EMA_20'])
    )
    
    # SELL: Tenkan-sen пересекает Kijun-sen сверху вниз + цена ниже Cloud
    ichimoku_sell = (
        (df['tenkan_sen'] < df['kijun_sen']) &
        (df['tenkan_sen'].shift(1) >= df['kijun_sen'].shift(1)) &
        (df['below_cloud']) &
        (df['close'] < df['EMA_20'])
    )
    
    df.loc[ichimoku_buy, 'signal'] = 'BUY'
    df.loc[ichimoku_sell, 'signal'] = 'SELL'
    
    logger.info("Ichimoku стратегия применена")
    return df

def composite_strategy(df, weights=None):
    """
    Композитная стратегия, объединяющая несколько подходов
    
    Особенности:
    - Взвешенное голосование стратегий
    - Фильтрация конфликтующих сигналов
    - Адаптивные веса на основе производительности
    """
    df = df.copy()
    
    # Список стратегий для комбинирования
    strategies = [
        adaptive_momentum_strategy,
        volume_confirmation_strategy,
        breakout_strategy,
        mean_reversion_strategy,
        volatility_regime_strategy
    ]
    
    # Веса по умолчанию (равные)
    if weights is None:
        weights = [1.0] * len(strategies)
    
    # Применяем каждую стратегию
    strategy_signals = []
    for strategy in strategies:
        try:
            temp_df = strategy(df.copy())
            strategy_signals.append(temp_df['signal'])
            logger.info(f"Стратегия {strategy.__name__} применена")
        except Exception as e:
            logger.error(f"Ошибка в стратегии {strategy.__name__}: {str(e)}")
            strategy_signals.append(pd.Series(['HOLD'] * len(df), index=df.index))
    
    # Создаем DataFrame с сигналами всех стратегий
    signals_df = pd.DataFrame(strategy_signals).T
    signals_df.columns = [f'strategy_{i}' for i in range(len(strategies))]
    
    # Подсчитываем взвешенные голоса
    df['buy_votes'] = 0
    df['sell_votes'] = 0
    
    for i, weight in enumerate(weights):
        df['buy_votes'] += np.where(signals_df[f'strategy_{i}'] == 'BUY', weight, 0)
        df['sell_votes'] += np.where(signals_df[f'strategy_{i}'] == 'SELL', weight, 0)
    
    # Финальный сигнал на основе большинства голосов
    df['signal'] = 'HOLD'
    
    # Минимальный порог для сигнала (например, 60% стратегий должны согласиться)
    min_threshold = sum(weights) * 0.6
    
    df.loc[df['buy_votes'] >= min_threshold, 'signal'] = 'BUY'
    df.loc[df['sell_votes'] >= min_threshold, 'signal'] = 'SELL'
    
    logger.info("Композитная стратегия применена")
    return df

def get_all_enhanced_strategies():
    """
    Возвращает список всех доступных стратегий
    """
    return {
        'simple_rsi_macd': simple_rsi_macd_strategy,  # Новая простая стратегия
        'adaptive_momentum': adaptive_momentum_strategy,
        'multi_timeframe': multi_timeframe_strategy,
        'volume_confirmation': volume_confirmation_strategy,
        'breakout': breakout_strategy,
        'mean_reversion': mean_reversion_strategy,
        'volatility_regime': volatility_regime_strategy,
        'ichimoku': ichimoku_strategy,
        'composite': composite_strategy
    }
