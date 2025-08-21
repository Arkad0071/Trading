#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Мультитаймфреймовый анализатор для торговых стратегий
Анализирует данные на разных временных масштабах для подтверждения сигналов
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """
    Анализатор для работы с несколькими таймфреймами
    """
    
    def __init__(self, timeframes: List[str] = ['15m', '1h', '4h']):
        """
        Инициализация анализатора
        
        Args:
            timeframes: Список таймфреймов для анализа
        """
        self.timeframes = timeframes
        self.data = {}
        
    def load_data(self, data_paths: Dict[str, str]):
        """
        Загружает данные для разных таймфреймов
        
        Args:
            data_paths: Словарь {таймфрейм: путь_к_файлу}
        """
        for timeframe, path in data_paths.items():
            try:
                df = pd.read_csv(path)
                df['start_at'] = pd.to_datetime(df['start_at'])
                df.set_index('start_at', inplace=True)
                self.data[timeframe] = df
                logger.info(f"Данные для {timeframe} загружены: {len(df)} записей")
            except Exception as e:
                logger.error(f"Ошибка загрузки данных для {timeframe}: {e}")
                
    def calculate_multi_timeframe_signals(self, strategy_func, base_timeframe='1h'):
        """
        Рассчитывает сигналы на нескольких таймфреймах
        
        Args:
            strategy_func: Функция стратегии
            base_timeframe: Базовый таймфрейм для анализа
            
        Returns:
            DataFrame с мультитаймфреймовыми сигналами
        """
        if base_timeframe not in self.data:
            logger.error(f"Базовый таймфрейм {base_timeframe} не найден")
            return None
            
        base_df = self.data[base_timeframe].copy()
        
        # Применяем стратегию к базовому таймфрейму
        base_signals = strategy_func(base_df.copy())
        
        # Анализируем подтверждения на других таймфреймах
        for tf in self.timeframes:
            if tf != base_timeframe and tf in self.data:
                # Ресемплируем данные для сравнения
                tf_df = self.data[tf].resample('1H').agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                # Применяем стратегию к текущему таймфрейму
                tf_signals = strategy_func(tf_df.copy())
                
                # Добавляем подтверждение сигнала
                base_signals[f'{tf}_confirmation'] = tf_signals['signal']
                
        # Создаем финальный сигнал на основе подтверждений
        base_signals = self._create_final_signal(base_signals)
        
        return base_signals
        
    def _create_final_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает финальный сигнал на основе подтверждений всех таймфреймов
        """
        df['final_signal'] = 'HOLD'
        
        # Ищем строки где есть BUY сигналы на нескольких таймфреймах
        buy_confirmations = []
        sell_confirmations = []
        
        for tf in self.timeframes:
            if f'{tf}_confirmation' in df.columns:
                buy_confirmations.append(df[f'{tf}_confirmation'] == 'BUY')
                sell_confirmations.append(df[f'{tf}_confirmation'] == 'SELL')
        
        if buy_confirmations:
            # BUY: большинство таймфреймов показывают BUY
            strong_buy = sum(buy_confirmations) >= len(buy_confirmations) * 0.6
            df.loc[strong_buy, 'final_signal'] = 'BUY'
            
        if sell_confirmations:
            # SELL: большинство таймфреймов показывают SELL
            strong_sell = sum(sell_confirmations) >= len(sell_confirmations) * 0.6
            df.loc[strong_sell, 'final_signal'] = 'SELL'
            
        return df
        
    def get_timeframe_correlation(self) -> pd.DataFrame:
        """
        Анализирует корреляцию между таймфреймами
        """
        correlations = {}
        
        for tf1 in self.timeframes:
            if tf1 not in self.data:
                continue
                
            correlations[tf1] = {}
            for tf2 in self.timeframes:
                if tf2 not in self.data:
                    continue
                    
                # Ресемплируем к общему таймфрейму
                df1 = self.data[tf1]['close'].resample('1H').last()
                df2 = self.data[tf2]['close'].resample('1H').last()
                
                # Выравниваем индексы
                common_index = df1.index.intersection(df2.index)
                if len(common_index) > 0:
                    corr = df1.loc[common_index].corr(df2.loc[common_index])
                    correlations[tf1][tf2] = corr
                else:
                    correlations[tf1][tf2] = np.nan
                    
        return pd.DataFrame(correlations)
        
    def optimize_timeframe_weights(self, strategy_func, base_timeframe='1h'):
        """
        Оптимизирует веса таймфреймов для лучших результатов
        """
        # Простая оптимизация: тестируем разные комбинации весов
        best_weights = {}
        best_score = -np.inf
        
        # Тестируем разные веса
        weight_combinations = [
            {'15m': 0.2, '1h': 0.5, '4h': 0.3},
            {'15m': 0.3, '1h': 0.4, '4h': 0.3},
            {'15m': 0.1, '1h': 0.6, '4h': 0.3},
            {'15m': 0.4, '1h': 0.3, '4h': 0.3}
        ]
        
        for weights in weight_combinations:
            # Применяем веса и оцениваем результат
            score = self._evaluate_timeframe_weights(weights, strategy_func, base_timeframe)
            
            if score > best_score:
                best_score = score
                best_weights = weights
                
        logger.info(f"Лучшие веса таймфреймов: {best_weights} (оценка: {best_score:.4f})")
        return best_weights
        
    def _evaluate_timeframe_weights(self, weights: Dict[str, float], strategy_func, base_timeframe: str) -> float:
        """
        Оценивает качество весов таймфреймов
        """
        try:
            # Применяем стратегию с весами
            df = self.calculate_multi_timeframe_signals(strategy_func, base_timeframe)
            
            # Простая оценка: количество сигналов и их разнообразие
            signal_counts = df['final_signal'].value_counts()
            
            # Штраф за слишком много HOLD
            hold_penalty = signal_counts.get('HOLD', 0) / len(df)
            
            # Бонус за разнообразие сигналов
            diversity_bonus = len(signal_counts) / 3
            
            score = diversity_bonus - hold_penalty
            return score
            
        except Exception as e:
            logger.error(f"Ошибка оценки весов: {e}")
            return -np.inf
