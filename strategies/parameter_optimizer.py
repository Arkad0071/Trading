#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматический оптимизатор параметров для торговых стратегий
Ищет лучшие комбинации параметров индикаторов для максимальной прибыльности
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from itertools import product
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """
    Автоматический оптимизатор параметров стратегий
    """
    
    def __init__(self, df: pd.DataFrame, strategy_func, backtester):
        """
        Инициализация оптимизатора
        
        Args:
            df: DataFrame с данными
            strategy_func: Функция стратегии для оптимизации
            backtester: Экземпляр бэктестера
        """
        self.df = df.copy()
        self.strategy_func = strategy_func
        self.backtester = backtester
        self.best_params = {}
        self.optimization_results = []
        
    def optimize_rsi_strategy(self, param_ranges: Dict[str, List] = None):
        """
        Оптимизирует параметры RSI стратегии
        """
        if param_ranges is None:
            param_ranges = {
                'rsi_period': [10, 14, 20, 30],
                'oversold': [20, 25, 30, 35],
                'overbought': [65, 70, 75, 80]
            }
            
        logger.info("Начинаю оптимизацию RSI стратегии...")
        
        best_score = -np.inf
        best_params = {}
        
        # Генерируем все комбинации параметров
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        total_combinations = np.prod([len(vals) for vals in param_values])
        logger.info(f"Тестирую {total_combinations} комбинаций параметров...")
        
        for i, combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, combination))
            
            try:
                # Применяем стратегию с текущими параметрами
                df_with_signals = self._apply_rsi_strategy(params)
                
                # Запускаем бэктест
                metrics = self.backtester.run_backtest(df_with_signals, strategy_name="RSI Optimized")
                
                if metrics and len(metrics) > 0:
                    # Оцениваем результат
                    score = self._calculate_strategy_score(metrics)
                    
                    # Сохраняем результат
                    result = {
                        'params': params.copy(),
                        'metrics': metrics,
                        'score': score
                    }
                    self.optimization_results.append(result)
                    
                    # Обновляем лучший результат
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        logger.info(f"Новый лучший результат: {best_params} (оценка: {score:.4f})")
                        
            except Exception as e:
                logger.warning(f"Ошибка при тестировании параметров {params}: {e}")
                continue
                
            # Прогресс
            if (i + 1) % 100 == 0:
                logger.info(f"Прогресс: {i + 1}/{total_combinations} ({((i + 1) / total_combinations * 100):.1f}%)")
        
        self.best_params = best_params
        logger.info(f"Оптимизация RSI завершена. Лучшие параметры: {best_params}")
        return best_params
        
    def optimize_macd_strategy(self, param_ranges: Dict[str, List] = None):
        """
        Оптимизирует параметры MACD стратегии
        """
        if param_ranges is None:
            param_ranges = {
                'fast_period': [8, 12, 16, 20],
                'slow_period': [20, 24, 26, 30],
                'signal_period': [7, 9, 11, 13]
            }
            
        logger.info("Начинаю оптимизацию MACD стратегии...")
        
        best_score = -np.inf
        best_params = {}
        
        # Генерируем все комбинации параметров
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        total_combinations = np.prod([len(vals) for vals in param_values])
        logger.info(f"Тестирую {total_combinations} комбинаций параметров...")
        
        for i, combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, combination))
            
            try:
                # Применяем стратегию с текущими параметрами
                df_with_signals = self._apply_macd_strategy(params)
                
                # Запускаем бэктест
                metrics = self.backtester.run_backtest(df_with_signals, strategy_name="MACD Optimized")
                
                if metrics and len(metrics) > 0:
                    # Оцениваем результат
                    score = self._calculate_strategy_score(metrics)
                    
                    # Сохраняем результат
                    result = {
                        'params': params.copy(),
                        'metrics': metrics,
                        'score': score
                    }
                    self.optimization_results.append(result)
                    
                    # Обновляем лучший результат
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        logger.info(f"Новый лучший результат: {best_params} (оценка: {score:.4f})")
                        
            except Exception as e:
                logger.warning(f"Ошибка при тестировании параметров {params}: {e}")
                continue
                
            # Прогресс
            if (i + 1) % 100 == 0:
                logger.info(f"Прогресс: {i + 1}/{total_combinations} ({((i + 1) / total_combinations * 100):.1f}%)")
        
        self.best_params = best_params
        logger.info(f"Оптимизация MACD завершена. Лучшие параметры: {best_params}")
        return best_params
        
    def optimize_composite_strategy(self, strategy_weights: Dict[str, float] = None):
        """
        Оптимизирует веса композитной стратегии
        """
        if strategy_weights is None:
            # Тестируем разные комбинации весов
            weight_combinations = [
                {'RSI': 0.3, 'MACD': 0.3, 'Bollinger': 0.4},
                {'RSI': 0.4, 'MACD': 0.3, 'Bollinger': 0.3},
                {'RSI': 0.3, 'MACD': 0.4, 'Bollinger': 0.3},
                {'RSI': 0.5, 'MACD': 0.3, 'Bollinger': 0.2},
                {'RSI': 0.2, 'MACD': 0.5, 'Bollinger': 0.3},
                {'RSI': 0.2, 'MACD': 0.3, 'Bollinger': 0.5}
            ]
        else:
            weight_combinations = [strategy_weights]
            
        logger.info("Начинаю оптимизацию композитной стратегии...")
        
        best_score = -np.inf
        best_weights = {}
        
        for weights in weight_combinations:
            try:
                # Применяем композитную стратегию с текущими весами
                df_with_signals = self._apply_composite_strategy(weights)
                
                # Запускаем бэктест
                metrics = self.backtester.run_backtest(df_with_signals, strategy_name="Composite Optimized")
                
                if metrics and len(metrics) > 0:
                    # Оцениваем результат
                    score = self._calculate_strategy_score(metrics)
                    
                    # Сохраняем результат
                    result = {
                        'weights': weights.copy(),
                        'metrics': metrics,
                        'score': score
                    }
                    self.optimization_results.append(result)
                    
                    # Обновляем лучший результат
                    if score > best_score:
                        best_score = score
                        best_weights = weights.copy()
                        logger.info(f"Новый лучший результат: {best_weights} (оценка: {score:.4f})")
                        
            except Exception as e:
                logger.warning(f"Ошибка при тестировании весов {weights}: {e}")
                continue
        
        self.best_params = best_weights
        logger.info(f"Оптимизация композитной стратегии завершена. Лучшие веса: {best_weights}")
        return best_weights
        
    def _apply_rsi_strategy(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Применяет RSI стратегию с заданными параметрами
        """
        df = self.df.copy()
        
        # Рассчитываем RSI с заданными параметрами
        rsi_period = params.get('rsi_period', 14)
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        
        # Простой расчет RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Создаем сигналы
        df['signal'] = 'HOLD'
        df.loc[df['RSI'] < oversold, 'signal'] = 'BUY'
        df.loc[df['RSI'] > overbought, 'signal'] = 'SELL'
        
        return df
        
    def _apply_macd_strategy(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Применяет MACD стратегию с заданными параметрами
        """
        df = self.df.copy()
        
        # Параметры MACD
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        
        # Рассчитываем MACD
        ema_fast = df['close'].ewm(span=fast_period).mean()
        ema_slow = df['close'].ewm(span=slow_period).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=signal_period).mean()
        
        # Создаем сигналы
        df['signal'] = 'HOLD'
        df.loc[df['MACD'] > df['MACD_signal'], 'signal'] = 'BUY'
        df.loc[df['MACD'] < df['MACD_signal'], 'signal'] = 'SELL'
        
        return df
        
    def _apply_composite_strategy(self, weights: Dict[str, float]) -> pd.DataFrame:
        """
        Применяет композитную стратегию с заданными весами
        """
        df = self.df.copy()
        
        # Применяем базовые стратегии
        rsi_df = self._apply_rsi_strategy({'rsi_period': 14, 'oversold': 30, 'overbought': 70})
        macd_df = self._apply_macd_strategy({'fast_period': 12, 'slow_period': 26, 'signal_period': 9})
        
        # Рассчитываем взвешенные голоса
        df['rsi_vote'] = np.where(rsi_df['signal'] == 'BUY', weights.get('RSI', 0.33), 0)
        df['macd_vote'] = np.where(macd_df['signal'] == 'BUY', weights.get('MACD', 0.33), 0)
        
        # Финальный сигнал
        df['signal'] = 'HOLD'
        total_votes = df['rsi_vote'] + df['macd_vote']
        
        # BUY если сумма голосов > 0.5
        df.loc[total_votes > 0.5, 'signal'] = 'BUY'
        
        return df
        
    def _calculate_strategy_score(self, metrics: Dict[str, Any]) -> float:
        """
        Рассчитывает общую оценку стратегии на основе метрик
        """
        if not metrics:
            return -np.inf
            
        # Основные метрики для оценки
        total_return = metrics.get('Total Return (%)', 0)
        win_rate = metrics.get('Win Rate (%)', 0)
        profit_factor = metrics.get('Profit Factor', 0)
        max_drawdown = abs(metrics.get('Max Drawdown (%)', 0))
        sharpe_ratio = metrics.get('Sharpe Ratio', 0)
        
        # Взвешенная оценка
        score = (
            total_return * 0.3 +           # 30% - общая доходность
            win_rate * 0.2 +               # 20% - процент выигрышных сделок
            (profit_factor - 1) * 50 +     # 20% - profit factor (нормализованный)
            (100 - max_drawdown) * 0.2 +   # 20% - контроль просадки
            sharpe_ratio * 10              # 10% - коэффициент Шарпа
        )
        
        return score
        
    def get_optimization_summary(self) -> pd.DataFrame:
        """
        Возвращает сводку результатов оптимизации
        """
        if not self.optimization_results:
            return pd.DataFrame()
            
        summary_data = []
        for result in self.optimization_results:
            if 'params' in result:
                summary_data.append({
                    'Parameters': str(result['params']),
                    'Total Return (%)': result['metrics'].get('Total Return (%)', 0),
                    'Win Rate (%)': result['metrics'].get('Win Rate (%)', 0),
                    'Profit Factor': result['metrics'].get('Profit Factor', 0),
                    'Max Drawdown (%)': result['metrics'].get('Max Drawdown (%)', 0),
                    'Score': result['score']
                })
            elif 'weights' in result:
                summary_data.append({
                    'Parameters': f"Weights: {result['weights']}",
                    'Total Return (%)': result['metrics'].get('Total Return (%)', 0),
                    'Win Rate (%)': result['metrics'].get('Win Rate (%)', 0),
                    'Profit Factor': result['metrics'].get('Profit Factor', 0),
                    'Max Drawdown (%)': result['metrics'].get('Max Drawdown (%)', 0),
                    'Score': result['score']
                })
                
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Score', ascending=False)
        
        return summary_df
