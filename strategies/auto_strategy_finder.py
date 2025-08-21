#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматический поиск лучших торговых стратегий
Система сама находит оптимальные комбинации индикаторов и параметров
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from itertools import combinations, product
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AutoStrategyFinder:
    """
    Автоматический поиск и создание торговых стратегий
    """
    
    def __init__(self, backtester, min_trades=10, min_win_rate=0.4):
        """
        Инициализация поискового алгоритма
        
        Args:
            backtester: Экземпляр бэктестера
            min_trades: Минимальное количество сделок для валидности стратегии
            min_win_rate: Минимальный винрейт для рассмотрения стратегии
        """
        self.backtester = backtester
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.best_strategies = []
        self.search_results = []
        
        # Доступные индикаторы для комбинирования
        self.available_indicators = {
            'RSI': {
                'periods': [10, 14, 20, 30],
                'oversold': [20, 25, 30, 35],
                'overbought': [65, 70, 75, 80]
            },
            'MACD': {
                'fast': [8, 12, 16, 20],
                'slow': [20, 24, 26, 30],
                'signal': [7, 9, 11, 13]
            },
            'SMA': {
                'periods': [10, 20, 50, 100, 200]
            },
            'EMA': {
                'periods': [10, 20, 50, 100]
            },
            'BB': {
                'periods': [15, 20, 25],
                'std_dev': [1.5, 2.0, 2.5]
            },
            'ATR': {
                'periods': [10, 14, 20]
            },
            'Volume': {
                'periods': [10, 20, 30],
                'thresholds': [1.2, 1.5, 2.0]
            }
        }
        
        # Операторы для создания условий
        self.operators = ['>', '<', '>=', '<=', '==']
        self.logical_operators = ['AND', 'OR']
        
    def generate_single_indicator_strategies(self, df: pd.DataFrame) -> List[Dict]:
        """
        Генерирует стратегии на основе одного индикатора
        """
        strategies = []
        
        logger.info("Генерация стратегий на основе одного индикатора...")
        
        # RSI стратегии
        for period in self.available_indicators['RSI']['periods']:
            for oversold in self.available_indicators['RSI']['oversold']:
                for overbought in self.available_indicators['RSI']['overbought']:
                    strategy = {
                        'name': f'RSI_{period}_{oversold}_{overbought}',
                        'type': 'single_indicator',
                        'indicator': 'RSI',
                        'params': {
                            'period': period,
                            'oversold': oversold,
                            'overbought': overbought
                        },
                        'buy_condition': f'RSI < {oversold}',
                        'sell_condition': f'RSI > {overbought}'
                    }
                    strategies.append(strategy)
        
        # MACD стратегии
        for fast in self.available_indicators['MACD']['fast']:
            for slow in self.available_indicators['MACD']['slow']:
                for signal in self.available_indicators['MACD']['signal']:
                    if fast < slow:  # Логичное условие
                        strategy = {
                            'name': f'MACD_{fast}_{slow}_{signal}',
                            'type': 'single_indicator',
                            'indicator': 'MACD',
                            'params': {
                                'fast': fast,
                                'slow': slow,
                                'signal': signal
                            },
                            'buy_condition': 'MACD > MACD_signal',
                            'sell_condition': 'MACD < MACD_signal'
                        }
                        strategies.append(strategy)
        
        # SMA Cross стратегии
        sma_periods = self.available_indicators['SMA']['periods']
        for short_period, long_period in combinations(sma_periods, 2):
            if short_period < long_period:
                strategy = {
                    'name': f'SMA_Cross_{short_period}_{long_period}',
                    'type': 'single_indicator',
                    'indicator': 'SMA_Cross',
                    'params': {
                        'short_period': short_period,
                        'long_period': long_period
                    },
                    'buy_condition': f'SMA_{short_period} > SMA_{long_period}',
                    'sell_condition': f'SMA_{short_period} < SMA_{long_period}'
                }
                strategies.append(strategy)
        
        logger.info(f"Сгенерировано {len(strategies)} стратегий на основе одного индикатора")
        return strategies
    
    def generate_multi_indicator_strategies(self, df: pd.DataFrame) -> List[Dict]:
        """
        Генерирует стратегии на основе комбинаций индикаторов
        """
        strategies = []
        
        logger.info("Генерация стратегий на основе комбинаций индикаторов...")
        
        # RSI + MACD комбинации
        for rsi_period in [14, 20]:
            for rsi_oversold in [25, 30]:
                for rsi_overbought in [70, 75]:
                    for macd_fast in [12, 16]:
                        for macd_slow in [24, 26]:
                            if macd_fast < macd_slow:
                                strategy = {
                                    'name': f'RSI_MACD_{rsi_period}_{macd_fast}_{macd_slow}',
                                    'type': 'multi_indicator',
                                    'indicators': ['RSI', 'MACD'],
                                    'params': {
                                        'rsi_period': rsi_period,
                                        'rsi_oversold': rsi_oversold,
                                        'rsi_overbought': rsi_overbought,
                                        'macd_fast': macd_fast,
                                        'macd_slow': macd_slow,
                                        'macd_signal': 9
                                    },
                                    'buy_condition': f'(RSI < {rsi_oversold}) AND (MACD > MACD_signal)',
                                    'sell_condition': f'(RSI > {rsi_overbought}) OR (MACD < MACD_signal)'
                                }
                                strategies.append(strategy)
        
        # RSI + Bollinger Bands
        for rsi_period in [14, 20]:
            for bb_period in [20, 25]:
                for bb_std in [2.0, 2.5]:
                    strategy = {
                        'name': f'RSI_BB_{rsi_period}_{bb_period}_{bb_std}',
                        'type': 'multi_indicator',
                        'indicators': ['RSI', 'BB'],
                        'params': {
                            'rsi_period': rsi_period,
                            'bb_period': bb_period,
                            'bb_std': bb_std
                        },
                        'buy_condition': f'(RSI < 30) AND (close < BB_lower)',
                        'sell_condition': f'(RSI > 70) OR (close > BB_upper)'
                    }
                    strategies.append(strategy)
        
        # Triple Moving Average
        for short in [10, 20]:
            for medium in [50, 100]:
                for long in [100, 200]:
                    if short < medium < long:
                        strategy = {
                            'name': f'Triple_MA_{short}_{medium}_{long}',
                            'type': 'multi_indicator',
                            'indicators': ['SMA'],
                            'params': {
                                'short': short,
                                'medium': medium,
                                'long': long
                            },
                            'buy_condition': f'(SMA_{short} > SMA_{medium}) AND (SMA_{medium} > SMA_{long})',
                            'sell_condition': f'(SMA_{short} < SMA_{medium}) OR (SMA_{medium} < SMA_{long})'
                        }
                        strategies.append(strategy)
        
        # Volume + Price strategies
        for vol_period in [20, 30]:
            for vol_threshold in [1.5, 2.0]:
                for sma_period in [20, 50]:
                    strategy = {
                        'name': f'Volume_Breakout_{vol_period}_{vol_threshold}_{sma_period}',
                        'type': 'multi_indicator',
                        'indicators': ['Volume', 'SMA'],
                        'params': {
                            'vol_period': vol_period,
                            'vol_threshold': vol_threshold,
                            'sma_period': sma_period
                        },
                        'buy_condition': f'(volume > {vol_threshold} * volume_sma_{vol_period}) AND (close > SMA_{sma_period})',
                        'sell_condition': f'close < SMA_{sma_period}'
                    }
                    strategies.append(strategy)
        
        logger.info(f"Сгенерировано {len(strategies)} мульти-индикаторных стратегий")
        return strategies
    
    def apply_strategy_conditions(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """
        Применяет условия стратегии к данным
        """
        df_signals = df.copy()
        df_signals['signal'] = 'HOLD'
        
        try:
            # Подготавливаем нужные индикаторы на основе параметров стратегии
            df_signals = self._prepare_strategy_indicators(df_signals, strategy)
            
            # Применяем условия покупки и продажи
            buy_condition = self._parse_condition(strategy['buy_condition'], df_signals)
            sell_condition = self._parse_condition(strategy['sell_condition'], df_signals)
            
            if buy_condition is not None:
                df_signals.loc[buy_condition, 'signal'] = 'BUY'
            
            if sell_condition is not None:
                df_signals.loc[sell_condition, 'signal'] = 'SELL'
                
        except Exception as e:
            logger.warning(f"Ошибка применения стратегии {strategy['name']}: {e}")
            # Возвращаем пустую стратегию
            df_signals['signal'] = 'HOLD'
        
        return df_signals
    
    def _prepare_strategy_indicators(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """
        Подготавливает индикаторы для стратегии
        """
        params = strategy.get('params', {})
        
        # RSI
        if 'rsi_period' in params:
            period = params['rsi_period']
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        if 'macd_fast' in params and 'macd_slow' in params:
            fast = params['macd_fast']
            slow = params['macd_slow']
            signal_period = params.get('macd_signal', 9)
            
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            df['MACD'] = ema_fast - ema_slow
            df['MACD_signal'] = df['MACD'].ewm(span=signal_period).mean()
        
        # SMA
        for key, value in params.items():
            if 'period' in key or key in ['short', 'medium', 'long']:
                period = value
                df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        
        # Bollinger Bands
        if 'bb_period' in params:
            period = params['bb_period']
            std_dev = params.get('bb_std', 2.0)
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df['BB_upper'] = sma + (std * std_dev)
            df['BB_lower'] = sma - (std * std_dev)
            df['BB_middle'] = sma
        
        # Volume indicators
        if 'vol_period' in params:
            period = params['vol_period']
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        
        return df
    
    def _parse_condition(self, condition: str, df: pd.DataFrame):
        """
        Парсит строковое условие в pandas condition
        """
        try:
            # Заменяем названия колонок на df['название']
            for col in df.columns:
                if col in condition:
                    condition = condition.replace(col, f"df['{col}']")
            
            # Заменяем логические операторы
            condition = condition.replace(' AND ', ' & ')
            condition = condition.replace(' OR ', ' | ')
            
            # Выполняем условие
            return eval(condition)
            
        except Exception as e:
            logger.warning(f"Ошибка парсинга условия '{condition}': {e}")
            return None
    
    def test_strategy(self, df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
        """
        Тестирует отдельную стратегию
        """
        try:
            # Применяем стратегию
            df_with_signals = self.apply_strategy_conditions(df, strategy)
            
            # Запускаем бэктест
            metrics = self.backtester.run_backtest(df_with_signals, strategy_name=strategy['name'])
            
            if not metrics or len(metrics) == 0:
                return None
            
            # Проверяем минимальные требования
            total_trades = metrics.get('Total Trades', 0)
            win_rate = metrics.get('Win Rate (%)', 0) / 100
            
            if total_trades < self.min_trades or win_rate < self.min_win_rate:
                return None
            
            # Добавляем стратегию в результаты
            result = {
                'strategy': strategy,
                'metrics': metrics,
                'score': self._calculate_strategy_score(metrics)
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Ошибка тестирования стратегии {strategy['name']}: {e}")
            return None
    
    def _calculate_strategy_score(self, metrics: Dict) -> float:
        """
        Рассчитывает комплексную оценку стратегии
        """
        try:
            # Основные метрики
            total_return = metrics.get('Total Return (%)', 0)
            win_rate = metrics.get('Win Rate (%)', 0)
            profit_factor = metrics.get('Profit Factor', 0)
            max_drawdown = abs(metrics.get('Max Drawdown (%)', 0))
            sharpe_ratio = metrics.get('Sharpe Ratio', 0)
            sortino_ratio = metrics.get('Sortino Ratio', 0)
            total_trades = metrics.get('Total Trades', 0)
            
            # Взвешенная оценка
            score = (
                total_return * 0.25 +                    # 25% - общая доходность
                win_rate * 0.15 +                        # 15% - процент выигрышных сделок
                (profit_factor - 1) * 30 +               # 20% - profit factor
                max(0, 100 - max_drawdown) * 0.15 +      # 15% - контроль просадки
                sharpe_ratio * 8 +                       # 10% - коэффициент Шарпа
                sortino_ratio * 5 +                      # 5% - коэффициент Сортино
                min(total_trades / 50, 1) * 10           # 10% - количество сделок (до 50)
            )
            
            return max(0, score)
            
        except Exception as e:
            logger.error(f"Ошибка расчета оценки стратегии: {e}")
            return 0
    
    def search_best_strategies(self, df: pd.DataFrame, max_strategies=1000) -> List[Dict]:
        """
        Ищет лучшие стратегии среди всех возможных комбинаций
        """
        logger.info("Начинаю поиск лучших торговых стратегий...")
        
        # Генерируем все возможные стратегии
        all_strategies = []
        all_strategies.extend(self.generate_single_indicator_strategies(df))
        all_strategies.extend(self.generate_multi_indicator_strategies(df))
        
        # Ограничиваем количество для тестирования
        if len(all_strategies) > max_strategies:
            logger.info(f"Ограничиваю количество стратегий до {max_strategies}")
            all_strategies = all_strategies[:max_strategies]
        
        logger.info(f"Тестирую {len(all_strategies)} стратегий...")
        
        # Тестируем каждую стратегию
        valid_results = []
        for i, strategy in enumerate(all_strategies):
            if (i + 1) % 100 == 0:
                logger.info(f"Протестировано {i + 1}/{len(all_strategies)} стратегий")
            
            result = self.test_strategy(df, strategy)
            if result:
                valid_results.append(result)
                self.search_results.append(result)
        
        # Сортируем по оценке
        valid_results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Найдено {len(valid_results)} валидных стратегий")
        
        # Сохраняем лучшие
        self.best_strategies = valid_results[:20]  # Топ-20
        
        return self.best_strategies
    
    def get_best_strategies_summary(self) -> pd.DataFrame:
        """
        Возвращает сводку лучших стратегий
        """
        if not self.best_strategies:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.best_strategies:
            strategy = result['strategy']
            metrics = result['metrics']
            
            summary_data.append({
                'Strategy Name': strategy['name'],
                'Type': strategy['type'],
                'Indicators': ', '.join(strategy.get('indicators', [strategy.get('indicator', '')])),
                'Total Return (%)': metrics.get('Total Return (%)', 0),
                'Win Rate (%)': metrics.get('Win Rate (%)', 0),
                'Profit Factor': metrics.get('Profit Factor', 0),
                'Max Drawdown (%)': metrics.get('Max Drawdown (%)', 0),
                'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
                'Total Trades': metrics.get('Total Trades', 0),
                'Score': result['score']
            })
        
        return pd.DataFrame(summary_data)
    
    def create_custom_strategy_from_best(self, top_n=5) -> str:
        """
        Создает кастомную стратегию на основе лучших найденных
        """
        if len(self.best_strategies) < top_n:
            top_n = len(self.best_strategies)
        
        if top_n == 0:
            return "def custom_strategy(df):\n    df['signal'] = 'HOLD'\n    return df"
        
        # Берем топ стратегии
        top_strategies = self.best_strategies[:top_n]
        
        # Создаем код стратегии
        strategy_code = [
            "def auto_generated_strategy(df):",
            "    \"\"\"",
            "    Автоматически сгенерированная стратегия на основе лучших найденных комбинаций",
            "    \"\"\"",
            "    import numpy as np",
            "    import pandas as pd",
            "    ",
            "    df = df.copy()",
            "    df['signal'] = 'HOLD'",
            "    df['strategy_votes'] = 0",
            "    ",
        ]
        
        # Добавляем логику каждой стратегии с весами
        for i, result in enumerate(top_strategies):
            strategy = result['strategy']
            weight = result['score'] / sum(r['score'] for r in top_strategies)
            
            strategy_code.extend([
                f"    # Стратегия {i+1}: {strategy['name']} (вес: {weight:.3f})",
                f"    # {strategy.get('buy_condition', '')}",
            ])
            
            # Добавляем подготовку индикаторов
            params = strategy.get('params', {})
            if 'rsi_period' in params:
                period = params['rsi_period']
                strategy_code.extend([
                    f"    # RSI {period}",
                    f"    delta = df['close'].diff()",
                    f"    gain = (delta.where(delta > 0, 0)).rolling(window={period}).mean()",
                    f"    loss = (-delta.where(delta < 0, 0)).rolling(window={period}).mean()",
                    f"    rs = gain / loss",
                    f"    df['RSI_{period}'] = 100 - (100 / (1 + rs))",
                ])
            
            # Добавляем условия голосования
            buy_condition = strategy.get('buy_condition', '').replace('RSI', f'RSI_{params.get("rsi_period", 14)}')
            if buy_condition:
                strategy_code.append(f"    df.loc[{buy_condition.replace('df[', '').replace(']', '')}, 'strategy_votes'] += {weight}")
        
        strategy_code.extend([
            "    ",
            "    # Финальное решение на основе взвешенного голосования",
            "    df.loc[df['strategy_votes'] > 0.5, 'signal'] = 'BUY'",
            "    df.loc[df['strategy_votes'] < -0.5, 'signal'] = 'SELL'",
            "    ",
            "    return df"
        ])
        
        return '\n'.join(strategy_code)
    
    def save_results(self, filepath: str):
        """
        Сохраняет результаты поиска
        """
        try:
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'total_strategies_tested': len(self.search_results),
                'valid_strategies_found': len(self.best_strategies),
                'best_strategies': []
            }
            
            for result in self.best_strategies:
                results_data['best_strategies'].append({
                    'strategy': result['strategy'],
                    'metrics': result['metrics'],
                    'score': result['score']
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Результаты поиска сохранены: {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {e}")
    
    def load_results(self, filepath: str):
        """
        Загружает результаты поиска
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            self.best_strategies = results_data.get('best_strategies', [])
            logger.info(f"Результаты поиска загружены: {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки результатов: {e}")

