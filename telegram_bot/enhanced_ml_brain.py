#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СУПЕР-МОЗГ ДЛЯ TELEGRAM БОТА
Продвинутая ML система, которая:
1. Загружает ВСЕ данные Bitcoin
2. Рассчитывает ВСЕ индикаторы
3. Находит лучшие комбинации автоматически
4. Оптимизирует плечи, стоп-лоссы и тейк-профиты
5. Постоянно ищет лучшие варианты
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import os
import json
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedMLBrain:
    """
    Продвинутый ML мозг для Telegram бота
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_strategies = []
        self.performance_history = []
        
        # Настройки поиска
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        self.max_lookback_days = 30
        self.min_confidence = 0.65
        
        # Торговые настройки
        self.leverage_range = [1, 2, 3, 5, 10]
        self.sl_range = [0.5, 1.0, 1.5, 2.0, 3.0]  # проценты
        self.tp_ratios = [1.5, 2.0, 2.5, 3.0, 4.0]  # отношение к SL
        
        # Кэш данных
        self.data_cache = {}
        self.indicators_cache = {}
        
    def load_all_bitcoin_data(self, symbol="BTC/USDT") -> Dict[str, pd.DataFrame]:
        """
        Загружает данные Bitcoin по всем таймфреймам
        """
        print("📊 Загрузка данных Bitcoin по всем таймфреймам...")
        
        all_data = {}
        
        for timeframe in self.timeframes:
            try:
                from data.data_manager import get_candlestick_data
                
                # Определяем количество свечей на основе таймфрейма
                if timeframe == '1m':
                    limit = 1440  # 1 день
                elif timeframe == '5m':
                    limit = 2016  # 7 дней  
                elif timeframe == '15m':
                    limit = 2016  # 21 день
                elif timeframe == '1h':
                    limit = 720   # 30 дней
                elif timeframe == '4h':
                    limit = 720   # 120 дней
                else:
                    limit = 500
                
                df = get_candlestick_data(symbol, timeframe, limit=limit, private=True)
                
                if not df.empty:
                    all_data[timeframe] = df
                    print(f"✅ {timeframe}: {len(df)} записей")
                else:
                    print(f"⚠️ {timeframe}: данные не получены")
                    
            except Exception as e:
                print(f"❌ Ошибка загрузки {timeframe}: {e}")
        
        self.data_cache = all_data
        print(f"📈 Загружено {len(all_data)} таймфреймов")
        return all_data
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает ВСЕ доступные индикаторы
        """
        try:
            # Базовые индикаторы
            from indicators.indicators import (
                calculate_rsi, calculate_macd, calculate_atr,
                calculate_sma, calculate_ema, calculate_bollinger_bands,
                calculate_stochastic, calculate_williams_r, calculate_cci,
                calculate_momentum, calculate_roc, calculate_obv
            )
            
            # Применяем базовые индикаторы
            df = calculate_rsi(df, period=14)
            df = calculate_rsi(df, period=21)  # Дополнительный RSI
            df = calculate_macd(df)
            df = calculate_atr(df)
            
            # Moving Averages
            for period in [10, 20, 50, 100, 200]:
                df = calculate_sma(df, period=period)
                df = calculate_ema(df, period=period)
            
            df = calculate_bollinger_bands(df)
            df = calculate_stochastic(df)
            df = calculate_williams_r(df)
            df = calculate_cci(df)
            df = calculate_momentum(df)
            df = calculate_roc(df)
            df = calculate_obv(df)
            
            # Расширенные индикаторы
            try:
                from indicators.enhanced_indicators import calculate_all_enhanced_indicators
                df = calculate_all_enhanced_indicators(df)
            except Exception as e:
                logger.warning(f"Не удалось загрузить расширенные индикаторы: {e}")
            
            print(f"📊 Рассчитано индикаторов: {len(df.columns)} колонок")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает продвинутые признаки для ML
        """
        df_features = df.copy()
        
        # Процентные изменения
        for period in [1, 3, 5, 10, 20]:
            df_features[f'price_change_{period}'] = df_features['close'].pct_change(period) * 100
            df_features[f'volume_change_{period}'] = df_features['volume'].pct_change(period) * 100
        
        # Волатильность
        for window in [5, 10, 20]:
            df_features[f'volatility_{window}'] = df_features['close'].pct_change().rolling(window).std() * 100
            df_features[f'volume_volatility_{window}'] = df_features['volume'].pct_change().rolling(window).std() * 100
        
        # Технические паттерны
        df_features['higher_highs'] = (df_features['high'] > df_features['high'].shift(1)).rolling(5).sum()
        df_features['lower_lows'] = (df_features['low'] < df_features['low'].shift(1)).rolling(5).sum()
        
        # Соотношения
        df_features['hl_ratio'] = (df_features['high'] - df_features['low']) / df_features['close'] * 100
        df_features['oc_ratio'] = (df_features['close'] - df_features['open']) / df_features['open'] * 100
        
        # Уровни поддержки/сопротивления
        df_features['support'] = df_features['low'].rolling(20).min()
        df_features['resistance'] = df_features['high'].rolling(20).max()
        df_features['support_distance'] = (df_features['close'] - df_features['support']) / df_features['close'] * 100
        df_features['resistance_distance'] = (df_features['resistance'] - df_features['close']) / df_features['close'] * 100
        
        # Убираем NaN
        df_features = df_features.dropna()
        
        print(f"🧠 Создано {len(df_features.columns)} признаков")
        return df_features
    
    def find_best_indicator_combinations(self, df: pd.DataFrame, max_combinations=100) -> List[Dict]:
        """
        Автоматически находит лучшие комбинации индикаторов
        """
        print("🔍 Поиск лучших комбинаций индикаторов...")
        
        # Основные индикаторы для комбинирования
        base_indicators = ['RSI', 'MACD', 'SMA_20', 'EMA_20', 'BB_upper', 'BB_lower', 'ATR']
        available_indicators = [col for col in base_indicators if col in df.columns]
        
        if len(available_indicators) < 2:
            print("⚠️ Недостаточно индикаторов для комбинирования")
            return []
        
        combinations = []
        
        # Генерируем комбинации по 2-4 индикатора
        from itertools import combinations as iter_combinations
        
        for combo_size in [2, 3, 4]:
            for combo in iter_combinations(available_indicators, combo_size):
                strategy = self._create_combination_strategy(df, list(combo))
                if strategy and strategy['win_rate'] > 40:  # Минимум 40% винрейт
                    combinations.append(strategy)
        
        # Сортируем по производительности
        combinations.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"🎯 Найдено {len(combinations)} хороших комбинаций")
        return combinations[:max_combinations]
    
    def _create_combination_strategy(self, df: pd.DataFrame, indicators: List[str]) -> Optional[Dict]:
        """
        Создает и тестирует стратегию на основе комбинации индикаторов
        """
        try:
            df_test = df.copy()
            df_test['signal'] = 'HOLD'
            
            # Простая логика комбинирования
            buy_conditions = []
            sell_conditions = []
            
            for indicator in indicators:
                if indicator == 'RSI':
                    buy_conditions.append(df_test['RSI'] < 35)
                    sell_conditions.append(df_test['RSI'] > 65)
                elif indicator == 'MACD':
                    buy_conditions.append(df_test['MACD'] > df_test['MACD_signal'])
                    sell_conditions.append(df_test['MACD'] < df_test['MACD_signal'])
                elif 'SMA' in indicator or 'EMA' in indicator:
                    buy_conditions.append(df_test['close'] > df_test[indicator])
                    sell_conditions.append(df_test['close'] < df_test[indicator])
                elif indicator == 'BB_upper':
                    sell_conditions.append(df_test['close'] > df_test['BB_upper'])
                elif indicator == 'BB_lower':
                    buy_conditions.append(df_test['close'] < df_test['BB_lower'])
            
            # Применяем условия
            if buy_conditions:
                buy_signal = buy_conditions[0]
                for condition in buy_conditions[1:]:
                    buy_signal = buy_signal & condition
                df_test.loc[buy_signal, 'signal'] = 'BUY'
            
            if sell_conditions:
                sell_signal = sell_conditions[0]
                for condition in sell_conditions[1:]:
                    sell_signal = sell_signal | condition  # OR для продаж
                df_test.loc[sell_signal, 'signal'] = 'SELL'
            
            # Быстрый бэктест
            performance = self._quick_backtest(df_test)
            
            if performance and performance['total_trades'] >= 5:
                return {
                    'indicators': indicators,
                    'win_rate': performance['win_rate'],
                    'total_return': performance['total_return'],
                    'total_trades': performance['total_trades'],
                    'max_drawdown': performance['max_drawdown'],
                    'score': performance['win_rate'] * performance['total_return'] / max(1, abs(performance['max_drawdown']))
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Ошибка тестирования комбинации {indicators}: {e}")
            return None
    
    def _quick_backtest(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Быстрый бэктест для оценки стратегии
        """
        try:
            balance = 1000
            position = 0
            trades = []
            
            for i in range(1, len(df)):
                signal = df['signal'].iloc[i]
                price = df['close'].iloc[i]
                
                if signal == 'BUY' and position == 0:
                    # Покупаем
                    position = balance / price
                    balance = 0
                    entry_price = price
                    
                elif signal == 'SELL' and position > 0:
                    # Продаем
                    balance = position * price
                    profit = (price - entry_price) / entry_price * 100
                    trades.append(profit)
                    position = 0
            
            # Закрываем позицию если осталась
            if position > 0:
                balance = position * df['close'].iloc[-1]
                profit = (df['close'].iloc[-1] - entry_price) / entry_price * 100
                trades.append(profit)
            
            if len(trades) == 0:
                return None
            
            # Рассчитываем метрики
            total_return = (balance - 1000) / 1000 * 100
            win_rate = sum(1 for t in trades if t > 0) / len(trades) * 100
            
            # Максимальная просадка
            equity_curve = [1000]
            running_balance = 1000
            for trade in trades:
                running_balance *= (1 + trade / 100)
                equity_curve.append(running_balance)
            
            peak = equity_curve[0]
            max_drawdown = 0
            for balance in equity_curve:
                peak = max(peak, balance)
                drawdown = (peak - balance) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            return {
                'total_return': total_return,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.warning(f"Ошибка быстрого бэктеста: {e}")
            return None
    
    def optimize_trading_parameters(self, best_strategies: List[Dict]) -> Dict:
        """
        Оптимизирует торговые параметры: плечи, SL, TP
        """
        print("⚙️ Оптимизация торговых параметров...")
        
        best_params = {
            'leverage': 1,
            'stop_loss_pct': 2.0,
            'take_profit_ratio': 2.0,
            'confidence_threshold': 0.65
        }
        
        if not best_strategies:
            return best_params
        
        # Берем топ-3 стратегии для оптимизации
        top_strategies = best_strategies[:3]
        
        best_score = 0
        
        # Перебираем комбинации параметров
        for leverage in self.leverage_range:
            for sl_pct in self.sl_range:
                for tp_ratio in self.tp_ratios:
                    # Рассчитываем ожидаемую доходность с учетом плеча
                    score = 0
                    
                    for strategy in top_strategies:
                        # Учитываем плечо
                        leveraged_return = strategy['total_return'] * leverage
                        
                        # Штраф за большие плечи (риск)
                        risk_penalty = leverage * 0.1
                        
                        # Штраф за большие стоп-лоссы
                        sl_penalty = sl_pct * 0.05
                        
                        # Бонус за хорошее соотношение TP/SL
                        tp_bonus = tp_ratio * 0.1
                        
                        strategy_score = (
                            leveraged_return * strategy['win_rate'] / 100 
                            - risk_penalty 
                            - sl_penalty 
                            + tp_bonus
                        )
                        
                        score += strategy_score
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'leverage': leverage,
                            'stop_loss_pct': sl_pct,
                            'take_profit_ratio': tp_ratio,
                            'confidence_threshold': 0.65,
                            'expected_score': score
                        }
        
        print(f"🎯 Лучшие параметры: Плечо {best_params['leverage']}x, SL {best_params['stop_loss_pct']}%, TP {best_params['take_profit_ratio']}x")
        return best_params
    
    def train_ml_ensemble(self, df: pd.DataFrame) -> Dict:
        """
        Обучает ансамбль ML моделей
        """
        print("🤖 Обучение ансамбля ML моделей...")
        
        if not SKLEARN_AVAILABLE:
            print("⚠️ Scikit-learn не доступен")
            return {}
        
        # Подготовка данных
        df_features = self.create_advanced_features(df)
        
        # Выбираем числовые признаки
        feature_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if len(feature_cols) < 5:
            print("⚠️ Недостаточно признаков для ML")
            return {}
        
        # Создаем целевую переменную (будущее изменение цены)
        df_features['future_return'] = df_features['close'].pct_change(5).shift(-5) * 100  # 5 периодов вперед
        df_features = df_features.dropna()
        
        if len(df_features) < 100:
            print("⚠️ Недостаточно данных для обучения")
            return {}
        
        X = df_features[feature_cols]
        y = df_features['future_return']
        
        # Разделение данных
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Нормализация
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # Random Forest
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            rf_score = rf.score(X_test_scaled, y_test)
            models['random_forest'] = {'model': rf, 'score': rf_score}
            print(f"✅ Random Forest: R² = {rf_score:.4f}")
        except Exception as e:
            print(f"❌ Random Forest: {e}")
        
        # Gradient Boosting
        try:
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X_train_scaled, y_train)
            gb_score = gb.score(X_test_scaled, y_test)
            models['gradient_boosting'] = {'model': gb, 'score': gb_score}
            print(f"✅ Gradient Boosting: R² = {gb_score:.4f}")
        except Exception as e:
            print(f"❌ Gradient Boosting: {e}")
        
        # Сохраняем лучшую модель
        if models:
            best_model_name = max(models.keys(), key=lambda k: models[k]['score'])
            best_model = models[best_model_name]
            
            # Сохраняем модель и скейлер
            joblib.dump(best_model['model'], 'enhanced_ml_model.pkl')
            joblib.dump(scaler, 'enhanced_ml_scaler.pkl')
            joblib.dump(feature_cols, 'enhanced_ml_features.pkl')
            
            print(f"🏆 Лучшая модель: {best_model_name} (R² = {best_model['score']:.4f})")
            
            return {
                'best_model': best_model_name,
                'score': best_model['score'],
                'features': feature_cols,
                'models_trained': len(models)
            }
        
        return {}
    
    def generate_enhanced_prediction(self, current_data: pd.DataFrame) -> Dict:
        """
        Генерирует улучшенный прогноз с использованием всех данных
        """
        try:
            # Загружаем все данные если нужно
            if not self.data_cache:
                self.load_all_bitcoin_data()
            
            # Рассчитываем все индикаторы
            df_with_indicators = self.calculate_all_indicators(current_data)
            
            # Создаем признаки
            df_features = self.create_advanced_features(df_with_indicators)
            
            # Пытаемся загрузить обученную модель
            prediction_result = {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reasoning': 'Базовый анализ',
                'recommended_leverage': 1,
                'stop_loss_pct': 2.0,
                'take_profit_ratio': 2.0
            }
            
            try:
                if os.path.exists('enhanced_ml_model.pkl'):
                    model = joblib.load('enhanced_ml_model.pkl')
                    scaler = joblib.load('enhanced_ml_scaler.pkl')
                    feature_cols = joblib.load('enhanced_ml_features.pkl')
                    
                    # Подготавливаем данные для предсказания
                    available_features = [col for col in feature_cols if col in df_features.columns]
                    
                    if len(available_features) >= len(feature_cols) * 0.8:  # Минимум 80% признаков
                        X_pred = df_features[available_features].iloc[-1:].fillna(0)
                        X_pred_scaled = scaler.transform(X_pred)
                        
                        prediction = model.predict(X_pred_scaled)[0]
                        
                        # Конвертируем предсказание в сигнал
                        if prediction > 1.0:
                            prediction_result['signal'] = 'STRONG_BUY'
                            prediction_result['confidence'] = min(abs(prediction) / 5, 1.0)
                        elif prediction > 0.5:
                            prediction_result['signal'] = 'BUY'
                            prediction_result['confidence'] = min(abs(prediction) / 3, 1.0)
                        elif prediction < -1.0:
                            prediction_result['signal'] = 'STRONG_SELL'
                            prediction_result['confidence'] = min(abs(prediction) / 5, 1.0)
                        elif prediction < -0.5:
                            prediction_result['signal'] = 'SELL'
                            prediction_result['confidence'] = min(abs(prediction) / 3, 1.0)
                        else:
                            prediction_result['signal'] = 'HOLD'
                            prediction_result['confidence'] = 0.1
                        
                        prediction_result['reasoning'] = f'Enhanced ML prediction: {prediction:.3f}'
                        
            except Exception as e:
                logger.warning(f"Не удалось использовать enhanced модель: {e}")
            
            # Используем найденные лучшие параметры если есть
            if self.best_strategies:
                best_params = self.optimize_trading_parameters(self.best_strategies)
                prediction_result.update({
                    'recommended_leverage': best_params['leverage'],
                    'stop_loss_pct': best_params['stop_loss_pct'],
                    'take_profit_ratio': best_params['take_profit_ratio']
                })
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Ошибка генерации прогноза: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reasoning': f'Ошибка: {e}',
                'recommended_leverage': 1,
                'stop_loss_pct': 2.0,
                'take_profit_ratio': 2.0
            }
    
    def run_full_analysis(self) -> Dict:
        """
        Запускает полный анализ и поиск лучших стратегий
        """
        print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА ML МОЗГА")
        print("=" * 50)
        
        results = {
            'timestamp': datetime.now(),
            'data_loaded': False,
            'indicators_calculated': False,
            'strategies_found': 0,
            'models_trained': 0,
            'best_parameters': {},
            'status': 'starting'
        }
        
        try:
            # 1. Загружаем все данные
            all_data = self.load_all_bitcoin_data()
            if all_data:
                results['data_loaded'] = True
                print("✅ Данные загружены")
            
            # 2. Берем данные 1h для основного анализа
            main_data = all_data.get('1h', all_data.get('4h'))
            if main_data is None or main_data.empty:
                print("❌ Нет данных для анализа")
                results['status'] = 'failed'
                return results
            
            # 3. Рассчитываем все индикаторы
            df_with_indicators = self.calculate_all_indicators(main_data)
            results['indicators_calculated'] = True
            print("✅ Индикаторы рассчитаны")
            
            # 4. Ищем лучшие комбинации
            self.best_strategies = self.find_best_indicator_combinations(df_with_indicators)
            results['strategies_found'] = len(self.best_strategies)
            print(f"✅ Найдено стратегий: {len(self.best_strategies)}")
            
            # 5. Оптимизируем параметры торговли
            if self.best_strategies:
                best_params = self.optimize_trading_parameters(self.best_strategies)
                results['best_parameters'] = best_params
                print("✅ Параметры оптимизированы")
            
            # 6. Обучаем ML модели
            ml_results = self.train_ml_ensemble(df_with_indicators)
            results['models_trained'] = ml_results.get('models_trained', 0)
            if ml_results:
                print("✅ ML модели обучены")
            
            results['status'] = 'completed'
            print("\n🎉 ПОЛНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка полного анализа: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
            return results

# Глобальный экземпляр ML мозга
ml_brain = EnhancedMLBrain()

def get_enhanced_prediction(current_data: pd.DataFrame = None) -> Dict:
    """
    Функция для получения улучшенного прогноза из Telegram бота
    """
    if current_data is None:
        # Загружаем текущие данные
        try:
            from data.data_manager import get_candlestick_data
            current_data = get_candlestick_data("BTC/USDT", "1h", limit=200, private=True)
        except Exception as e:
            logger.error(f"Не удалось загрузить данные: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reasoning': 'Нет данных для анализа',
                'recommended_leverage': 1,
                'stop_loss_pct': 2.0,
                'take_profit_ratio': 2.0
            }
    
    return ml_brain.generate_enhanced_prediction(current_data)

def run_brain_training() -> Dict:
    """
    Функция для запуска обучения ML мозга
    """
    try:
        return ml_brain.run_full_analysis()
    except Exception as e:
        logger.error(f"Ошибка обучения ML мозга: {e}")
        return {
            'status': 'error',
            'error': f'Ошибка обучения: {e}',
            'timestamp': datetime.now()
        }
