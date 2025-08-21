#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Предиктивная система торговых сигналов на основе машинного обучения
Комбинирует технический анализ с прогнозами ML моделей
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PredictiveSignalGenerator:
    """
    Генератор торговых сигналов на основе ML прогнозов
    """
    
    def __init__(self, ml_model=None, confidence_threshold=0.6):
        """
        Инициализация генератора сигналов
        
        Args:
            ml_model: Обученная ML модель для прогнозирования
            confidence_threshold: Порог уверенности для генерации сигналов
        """
        self.ml_model = ml_model
        self.confidence_threshold = confidence_threshold
        self.signal_history = []
        self.last_signals = {}
        
    def generate_ml_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерирует сигналы на основе ML прогнозов
        """
        df_signals = df.copy()
        df_signals['ml_signal'] = 'HOLD'
        df_signals['ml_confidence'] = 0.0
        df_signals['predicted_price'] = np.nan
        
        if self.ml_model is None:
            logger.warning("ML модель не загружена")
            return df_signals
        
        try:
            # Делаем прогноз
            prediction = self.ml_model.predict(df)
            
            if len(prediction) > 0:
                # Берем последний прогноз
                predicted_price = prediction[-1] if isinstance(prediction, (list, np.ndarray)) else prediction
                current_price = df['close'].iloc[-1]
                
                # Рассчитываем ожидаемое изменение цены
                price_change_pct = (predicted_price - current_price) / current_price * 100
                
                # Определяем уверенность на основе величины изменения
                confidence = min(abs(price_change_pct) / 5.0, 1.0)  # Максимум при 5% изменении
                
                # Генерируем сигнал
                if price_change_pct > 1.0 and confidence > self.confidence_threshold:
                    signal = 'BUY'
                elif price_change_pct < -1.0 and confidence > self.confidence_threshold:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                # Записываем результаты в последнюю строку
                df_signals.iloc[-1, df_signals.columns.get_loc('ml_signal')] = signal
                df_signals.iloc[-1, df_signals.columns.get_loc('ml_confidence')] = confidence
                df_signals.iloc[-1, df_signals.columns.get_loc('predicted_price')] = predicted_price
                
                logger.info(f"ML сигнал: {signal} (уверенность: {confidence:.3f}, прогноз: {predicted_price:.2f})")
        
        except Exception as e:
            logger.error(f"Ошибка генерации ML сигналов: {e}")
        
        return df_signals
    
    def generate_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерирует сигналы на основе технического анализа
        """
        df_signals = df.copy()
        df_signals['tech_signal'] = 'HOLD'
        df_signals['tech_strength'] = 0.0
        
        # Множественные технические сигналы
        signals = []
        strengths = []
        
        # 1. RSI сигналы
        if 'RSI' in df.columns:
            rsi_signals = []
            rsi_values = df['RSI'].dropna()
            
            if len(rsi_values) > 0:
                current_rsi = rsi_values.iloc[-1]
                
                if current_rsi < 30:
                    rsi_signals.append(('BUY', 0.8))
                elif current_rsi < 35:
                    rsi_signals.append(('BUY', 0.5))
                elif current_rsi > 70:
                    rsi_signals.append(('SELL', 0.8))
                elif current_rsi > 65:
                    rsi_signals.append(('SELL', 0.5))
                else:
                    rsi_signals.append(('HOLD', 0.1))
                
                signals.extend(rsi_signals)
        
        # 2. MACD сигналы
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd_signals = []
            macd_diff = df['MACD'] - df['MACD_signal']
            
            if len(macd_diff.dropna()) > 1:
                current_diff = macd_diff.dropna().iloc[-1]
                prev_diff = macd_diff.dropna().iloc[-2]
                
                # Пересечение линий
                if prev_diff <= 0 and current_diff > 0:
                    macd_signals.append(('BUY', 0.7))
                elif prev_diff >= 0 and current_diff < 0:
                    macd_signals.append(('SELL', 0.7))
                elif current_diff > 0:
                    macd_signals.append(('BUY', 0.3))
                else:
                    macd_signals.append(('SELL', 0.3))
                
                signals.extend(macd_signals)
        
        # 3. Bollinger Bands сигналы
        if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'close']):
            bb_signals = []
            
            current_price = df['close'].iloc[-1]
            bb_upper = df['BB_upper'].iloc[-1]
            bb_lower = df['BB_lower'].iloc[-1]
            bb_middle = (bb_upper + bb_lower) / 2
            
            # Позиция относительно полос
            if current_price < bb_lower:
                bb_signals.append(('BUY', 0.6))
            elif current_price > bb_upper:
                bb_signals.append(('SELL', 0.6))
            elif current_price < bb_middle:
                bb_signals.append(('BUY', 0.2))
            else:
                bb_signals.append(('SELL', 0.2))
            
            signals.extend(bb_signals)
        
        # 4. Moving Average сигналы
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            ma_signals = []
            
            sma_20 = df['SMA_20'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Тренд по MA
            if sma_20 > sma_50 and current_price > sma_20:
                ma_signals.append(('BUY', 0.5))
            elif sma_20 < sma_50 and current_price < sma_20:
                ma_signals.append(('SELL', 0.5))
            else:
                ma_signals.append(('HOLD', 0.1))
            
            signals.extend(ma_signals)
        
        # 5. Volume сигналы
        if 'volume' in df.columns:
            volume_signals = []
            
            if len(df) >= 20:
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].rolling(20).mean().iloc[-1]
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio > 2.0:  # Высокий объем
                    # Определяем направление по цене
                    price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                    if price_change > 0:
                        volume_signals.append(('BUY', 0.4))
                    else:
                        volume_signals.append(('SELL', 0.4))
                
                signals.extend(volume_signals)
        
        # Агрегируем сигналы
        if signals:
            buy_strength = sum(strength for signal, strength in signals if signal == 'BUY')
            sell_strength = sum(strength for signal, strength in signals if signal == 'SELL')
            
            total_strength = buy_strength + sell_strength
            
            if buy_strength > sell_strength and buy_strength > 1.0:
                final_signal = 'BUY'
                final_strength = buy_strength / max(total_strength, 1)
            elif sell_strength > buy_strength and sell_strength > 1.0:
                final_signal = 'SELL'
                final_strength = sell_strength / max(total_strength, 1)
            else:
                final_signal = 'HOLD'
                final_strength = 0.1
        else:
            final_signal = 'HOLD'
            final_strength = 0.0
        
        # Записываем результаты
        df_signals.iloc[-1, df_signals.columns.get_loc('tech_signal')] = final_signal
        df_signals.iloc[-1, df_signals.columns.get_loc('tech_strength')] = final_strength
        
        return df_signals
    
    def combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Комбинирует ML и технические сигналы в финальный сигнал
        """
        # Генерируем оба типа сигналов
        df_with_ml = self.generate_ml_signals(df)
        df_with_tech = self.generate_technical_signals(df_with_ml)
        
        df_final = df_with_tech.copy()
        df_final['final_signal'] = 'HOLD'
        df_final['signal_confidence'] = 0.0
        df_final['signal_reasoning'] = ''
        
        # Берем последние сигналы
        ml_signal = df_final['ml_signal'].iloc[-1]
        ml_confidence = df_final['ml_confidence'].iloc[-1]
        tech_signal = df_final['tech_signal'].iloc[-1]
        tech_strength = df_final['tech_strength'].iloc[-1]
        
        # Логика комбинирования
        reasoning_parts = []
        
        # Веса для разных типов сигналов
        ml_weight = 0.6
        tech_weight = 0.4
        
        # Рассчитываем взвешенные голоса
        ml_vote = 0
        tech_vote = 0
        
        if ml_signal == 'BUY':
            ml_vote = ml_confidence * ml_weight
            reasoning_parts.append(f"ML: BUY ({ml_confidence:.2f})")
        elif ml_signal == 'SELL':
            ml_vote = -ml_confidence * ml_weight
            reasoning_parts.append(f"ML: SELL ({ml_confidence:.2f})")
        
        if tech_signal == 'BUY':
            tech_vote = tech_strength * tech_weight
            reasoning_parts.append(f"Tech: BUY ({tech_strength:.2f})")
        elif tech_signal == 'SELL':
            tech_vote = -tech_strength * tech_weight
            reasoning_parts.append(f"Tech: SELL ({tech_strength:.2f})")
        
        total_vote = ml_vote + tech_vote
        final_confidence = abs(total_vote)
        
        # Определяем финальный сигнал
        if total_vote > 0.4:
            final_signal = 'BUY'
        elif total_vote < -0.4:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        # Дополнительные фильтры
        if final_signal != 'HOLD':
            # Фильтр по минимальной уверенности
            if final_confidence < 0.3:
                final_signal = 'HOLD'
                reasoning_parts.append("Filtered: Low confidence")
            
            # Фильтр по согласованности сигналов
            if ml_signal != 'HOLD' and tech_signal != 'HOLD' and ml_signal != tech_signal:
                final_confidence *= 0.5  # Снижаем уверенность при конфликте
                reasoning_parts.append("Conflict detected")
        
        # Записываем результаты
        df_final.iloc[-1, df_final.columns.get_loc('final_signal')] = final_signal
        df_final.iloc[-1, df_final.columns.get_loc('signal_confidence')] = final_confidence
        df_final.iloc[-1, df_final.columns.get_loc('signal_reasoning')] = '; '.join(reasoning_parts)
        
        # Сохраняем для истории
        signal_info = {
            'timestamp': datetime.now(),
            'price': df_final['close'].iloc[-1],
            'signal': final_signal,
            'confidence': final_confidence,
            'ml_signal': ml_signal,
            'ml_confidence': ml_confidence,
            'tech_signal': tech_signal,
            'tech_strength': tech_strength,
            'reasoning': '; '.join(reasoning_parts)
        }
        
        self.signal_history.append(signal_info)
        self.last_signals = signal_info
        
        logger.info(f"Финальный сигнал: {final_signal} (уверенность: {final_confidence:.3f})")
        logger.info(f"Обоснование: {'; '.join(reasoning_parts)}")
        
        return df_final
    
    def get_signal_strength_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Анализирует силу текущих сигналов
        """
        df_with_signals = self.combine_signals(df)
        
        analysis = {
            'current_price': df['close'].iloc[-1],
            'signals': {
                'ml_signal': df_with_signals['ml_signal'].iloc[-1],
                'ml_confidence': df_with_signals['ml_confidence'].iloc[-1],
                'tech_signal': df_with_signals['tech_signal'].iloc[-1],
                'tech_strength': df_with_signals['tech_strength'].iloc[-1],
                'final_signal': df_with_signals['final_signal'].iloc[-1],
                'final_confidence': df_with_signals['signal_confidence'].iloc[-1]
            },
            'reasoning': df_with_signals['signal_reasoning'].iloc[-1],
            'timestamp': datetime.now()
        }
        
        # Дополнительный анализ
        if 'predicted_price' in df_with_signals.columns:
            predicted_price = df_with_signals['predicted_price'].iloc[-1]
            if not np.isnan(predicted_price):
                price_change_pct = (predicted_price - analysis['current_price']) / analysis['current_price'] * 100
                analysis['price_prediction'] = {
                    'predicted_price': predicted_price,
                    'expected_change_pct': price_change_pct,
                    'potential_profit': abs(price_change_pct)
                }
        
        return analysis
    
    def generate_trading_recommendation(self, df: pd.DataFrame, position_size=1000) -> Dict:
        """
        Генерирует торговые рекомендации
        """
        analysis = self.get_signal_strength_analysis(df)
        
        recommendation = {
            'action': analysis['signals']['final_signal'],
            'confidence': analysis['signals']['final_confidence'],
            'reasoning': analysis['reasoning'],
            'timestamp': analysis['timestamp'],
            'current_price': analysis['current_price']
        }
        
        # Рекомендации по позиции
        if recommendation['action'] == 'BUY':
            confidence = recommendation['confidence']
            recommended_size = position_size * min(confidence * 2, 1.0)  # Максимум 100% при уверенности 0.5+
            
            recommendation['position'] = {
                'action': 'LONG',
                'recommended_size': recommended_size,
                'entry_price': analysis['current_price'],
                'stop_loss': analysis['current_price'] * 0.98,  # 2% stop loss
                'take_profit': analysis['current_price'] * 1.04,  # 4% take profit
                'risk_reward_ratio': 2.0
            }
            
        elif recommendation['action'] == 'SELL':
            confidence = recommendation['confidence']
            recommended_size = position_size * min(confidence * 2, 1.0)
            
            recommendation['position'] = {
                'action': 'SHORT',
                'recommended_size': recommended_size,
                'entry_price': analysis['current_price'],
                'stop_loss': analysis['current_price'] * 1.02,  # 2% stop loss
                'take_profit': analysis['current_price'] * 0.96,  # 4% take profit
                'risk_reward_ratio': 2.0
            }
        else:
            recommendation['position'] = {
                'action': 'HOLD',
                'message': 'Недостаточно сильный сигнал для входа в позицию'
            }
        
        # Добавляем прогноз цены если есть
        if 'price_prediction' in analysis:
            recommendation['price_prediction'] = analysis['price_prediction']
        
        return recommendation
    
    def get_signal_history_summary(self) -> pd.DataFrame:
        """
        Возвращает сводку истории сигналов
        """
        if not self.signal_history:
            return pd.DataFrame()
        
        df_history = pd.DataFrame(self.signal_history)
        
        # Добавляем статистику
        signal_counts = df_history['signal'].value_counts()
        avg_confidence = df_history.groupby('signal')['confidence'].mean()
        
        summary = {
            'total_signals': len(df_history),
            'signal_distribution': signal_counts.to_dict(),
            'average_confidence': avg_confidence.to_dict(),
            'last_signal': self.last_signals
        }
        
        return df_history, summary
    
    def set_ml_model(self, ml_model):
        """
        Устанавливает ML модель для прогнозирования
        """
        self.ml_model = ml_model
        logger.info("ML модель обновлена")
    
    def update_confidence_threshold(self, new_threshold: float):
        """
        Обновляет порог уверенности
        """
        self.confidence_threshold = max(0.1, min(1.0, new_threshold))
        logger.info(f"Порог уверенности обновлен: {self.confidence_threshold}")
    
    def clear_history(self):
        """
        Очищает историю сигналов
        """
        self.signal_history = []
        self.last_signals = {}
        logger.info("История сигналов очищена")

