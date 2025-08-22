#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Реальная торговая система для автоматического выполнения сделок
Интегрируется с ML прогнозами и управляет рисками
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Торговые библиотеки
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️ CCXT не установлен. Установите: pip install ccxt")

logger = logging.getLogger(__name__)

class LiveTrader:
    """
    Система реальной торговли с ML прогнозами
    """
    
    def __init__(self, exchange_config: Dict, initial_balance: float = 1000):
        """
        Инициализация торговой системы
        
        Args:
            exchange_config: Конфигурация биржи (API ключи)
            initial_balance: Начальный баланс для торговли
        """
        self.exchange_config = exchange_config
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.exchange = None
        self.positions = {}
        self.trades_history = []
        self.ml_model = None
        self.signal_generator = None
        
        # Параметры торговли
        self.max_position_size = 0.1  # Максимум 10% от баланса на сделку
        self.stop_loss_pct = 0.02     # 2% стоп-лосс
        self.take_profit_pct = 0.04   # 4% тейк-профит
        self.min_confidence = 0.6     # Минимальная уверенность для входа
        
        # Состояние системы
        self.is_trading = False
        self.last_update = None
        self.trading_symbols = ['BTC/USDT', 'ETH/USDT']
        
        # Статистика
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'start_time': datetime.now()
        }
    
    def initialize_exchange(self) -> bool:
        """
        Инициализирует подключение к бирже
        """
        if not CCXT_AVAILABLE:
            logger.error("CCXT библиотека не установлена")
            return False
        
        try:
            # Создаем подключение к Bybit
            self.exchange = ccxt.bybit({
                'apiKey': self.exchange_config.get('api_key', ''),
                'secret': self.exchange_config.get('secret', ''),
                'sandbox': self.exchange_config.get('sandbox', True),  # Testnet по умолчанию
                'enableRateLimit': True,
            })
            
            # Проверяем подключение
            balance = self.exchange.fetch_balance()
            logger.info("✅ Подключение к Bybit успешно")
            logger.info(f"Баланс USDT: {balance.get('USDT', {}).get('free', 0)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к бирже: {e}")
            return False
    
    def set_ml_system(self, ml_model, signal_generator):
        """
        Устанавливает ML модель и генератор сигналов
        """
        self.ml_model = ml_model
        self.signal_generator = signal_generator
        logger.info("ML система подключена к торговому модулю")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Получает текущую цену символа
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Ошибка получения цены {symbol}: {e}")
            return None
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Получает рыночные данные для анализа
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка получения данных {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """
        Рассчитывает размер позиции на основе уверенности и баланса
        """
        try:
            # Получаем текущий баланс
            balance = self.exchange.fetch_balance()
            available_usdt = balance['USDT']['free']
            
            # Рассчитываем размер позиции
            base_size = available_usdt * self.max_position_size
            confidence_multiplier = min(confidence * 2, 1.0)  # Максимум при уверенности 0.5+
            
            position_size = base_size * confidence_multiplier
            
            logger.info(f"Размер позиции для {symbol}: ${position_size:.2f} (уверенность: {confidence:.3f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {e}")
            return 0.0
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """
        Размещает рыночный ордер
        """
        try:
            order = self.exchange.create_market_order(symbol, side, amount)
            logger.info(f"✅ Ордер размещен: {side} {amount} {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"❌ Ошибка размещения ордера: {e}")
            return None
    
    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[Dict]:
        """
        Размещает лимитный ордер
        """
        try:
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            logger.info(f"✅ Лимитный ордер: {side} {amount} {symbol} @ ${price}")
            return order
            
        except Exception as e:
            logger.error(f"❌ Ошибка размещения лимитного ордера: {e}")
            return None
    
    def set_stop_loss_take_profit(self, symbol: str, side: str, entry_price: float, amount: float):
        """
        Устанавливает стоп-лосс и тейк-профит
        """
        try:
            if side == 'buy':
                # Для LONG позиции
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                take_profit_price = entry_price * (1 + self.take_profit_pct)
                
                # Стоп-лосс (продажа ниже входа)
                self.place_limit_order(symbol, 'sell', amount, stop_loss_price)
                
                # Тейк-профит (продажа выше входа)
                self.place_limit_order(symbol, 'sell', amount, take_profit_price)
                
            else:  # side == 'sell'
                # Для SHORT позиции
                stop_loss_price = entry_price * (1 + self.stop_loss_pct)
                take_profit_price = entry_price * (1 - self.take_profit_pct)
                
                # Стоп-лосс (покупка выше входа)
                self.place_limit_order(symbol, 'buy', amount, stop_loss_price)
                
                # Тейк-профит (покупка ниже входа)
                self.place_limit_order(symbol, 'buy', amount, take_profit_price)
            
            logger.info(f"SL/TP установлены для {symbol}: SL=${stop_loss_price:.2f}, TP=${take_profit_price:.2f}")
            
        except Exception as e:
            logger.error(f"Ошибка установки SL/TP: {e}")
    
    def execute_trade_signal(self, symbol: str, signal_data: Dict) -> bool:
        """
        Выполняет торговый сигнал
        """
        try:
            signal = signal_data.get('action', 'HOLD')
            confidence = signal_data.get('confidence', 0.0)
            current_price = signal_data.get('current_price', 0.0)
            
            # Проверяем минимальную уверенность
            if confidence < self.min_confidence:
                logger.info(f"Сигнал {signal} отклонен: низкая уверенность ({confidence:.3f})")
                return False
            
            # Проверяем, есть ли уже позиция
            if symbol in self.positions:
                logger.info(f"Позиция по {symbol} уже открыта")
                return False
            
            if signal == 'BUY':
                # Открываем LONG позицию
                position_size_usd = self.calculate_position_size(symbol, confidence)
                if position_size_usd <= 0:
                    return False
                
                amount = position_size_usd / current_price
                
                order = self.place_market_order(symbol, 'buy', amount)
                if order:
                    # Сохраняем информацию о позиции
                    self.positions[symbol] = {
                        'side': 'LONG',
                        'amount': amount,
                        'entry_price': current_price,
                        'entry_time': datetime.now(),
                        'order_id': order['id'],
                        'confidence': confidence
                    }
                    
                    # Устанавливаем SL/TP
                    self.set_stop_loss_take_profit(symbol, 'buy', current_price, amount)
                    
                    # Обновляем статистику
                    self.stats['total_trades'] += 1
                    
                    logger.info(f"🟢 LONG позиция открыта: {symbol} @ ${current_price:.2f}")
                    return True
            
            elif signal == 'SELL':
                # Открываем SHORT позицию (если поддерживается)
                position_size_usd = self.calculate_position_size(symbol, confidence)
                if position_size_usd <= 0:
                    return False
                
                amount = position_size_usd / current_price
                
                order = self.place_market_order(symbol, 'sell', amount)
                if order:
                    self.positions[symbol] = {
                        'side': 'SHORT',
                        'amount': amount,
                        'entry_price': current_price,
                        'entry_time': datetime.now(),
                        'order_id': order['id'],
                        'confidence': confidence
                    }
                    
                    self.set_stop_loss_take_profit(symbol, 'sell', current_price, amount)
                    self.stats['total_trades'] += 1
                    
                    logger.info(f"🔴 SHORT позиция открыта: {symbol} @ ${current_price:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка выполнения сигнала: {e}")
            return False
    
    def check_positions(self):
        """
        Проверяет текущие позиции и управляет ими
        """
        try:
            for symbol, position in list(self.positions.items()):
                current_price = self.get_current_price(symbol)
                if current_price is None:
                    continue
                
                entry_price = position['entry_price']
                side = position['side']
                
                # Рассчитываем P&L
                if side == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:  # SHORT
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                
                # Проверяем условия закрытия
                should_close = False
                close_reason = ""
                
                # Проверка времени (максимум 24 часа)
                time_open = datetime.now() - position['entry_time']
                if time_open.total_seconds() > 24 * 3600:
                    should_close = True
                    close_reason = "Превышено максимальное время"
                
                # Проверка убытков
                if pnl_pct <= -self.stop_loss_pct * 100:
                    should_close = True
                    close_reason = f"Stop Loss ({pnl_pct:.2f}%)"
                
                # Проверка прибыли
                if pnl_pct >= self.take_profit_pct * 100:
                    should_close = True
                    close_reason = f"Take Profit ({pnl_pct:.2f}%)"
                
                if should_close:
                    self.close_position(symbol, close_reason)
                else:
                    logger.info(f"Позиция {symbol}: {pnl_pct:.2f}% P&L")
        
        except Exception as e:
            logger.error(f"Ошибка проверки позиций: {e}")
    
    def close_position(self, symbol: str, reason: str = "Manual"):
        """
        Закрывает позицию
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"Позиция {symbol} не найдена")
                return False
            
            position = self.positions[symbol]
            side = 'sell' if position['side'] == 'LONG' else 'buy'
            amount = position['amount']
            
            # Закрываем позицию рыночным ордером
            order = self.place_market_order(symbol, side, amount)
            
            if order:
                # Рассчитываем финальный P&L
                current_price = self.get_current_price(symbol)
                entry_price = position['entry_price']
                
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    pnl_usd = (current_price - entry_price) * amount
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    pnl_usd = (entry_price - current_price) * amount
                
                # Сохраняем в историю
                trade_record = {
                    'symbol': symbol,
                    'side': position['side'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'amount': amount,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'confidence': position['confidence'],
                    'close_reason': reason
                }
                
                self.trades_history.append(trade_record)
                
                # Обновляем статистику
                self.stats['total_profit'] += pnl_usd
                if pnl_pct > 0:
                    self.stats['winning_trades'] += 1
                
                # Удаляем позицию
                del self.positions[symbol]
                
                logger.info(f"✅ Позиция закрыта: {symbol} | {pnl_pct:.2f}% | ${pnl_usd:.2f} | {reason}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции {symbol}: {e}")
            return False
    
    def analyze_and_trade(self, symbol: str):
        """
        Анализирует рынок и принимает торговое решение
        """
        try:
            # Получаем рыночные данные
            df = self.get_market_data(symbol, timeframe='1h', limit=100)
            if df is None or len(df) < 50:
                logger.warning(f"Недостаточно данных для анализа {symbol}")
                return
            
            # Подготавливаем данные с индикаторами
            from indicators.indicators import calculate_rsi, calculate_macd, calculate_sma, calculate_ema
            from indicators.enhanced_indicators import calculate_all_enhanced_indicators
            
            df = calculate_rsi(df)
            df = calculate_macd(df)
            df = calculate_sma(df, period=20)
            df = calculate_sma(df, period=50)
            df = calculate_ema(df, period=20)
            df = calculate_ema(df, period=50)
            
            # Генерируем сигналы с помощью ML
            if self.signal_generator:
                df_with_signals = self.signal_generator.combine_signals(df)
                recommendation = self.signal_generator.generate_trading_recommendation(df_with_signals, position_size=1000)
                
                if recommendation and recommendation['action'] != 'HOLD':
                    logger.info(f"🎯 Сигнал для {symbol}: {recommendation['action']} (уверенность: {recommendation['confidence']:.3f})")
                    
                    # Выполняем сделку
                    success = self.execute_trade_signal(symbol, recommendation)
                    if success:
                        logger.info(f"✅ Сделка выполнена: {symbol}")
                    else:
                        logger.info(f"⚠️ Сделка отклонена: {symbol}")
                else:
                    logger.info(f"📊 {symbol}: Нет сильных сигналов")
            
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
    
    async def trading_loop(self, interval_minutes: int = 15):
        """
        Основной торговый цикл
        """
        logger.info(f"🚀 Запуск торгового цикла (интервал: {interval_minutes} мин)")
        
        while self.is_trading:
            try:
                cycle_start = datetime.now()
                logger.info(f"📊 Цикл анализа: {cycle_start.strftime('%H:%M:%S')}")
                
                # Проверяем существующие позиции
                self.check_positions()
                
                # Анализируем каждый символ
                for symbol in self.trading_symbols:
                    if symbol not in self.positions:  # Если нет открытой позиции
                        self.analyze_and_trade(symbol)
                    
                    # Пауза между символами
                    await asyncio.sleep(2)
                
                # Выводим статистику
                self.print_statistics()
                
                # Пауза до следующего цикла
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, interval_minutes * 60 - cycle_duration)
                
                if sleep_time > 0:
                    logger.info(f"⏱️ Пауза до следующего цикла: {sleep_time:.0f} сек")
                    await asyncio.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("🛑 Получен сигнал остановки")
                break
            except Exception as e:
                logger.error(f"❌ Ошибка в торговом цикле: {e}")
                await asyncio.sleep(60)  # Пауза при ошибке
        
        logger.info("🏁 Торговый цикл завершен")
    
    def print_statistics(self):
        """
        Выводит текущую статистику торговли
        """
        total_trades = self.stats['total_trades']
        winning_trades = self.stats['winning_trades']
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = self.stats['total_profit']
        
        uptime = datetime.now() - self.stats['start_time']
        
        logger.info("📈 СТАТИСТИКА ТОРГОВЛИ:")
        logger.info(f"  Время работы: {uptime}")
        logger.info(f"  Всего сделок: {total_trades}")
        logger.info(f"  Прибыльных: {winning_trades} ({win_rate:.1f}%)")
        logger.info(f"  Общая прибыль: ${total_profit:.2f}")
        logger.info(f"  Открытых позиций: {len(self.positions)}")
    
    def start_trading(self):
        """
        Запускает торговую систему
        """
        if not self.exchange:
            logger.error("❌ Биржа не инициализирована")
            return False
        
        self.is_trading = True
        logger.info("🚀 ТОРГОВАЯ СИСТЕМА ЗАПУЩЕНА!")
        
        # Запускаем асинхронный цикл
        try:
            asyncio.run(self.trading_loop())
        except KeyboardInterrupt:
            logger.info("🛑 Остановка по запросу пользователя")
        finally:
            self.stop_trading()
        
        return True
    
    def stop_trading(self):
        """
        Останавливает торговую систему
        """
        self.is_trading = False
        
        # Закрываем все открытые позиции
        for symbol in list(self.positions.keys()):
            self.close_position(symbol, "Система остановлена")
        
        logger.info("🛑 ТОРГОВАЯ СИСТЕМА ОСТАНОВЛЕНА")
        self.print_statistics()
    
    def save_trading_log(self, filepath: str):
        """
        Сохраняет лог торговли
        """
        try:
            log_data = {
                'statistics': self.stats,
                'positions': self.positions,
                'trades_history': self.trades_history,
                'config': {
                    'max_position_size': self.max_position_size,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'min_confidence': self.min_confidence
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 Лог торговли сохранен: {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения лога: {e}")
