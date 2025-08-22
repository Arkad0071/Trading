#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система мониторинга торговли в реальном времени
Отслеживает производительность, риски и отправляет уведомления
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import json
import os
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """
    Система мониторинга торговли в реальном времени
    """
    
    def __init__(self, trader=None, notification_callbacks: List[Callable] = None):
        """
        Инициализация монитора
        
        Args:
            trader: Экземпляр LiveTrader
            notification_callbacks: Список функций для отправки уведомлений
        """
        self.trader = trader
        self.notification_callbacks = notification_callbacks or []
        
        # Параметры мониторинга
        self.monitoring_interval = 60  # Проверка каждые 60 секунд
        self.max_drawdown_alert = 0.05  # Уведомление при просадке > 5%
        self.max_daily_loss = 0.10      # Максимальная дневная потеря 10%
        self.max_open_positions = 5     # Максимум открытых позиций
        
        # Состояние системы
        self.is_monitoring = False
        self.alerts_sent = set()
        self.daily_stats = {}
        self.performance_metrics = {}
        
        # История производительности
        self.equity_curve = []
        self.drawdown_history = []
        self.trade_metrics = []
        
    def add_notification_callback(self, callback: Callable):
        """
        Добавляет функцию для отправки уведомлений
        """
        self.notification_callbacks.append(callback)
        logger.info("Добавлен новый канал уведомлений")
    
    def calculate_portfolio_metrics(self) -> Dict:
        """
        Рассчитывает метрики портфеля
        """
        if not self.trader or not self.trader.trades_history:
            return {}
        
        try:
            trades_df = pd.DataFrame(self.trader.trades_history)
            
            # Базовые метрики
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_return = trades_df['pnl_usd'].sum()
            avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if (total_trades - winning_trades) > 0 else 0
            
            # Расчет максимальной просадки
            equity_curve = trades_df['pnl_usd'].cumsum()
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            
            # Коэффициент Шарпа (упрощенный)
            returns = trades_df['pnl_pct'] / 100
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Profit Factor
            gross_profit = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum()
            gross_loss = abs(trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            metrics = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return_usd': total_return,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor,
                'current_equity': self.trader.initial_balance + total_return,
                'open_positions': len(self.trader.positions) if self.trader else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик портфеля: {e}")
            return {}
    
    def check_risk_limits(self, metrics: Dict) -> List[Dict]:
        """
        Проверяет лимиты риска и возвращает список предупреждений
        """
        alerts = []
        
        try:
            # Проверка максимальной просадки
            max_drawdown = abs(metrics.get('max_drawdown_pct', 0))
            if max_drawdown > self.max_drawdown_alert * 100:
                alert = {
                    'type': 'HIGH_DRAWDOWN',
                    'level': 'WARNING',
                    'message': f"Высокая просадка: {max_drawdown:.2f}%",
                    'value': max_drawdown,
                    'threshold': self.max_drawdown_alert * 100,
                    'timestamp': datetime.now()
                }
                alerts.append(alert)
            
            # Проверка количества открытых позиций
            open_positions = metrics.get('open_positions', 0)
            if open_positions > self.max_open_positions:
                alert = {
                    'type': 'TOO_MANY_POSITIONS',
                    'level': 'WARNING',
                    'message': f"Слишком много открытых позиций: {open_positions}",
                    'value': open_positions,
                    'threshold': self.max_open_positions,
                    'timestamp': datetime.now()
                }
                alerts.append(alert)
            
            # Проверка дневной потери
            today = datetime.now().date()
            if today in self.daily_stats:
                daily_return = self.daily_stats[today].get('total_return', 0)
                daily_return_pct = (daily_return / self.trader.initial_balance) * 100
                
                if daily_return_pct < -self.max_daily_loss * 100:
                    alert = {
                        'type': 'DAILY_LOSS_LIMIT',
                        'level': 'CRITICAL',
                        'message': f"Превышен дневной лимит потерь: {daily_return_pct:.2f}%",
                        'value': daily_return_pct,
                        'threshold': -self.max_daily_loss * 100,
                        'timestamp': datetime.now()
                    }
                    alerts.append(alert)
            
            # Проверка винрейта
            win_rate = metrics.get('win_rate', 0)
            if metrics.get('total_trades', 0) >= 10 and win_rate < 30:
                alert = {
                    'type': 'LOW_WIN_RATE',
                    'level': 'WARNING',
                    'message': f"Низкий винрейт: {win_rate:.1f}%",
                    'value': win_rate,
                    'threshold': 30,
                    'timestamp': datetime.now()
                }
                alerts.append(alert)
            
            # Проверка подключения к бирже
            if self.trader and self.trader.exchange:
                try:
                    # Простая проверка подключения
                    self.trader.exchange.fetch_ticker('BTC/USDT')
                except Exception:
                    alert = {
                        'type': 'CONNECTION_ERROR',
                        'level': 'CRITICAL',
                        'message': "Потеряно подключение к бирже",
                        'timestamp': datetime.now()
                    }
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Ошибка проверки лимитов риска: {e}")
            return []
    
    def send_notification(self, alert: Dict):
        """
        Отправляет уведомление через зарегистрированные каналы
        """
        # Проверяем, не отправляли ли уже это уведомление
        alert_key = f"{alert['type']}_{alert.get('value', '')}"
        
        if alert_key in self.alerts_sent:
            return
        
        # Добавляем в отправленные
        self.alerts_sent.add(alert_key)
        
        # Отправляем через все каналы
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Ошибка отправки уведомления: {e}")
        
        logger.info(f"📢 Уведомление отправлено: {alert['message']}")
    
    def update_daily_stats(self):
        """
        Обновляет дневную статистику
        """
        try:
            today = datetime.now().date()
            
            if not self.trader or not self.trader.trades_history:
                return
            
            # Фильтруем сегодняшние сделки
            today_trades = [
                trade for trade in self.trader.trades_history
                if trade['exit_time'].date() == today
            ]
            
            if today_trades:
                total_return = sum(trade['pnl_usd'] for trade in today_trades)
                total_trades = len(today_trades)
                winning_trades = len([t for t in today_trades if t['pnl_usd'] > 0])
                
                self.daily_stats[today] = {
                    'total_return': total_return,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0
                }
            
        except Exception as e:
            logger.error(f"Ошибка обновления дневной статистики: {e}")
    
    def generate_status_report(self) -> Dict:
        """
        Генерирует отчет о текущем состоянии
        """
        try:
            metrics = self.calculate_portfolio_metrics()
            alerts = self.check_risk_limits(metrics)
            
            # Информация о текущих позициях
            positions_info = []
            if self.trader and self.trader.positions:
                for symbol, position in self.trader.positions.items():
                    current_price = self.trader.get_current_price(symbol)
                    if current_price:
                        entry_price = position['entry_price']
                        if position['side'] == 'LONG':
                            pnl_pct = (current_price - entry_price) / entry_price * 100
                        else:
                            pnl_pct = (entry_price - current_price) / entry_price * 100
                        
                        positions_info.append({
                            'symbol': symbol,
                            'side': position['side'],
                            'entry_price': entry_price,
                            'current_price': current_price,
                            'pnl_pct': pnl_pct,
                            'confidence': position.get('confidence', 0),
                            'duration': datetime.now() - position['entry_time']
                        })
            
            report = {
                'timestamp': datetime.now(),
                'system_status': 'RUNNING' if self.is_monitoring else 'STOPPED',
                'portfolio_metrics': metrics,
                'active_alerts': alerts,
                'open_positions': positions_info,
                'daily_stats': self.daily_stats.get(datetime.now().date(), {}),
                'trader_status': {
                    'is_trading': self.trader.is_trading if self.trader else False,
                    'last_update': self.trader.last_update if self.trader else None,
                    'trading_symbols': self.trader.trading_symbols if self.trader else []
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return {'error': str(e)}
    
    def log_performance_snapshot(self):
        """
        Записывает снимок производительности
        """
        try:
            metrics = self.calculate_portfolio_metrics()
            
            snapshot = {
                'timestamp': datetime.now(),
                'equity': metrics.get('current_equity', 0),
                'total_return': metrics.get('total_return_usd', 0),
                'max_drawdown': metrics.get('max_drawdown_pct', 0),
                'win_rate': metrics.get('win_rate', 0),
                'open_positions': metrics.get('open_positions', 0),
                'total_trades': metrics.get('total_trades', 0)
            }
            
            self.equity_curve.append(snapshot)
            
            # Ограничиваем историю (последние 1000 записей)
            if len(self.equity_curve) > 1000:
                self.equity_curve = self.equity_curve[-1000:]
            
        except Exception as e:
            logger.error(f"Ошибка записи снимка производительности: {e}")
    
    async def monitoring_loop(self):
        """
        Основной цикл мониторинга
        """
        logger.info(f"🔍 Запуск мониторинга (интервал: {self.monitoring_interval} сек)")
        
        while self.is_monitoring:
            try:
                # Обновляем статистику
                self.update_daily_stats()
                
                # Записываем снимок производительности
                self.log_performance_snapshot()
                
                # Рассчитываем метрики
                metrics = self.calculate_portfolio_metrics()
                
                # Проверяем лимиты риска
                alerts = self.check_risk_limits(metrics)
                
                # Отправляем критические уведомления
                for alert in alerts:
                    if alert['level'] == 'CRITICAL':
                        self.send_notification(alert)
                    elif alert['level'] == 'WARNING':
                        # Отправляем предупреждения реже
                        if datetime.now().minute % 15 == 0:
                            self.send_notification(alert)
                
                # Очищаем старые уведомления (каждые 15 минут)
                if datetime.now().minute % 15 == 0:
                    self.alerts_sent.clear()
                
                # Логируем состояние системы
                if metrics:
                    total_return = metrics.get('total_return_usd', 0)
                    win_rate = metrics.get('win_rate', 0)
                    open_positions = metrics.get('open_positions', 0)
                    
                    logger.info(f"📊 Состояние: ${total_return:.2f} | {win_rate:.1f}% | {open_positions} позиций")
                
                # Пауза до следующей проверки
                await asyncio.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Получен сигнал остановки мониторинга")
                break
            except Exception as e:
                logger.error(f"❌ Ошибка в цикле мониторинга: {e}")
                await asyncio.sleep(60)  # Пауза при ошибке
        
        logger.info("🏁 Мониторинг завершен")
    
    def start_monitoring(self):
        """
        Запускает мониторинг
        """
        if not self.trader:
            logger.error("❌ Трейдер не установлен")
            return False
        
        self.is_monitoring = True
        logger.info("🔍 МОНИТОРИНГ ЗАПУЩЕН!")
        
        try:
            asyncio.run(self.monitoring_loop())
        except KeyboardInterrupt:
            logger.info("🛑 Остановка мониторинга по запросу")
        finally:
            self.stop_monitoring()
        
        return True
    
    def stop_monitoring(self):
        """
        Останавливает мониторинг
        """
        self.is_monitoring = False
        logger.info("🛑 МОНИТОРИНГ ОСТАНОВЛЕН")
    
    def save_monitoring_data(self, filepath: str):
        """
        Сохраняет данные мониторинга
        """
        try:
            monitoring_data = {
                'equity_curve': self.equity_curve,
                'daily_stats': self.daily_stats,
                'performance_metrics': self.performance_metrics,
                'alerts_history': list(self.alerts_sent),
                'config': {
                    'monitoring_interval': self.monitoring_interval,
                    'max_drawdown_alert': self.max_drawdown_alert,
                    'max_daily_loss': self.max_daily_loss,
                    'max_open_positions': self.max_open_positions
                },
                'last_report': self.generate_status_report()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(monitoring_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 Данные мониторинга сохранены: {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения данных мониторинга: {e}")

# Функции для уведомлений
def console_notification(alert: Dict):
    """
    Выводит уведомление в консоль
    """
    level_emoji = {'CRITICAL': '🚨', 'WARNING': '⚠️', 'INFO': 'ℹ️'}
    emoji = level_emoji.get(alert.get('level', 'INFO'), 'ℹ️')
    
    print(f"\n{emoji} УВЕДОМЛЕНИЕ {emoji}")
    print(f"Тип: {alert.get('type', 'Unknown')}")
    print(f"Сообщение: {alert.get('message', '')}")
    print(f"Время: {alert.get('timestamp', datetime.now()).strftime('%H:%M:%S')}")
    if 'value' in alert and 'threshold' in alert:
        print(f"Значение: {alert['value']} (лимит: {alert['threshold']})")
    print("-" * 50)

def telegram_notification(alert: Dict, bot_token: str = None, chat_id: str = None):
    """
    Отправляет уведомление в Telegram (заглушка)
    """
    # TODO: Реализовать отправку через Telegram Bot API
    pass

def email_notification(alert: Dict, email_config: Dict = None):
    """
    Отправляет уведомление по email (заглушка)
    """
    # TODO: Реализовать отправку email
    pass
