# backtesting/enhanced_backtester.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class EnhancedBacktester:
    """
    Улучшенный бэктестер с расширенными метриками и возможностью
    тестирования множественных стратегий
    """
    
    def __init__(self, initial_balance=10000, commission_rate=0.001):
        """
        Инициализация бэктестера
        
        Args:
            initial_balance: Начальный баланс
            commission_rate: Комиссия за сделку
        """
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.reset()
        
    def reset(self):
        """Сброс состояния бэктестера"""
        self.balance = self.initial_balance
        self.trades = []
        self.portfolio_values = []
        self.drawdowns = []
        self.positions = []
        self.current_position = None
        
    def simulate_trade(self, entry_price, exit_price, position_size, exit_type, 
                      entry_time, exit_time, entry_index, exit_index):
        """
        Симуляция сделки с расширенной информацией
        
        Args:
            entry_price: Цена входа
            exit_price: Цена выхода
            position_size: Размер позиции
            exit_type: Тип выхода (SL, TP, SELL, BUY, EOD)
            entry_time: Время входа
            exit_time: Время выхода
            entry_index: Индекс входа
            exit_index: Индекс выхода
        """
        # Рассчитываем комиссию на вход и выход
        entry_commission = entry_price * position_size * self.commission_rate
        exit_commission = exit_price * position_size * self.commission_rate
        total_commission = entry_commission + exit_commission
        
        # Прибыль с учётом комиссии
        gross_profit = (exit_price - entry_price) * position_size
        net_profit = gross_profit - total_commission
        
        # Обновляем баланс
        self.balance += net_profit
        
        # Рассчитываем длительность сделки
        if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
            duration = (exit_time - entry_time).total_seconds() / 3600  # в часах
        else:
            duration = exit_index - entry_index  # в барах
        
        # Информация о сделке
        trade_info = {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_index": entry_index,
            "exit_index": exit_index,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": position_size,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "commission": total_commission,
            "balance": self.balance,
            "exit_type": exit_type,
            "duration": duration,
            "return_pct": (net_profit / (entry_price * position_size)) * 100
        }
        
        self.trades.append(trade_info)
        
        # Обновляем портфель
        self.portfolio_values.append({
            "time": exit_time,
            "balance": self.balance,
            "equity": self.balance + (self.current_position['unrealized_pnl'] if self.current_position else 0)
        })
        
        logger.info(f"Сделка ({exit_type}) проведена: {trade_info}")
        return trade_info
    
    def run_backtest(self, df, signal_column="signal", strategy_name="Strategy"):
        """
        Запуск бэктеста с расширенной логикой
        
        Args:
            df: DataFrame с данными и сигналами
            signal_column: Колонка с сигналами
            strategy_name: Название стратегии
        """
        logger.info(f"Запуск бэктеста для стратегии: {strategy_name}")
        self.reset()
        
        in_position = False
        entry_price = None
        position_size = 0.0
        stop_price = None
        take_price = None
        entry_time = None
        entry_index = None
        
        # Добавляем временные метки если их нет
        if 'timestamp' not in df.columns and 'start_at' in df.columns:
            df['timestamp'] = df['start_at']
        elif 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        
        for index, row in df.iterrows():
            price = row["close"]
            sig = row[signal_column]
            current_time = row.get('timestamp', index)
            
            if in_position:
                # Проверяем стоп-лосс
                if row["low"] <= stop_price:
                    self.simulate_trade(
                        entry_price, stop_price, position_size, "SL",
                        entry_time, current_time, entry_index, index
                    )
                    in_position = False
                    self.current_position = None
                    logger.info(f"SL hit at {stop_price:.2f}")
                    continue
                
                # Проверяем тейк-профит
                if row["high"] >= take_price:
                    self.simulate_trade(
                        entry_price, take_price, position_size, "TP",
                        entry_time, current_time, entry_index, index
                    )
                    in_position = False
                    self.current_position = None
                    logger.info(f"TP hit at {take_price:.2f}")
                    continue
                
                # Закрытие по сигналу
                if sig == "SELL":
                    self.simulate_trade(
                        entry_price, price, position_size, "SELL",
                        entry_time, current_time, entry_index, index
                    )
                    in_position = False
                    self.current_position = None
                    logger.info(f"Close by SELL at {price:.2f}")
                    continue
                
                # Обновляем unrealized PnL
                if self.current_position:
                    self.current_position['unrealized_pnl'] = (price - entry_price) * position_size
            
            # Открываем позицию по сигналу BUY
            if sig == "BUY" and not in_position:
                entry_price = price
                entry_time = current_time
                entry_index = index
                
                # Рассчитываем размер позиции (упрощенно)
                risk_amount = self.balance * 0.01  # 1% риска
                if 'ATR' in df.columns:
                    stop_distance = row['ATR'] * 2
                else:
                    stop_distance = entry_price * 0.02  # 2% по умолчанию
                
                position_size = risk_amount / stop_distance
                stop_price = entry_price - stop_distance
                take_price = entry_price + stop_distance * 2  # 2:1 risk/reward
                
                in_position = True
                self.balance -= entry_price * position_size
                
                # Создаем текущую позицию
                self.current_position = {
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'stop_price': stop_price,
                    'take_price': take_price,
                    'unrealized_pnl': 0
                }
                
                logger.info(
                    f"OPEN ▶ price={entry_price:.2f}, size={position_size:.6f}, "
                    f"SL={stop_price:.2f}, TP={take_price:.2f}"
                )
        
        # Закрываем открытую позицию в конце
        if in_position:
            last_close = df['close'].iloc[-1]
            last_time = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else df.index[-1]
            self.simulate_trade(
                entry_price, last_close, position_size, "EOD",
                entry_time, last_time, entry_index, len(df) - 1
            )
        
        logger.info(f"Бэктест завершен. Финальный баланс: {self.balance:.2f}")
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Расчет расширенных метрик производительности
        
        Returns:
            Словарь с метриками
        """
        if not self.trades:
            logger.warning("Нет сделок для расчета метрик")
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Базовые метрики
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_profit'] > 0])
        losing_trades = len(trades_df[trades_df['net_profit'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Прибыль/убыток
        total_profit = trades_df['net_profit'].sum()
        gross_profit = trades_df[trades_df['net_profit'] > 0]['net_profit'].sum()
        gross_loss = abs(trades_df[trades_df['net_profit'] < 0]['net_profit'].sum())
        
        # Средние значения
        avg_win = trades_df[trades_df['net_profit'] > 0]['net_profit'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_profit'] < 0]['net_profit'].mean() if losing_trades > 0 else 0
        
        # Profit Factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Максимальная просадка
        cumulative_returns = trades_df['net_profit'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        max_drawdown = abs(drawdown.min())
        max_drawdown_pct = (max_drawdown / self.initial_balance) * 100
        
        # Sharpe Ratio (упрощенно)
        returns = trades_df['return_pct'] / 100
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino_ratio = returns.mean() / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar Ratio
        total_return_pct = (total_profit / self.initial_balance) * 100
        calmar_ratio = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # Длительность сделок
        avg_duration = trades_df['duration'].mean() if 'duration' in trades_df.columns else 0
        
        # Создаем словарь метрик
        metrics = {
            'Strategy Name': 'Enhanced Strategy',
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate (%)': win_rate * 100,
            'Total Profit': total_profit,
            'Total Return (%)': total_return_pct,
            'Gross Profit': gross_profit,
            'Gross Loss': gross_loss,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Max Drawdown': max_drawdown,
            'Max Drawdown (%)': max_drawdown_pct,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Average Duration': avg_duration,
            'Initial Balance': self.initial_balance,
            'Final Balance': self.balance
        }
        
        logger.info("Метрики производительности рассчитаны")
        return metrics
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Возвращает DataFrame с информацией о сделках"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_portfolio_values(self) -> pd.DataFrame:
        """Возвращает DataFrame с значениями портфеля"""
        if not self.portfolio_values:
            return pd.DataFrame()
        return pd.DataFrame(self.portfolio_values)
    
    def plot_equity_curve(self, save_path=None):
        """График кривой доходности"""
        if not self.portfolio_values:
            logger.warning("Нет данных портфеля для построения графика")
            return
        
        portfolio_df = pd.DataFrame(self.portfolio_values)
        
        plt.figure(figsize=(12, 8))
        plt.plot(portfolio_df['time'], portfolio_df['balance'], linewidth=2, label='Balance')
        plt.plot(portfolio_df['time'], portfolio_df['equity'], linewidth=2, label='Equity', alpha=0.7)
        
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"График equity curve сохранен в {save_path}")
        
        plt.show()
    
    def plot_drawdown(self, save_path=None):
        """График просадки"""
        if not self.trades:
            logger.warning("Нет сделок для построения графика просадки")
            return
        
        trades_df = pd.DataFrame(self.trades)
        cumulative_returns = trades_df['net_profit'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        plt.plot(drawdown, color='red', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Trade Number')
        plt.ylabel('Drawdown (USDT)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"График просадки сохранен в {save_path}")
        
        plt.show()

class StrategyComparator:
    """
    Класс для сравнения множественных стратегий
    """
    
    def __init__(self, initial_balance=10000, commission_rate=0.001):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.results = {}
        
    def compare_strategies(self, df, strategies_dict):
        """
        Сравнение множественных стратегий
        
        Args:
            df: DataFrame с данными
            strategies_dict: Словарь {название: функция_стратегии}
        """
        logger.info(f"Начинаю сравнение {len(strategies_dict)} стратегий")
        
        for strategy_name, strategy_func in strategies_dict.items():
            try:
                logger.info(f"Тестирую стратегию: {strategy_name}")
                
                # Применяем стратегию
                df_with_signals = strategy_func(df.copy())
                
                # Запускаем бэктест
                backtester = EnhancedBacktester(
                    initial_balance=self.initial_balance,
                    commission_rate=self.commission_rate
                )
                
                metrics = backtester.run_backtest(df_with_signals, strategy_name=strategy_name)
                
                # Сохраняем результаты
                self.results[strategy_name] = {
                    'metrics': metrics,
                    'trades': backtester.get_trades_dataframe(),
                    'portfolio': backtester.get_portfolio_values(),
                    'backtester': backtester
                }
                
                logger.info(f"Стратегия {strategy_name} протестирована успешно")
                
            except Exception as e:
                logger.error(f"Ошибка при тестировании стратегии {strategy_name}: {str(e)}")
                continue
        
        logger.info(f"Сравнение завершено. Протестировано {len(self.results)} стратегий")
        return self.results
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """Создает сводную таблицу сравнения стратегий"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for strategy_name, result in self.results.items():
            metrics = result['metrics']
            summary_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': metrics.get('Total Return (%)', 0),
                'Win Rate (%)': metrics.get('Win Rate (%)', 0),
                'Total Trades': metrics.get('Total Trades', 0),
                'Profit Factor': metrics.get('Profit Factor', 0),
                'Max Drawdown (%)': metrics.get('Max Drawdown (%)', 0),
                'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
                'Sortino Ratio': metrics.get('Sortino Ratio', 0),
                'Calmar Ratio': metrics.get('Calmar Ratio', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total Return (%)', ascending=False)
        
        return summary_df
    
    def plot_comparison(self, metric='Total Return (%)', save_path=None):
        """График сравнения стратегий по выбранной метрике"""
        if not self.results:
            logger.warning("Нет результатов для сравнения")
            return
        
        summary_df = self.get_comparison_summary()
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(summary_df['Strategy'], summary_df[metric], alpha=0.7)
        
        # Цвета для столбцов
        colors = ['green' if x > 0 else 'red' for x in summary_df[metric]]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title(f'Strategy Comparison: {metric}', fontsize=14, fontweight='bold')
        plt.xlabel('Strategy')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, summary_df[metric]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"График сравнения сохранен в {save_path}")
        
        plt.show()
