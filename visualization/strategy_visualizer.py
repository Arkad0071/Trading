# visualization/strategy_visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class StrategyVisualizer:
    def __init__(self, df, strategy_name="Strategy"):
        """
        Инициализация визуализатора стратегии
        
        Args:
            df: DataFrame с данными (OHLCV + индикаторы + сигналы)
            strategy_name: Название стратегии для отображения
        """
        self.df = df.copy()
        self.strategy_name = strategy_name
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """Настройка стиля графиков"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 10
        
    def plot_strategy_overview(self, save_path=None):
        """
        Основной график стратегии с ценой, индикаторами и сигналами
        
        Args:
            save_path: Путь для сохранения графика
        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 16), 
                                                   gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # График 1: Цена и сигналы
        self._plot_price_and_signals(ax1)
        
        # График 2: Объем
        self._plot_volume(ax2)
        
        # График 3: RSI
        self._plot_rsi(ax3)
        
        # График 4: MACD
        self._plot_macd(ax4)
        
        plt.suptitle(f'{self.strategy_name} - Анализ стратегии', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"График сохранен в {save_path}")
        
        plt.show()
        
    def _plot_price_and_signals(self, ax):
        """График цены с сигналами и индикаторами"""
        # Цена
        ax.plot(self.df.index, self.df['close'], label='Цена закрытия', linewidth=1.5, color='black')
        
        # Bollinger Bands (если есть)
        if 'BB_upper' in self.df.columns:
            ax.plot(self.df.index, self.df['BB_upper'], label='BB Upper', alpha=0.7, color='red')
            ax.plot(self.df.index, self.df['BB_lower'], label='BB Lower', alpha=0.7, color='red')
            ax.fill_between(self.df.index, self.df['BB_upper'], self.df['BB_lower'], alpha=0.1, color='red')
        
        # Moving Averages (если есть)
        if 'SMA_20' in self.df.columns:
            ax.plot(self.df.index, self.df['SMA_20'], label='SMA 20', alpha=0.8, color='blue')
        if 'EMA_20' in self.df.columns:
            ax.plot(self.df.index, self.df['EMA_20'], label='EMA 20', alpha=0.8, color='orange')
            
        # Сигналы
        buy_signals = self.df[self.df['signal'] == 'BUY']
        sell_signals = self.df[self.df['signal'] == 'SELL']
        
        ax.scatter(buy_signals.index, buy_signals['close'], 
                  marker='^', color='green', s=100, label='BUY', alpha=0.8)
        ax.scatter(sell_signals.index, sell_signals['close'], 
                  marker='v', color='red', s=100, label='SELL', alpha=0.8)
        
        ax.set_title('Цена и торговые сигналы')
        ax.set_ylabel('Цена (USDT)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_volume(self, ax):
        """График объема"""
        colors = ['green' if close > open else 'red' 
                 for close, open in zip(self.df['close'], self.df['open'])]
        
        ax.bar(self.df.index, self.df['volume'], color=colors, alpha=0.7)
        ax.set_title('Объем торгов')
        ax.set_ylabel('Объем')
        ax.grid(True, alpha=0.3)
        
    def _plot_rsi(self, ax):
        """График RSI"""
        if 'RSI' in self.df.columns:
            ax.plot(self.df.index, self.df['RSI'], label='RSI', color='purple', linewidth=1.5)
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Перекупленность')
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Перепроданность')
            ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
            ax.fill_between(self.df.index, 70, 100, alpha=0.1, color='red')
            ax.fill_between(self.df.index, 0, 30, alpha=0.1, color='green')
            
        ax.set_title('RSI')
        ax.set_ylabel('RSI')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_macd(self, ax):
        """График MACD"""
        if 'MACD' in self.df.columns:
            ax.plot(self.df.index, self.df['MACD'], label='MACD', color='blue', linewidth=1.5)
            ax.plot(self.df.index, self.df['MACD_signal'], label='Signal', color='red', linewidth=1.5)
            ax.bar(self.df.index, self.df['MACD'] - self.df['MACD_signal'], 
                   alpha=0.3, color='gray', label='Histogram')
            
        ax.set_title('MACD')
        ax.set_ylabel('MACD')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_trade_analysis(self, trades_df, save_path=None):
        """
        Анализ сделок с детальной информацией
        
        Args:
            trades_df: DataFrame с информацией о сделках
            save_path: Путь для сохранения графика
        """
        if trades_df.empty:
            logger.warning("Нет данных о сделках для анализа")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # График 1: Прибыльность сделок
        self._plot_trade_profitability(ax1, trades_df)
        
        # График 2: Распределение прибыли/убытков
        self._plot_pnl_distribution(ax2, trades_df)
        
        # График 3: Кумулятивная прибыль
        self._plot_cumulative_pnl(ax3, trades_df)
        
        # График 4: Длительность сделок
        self._plot_trade_duration(ax4, trades_df)
        
        plt.suptitle(f'{self.strategy_name} - Анализ сделок', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Анализ сделок сохранен в {save_path}")
        
        plt.show()
        
    def _plot_trade_profitability(self, ax, trades_df):
        """График прибыльности сделок"""
        profits = trades_df[trades_df['net_profit'] > 0]['net_profit']
        losses = trades_df[trades_df['net_profit'] < 0]['net_profit']
        
        ax.bar(range(len(profits)), profits, color='green', alpha=0.7, label='Прибыль')
        ax.bar(range(len(losses)), losses, color='red', alpha=0.7, label='Убыток')
        
        ax.set_title('Прибыльность сделок')
        ax.set_xlabel('Номер сделки')
        ax.set_ylabel('Прибыль/Убыток (USDT)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_pnl_distribution(self, ax, trades_df):
        """Распределение прибыли/убытков"""
        ax.hist(trades_df['net_profit'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Безубыточность')
        
        ax.set_title('Распределение прибыли/убытков')
        ax.set_xlabel('Прибыль/Убыток (USDT)')
        ax.set_ylabel('Количество сделок')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_cumulative_pnl(self, ax, trades_df):
        """Кумулятивная прибыль"""
        cumulative_pnl = trades_df['net_profit'].cumsum()
        ax.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='blue')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        
        ax.set_title('Кумулятивная прибыль')
        ax.set_xlabel('Номер сделки')
        ax.set_ylabel('Кумулятивная прибыль (USDT)')
        ax.grid(True, alpha=0.3)
        
    def _plot_trade_duration(self, ax, trades_df):
        """Длительность сделок"""
        if 'duration' in trades_df.columns:
            ax.hist(trades_df['duration'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax.set_title('Длительность сделок')
            ax.set_xlabel('Длительность (часы)')
            ax.set_ylabel('Количество сделок')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Данные о длительности\nсделок отсутствуют', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Длительность сделок')
            
    def plot_performance_metrics(self, performance_metrics, save_path=None):
        """
        Отображение ключевых метрик производительности
        
        Args:
            performance_metrics: Словарь с метриками
            save_path: Путь для сохранения графика
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        metrics = list(performance_metrics.keys())
        values = list(performance_metrics.values())
        
        bars = ax.bar(metrics, values, color=['green' if v > 0 else 'red' for v in values], alpha=0.7)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom')
        
        ax.set_title(f'{self.strategy_name} - Ключевые метрики производительности', fontsize=14, fontweight='bold')
        ax.set_ylabel('Значение')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Метрики производительности сохранены в {save_path}")
        
        plt.show()
        
    def create_strategy_report(self, trades_df, performance_metrics, save_dir="reports"):
        """
        Создание полного отчета по стратегии
        
        Args:
            trades_df: DataFrame с сделками
            performance_metrics: Словарь с метриками
            save_dir: Директория для сохранения отчетов
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name_clean = self.strategy_name.replace(" ", "_").replace("/", "_")
        
        # Сохраняем графики
        overview_path = f"{save_dir}/{strategy_name_clean}_overview_{timestamp}.png"
        trades_path = f"{save_dir}/{strategy_name_clean}_trades_{timestamp}.png"
        metrics_path = f"{save_dir}/{strategy_name_clean}_metrics_{timestamp}.png"
        
        self.plot_strategy_overview(overview_path)
        self.plot_trade_analysis(trades_df, trades_path)
        self.plot_performance_metrics(performance_metrics, metrics_path)
        
        # Создаем текстовый отчет
        report_path = f"{save_dir}/{strategy_name_clean}_report_{timestamp}.txt"
        self._create_text_report(trades_df, performance_metrics, report_path)
        
        logger.info(f"Полный отчет по стратегии сохранен в {save_dir}")
        
    def _create_text_report(self, trades_df, performance_metrics, report_path):
        """Создание текстового отчета"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"ОТЧЕТ ПО СТРАТЕГИИ: {self.strategy_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("КЛЮЧЕВЫЕ МЕТРИКИ:\n")
            f.write("-" * 20 + "\n")
            for metric, value in performance_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nАНАЛИЗ СДЕЛОК:\n")
            f.write("-" * 20 + "\n")
            if not trades_df.empty:
                f.write(f"Общее количество сделок: {len(trades_df)}\n")
                f.write(f"Прибыльных сделок: {len(trades_df[trades_df['net_profit'] > 0])}\n")
                f.write(f"Убыточных сделок: {len(trades_df[trades_df['net_profit'] < 0])}\n")
                f.write(f"Общая прибыль: {trades_df['net_profit'].sum():.2f} USDT\n")
                f.write(f"Средняя прибыль: {trades_df['net_profit'].mean():.2f} USDT\n")
                f.write(f"Максимальная прибыль: {trades_df['net_profit'].max():.2f} USDT\n")
                f.write(f"Максимальный убыток: {trades_df['net_profit'].min():.2f} USDT\n")
            else:
                f.write("Нет данных о сделках\n")
                
            f.write(f"\nОтчет создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
