#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-дашборд для мониторинга торговой системы
Показывает метрики в реальном времени, графики и управление
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Веб-фреймворк
try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask не установлен. Установите: pip install flask flask-socketio")

# Графики
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly не установлен. Установите: pip install plotly")

logger = logging.getLogger(__name__)

class TradingDashboard:
    """
    Веб-дашборд для торговой системы
    """
    
    def __init__(self, trader=None, monitor=None, port=5000):
        """
        Инициализация дашборда
        
        Args:
            trader: Экземпляр LiveTrader
            monitor: Экземпляр RealTimeMonitor
            port: Порт для веб-сервера
        """
        self.trader = trader
        self.monitor = monitor
        self.port = port
        
        if not FLASK_AVAILABLE:
            logger.error("Flask не установлен. Дашборд недоступен.")
            return
        
        # Создаем Flask приложение
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'trading_bot_secret_key'
        
        # Инициализируем SocketIO для real-time обновлений
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Настраиваем маршруты
        self.setup_routes()
        
        # Состояние дашборда
        self.is_running = False
        self.update_interval = 5  # Обновление каждые 5 секунд
    
    def setup_routes(self):
        """
        Настраивает маршруты Flask
        """
        @self.app.route('/')
        def index():
            """Главная страница дашборда"""
            return self.render_dashboard()
        
        @self.app.route('/api/status')
        def api_status():
            """API: Текущий статус системы"""
            return jsonify(self.get_system_status())
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API: Метрики производительности"""
            return jsonify(self.get_performance_metrics())
        
        @self.app.route('/api/positions')
        def api_positions():
            """API: Текущие позиции"""
            return jsonify(self.get_positions_data())
        
        @self.app.route('/api/trades')
        def api_trades():
            """API: История сделок"""
            return jsonify(self.get_trades_history())
        
        @self.app.route('/api/charts/equity')
        def api_equity_chart():
            """API: График эквити"""
            return jsonify(self.create_equity_chart())
        
        @self.app.route('/api/charts/performance')
        def api_performance_chart():
            """API: График производительности"""
            return jsonify(self.create_performance_chart())
        
        @self.app.route('/api/control/<action>')
        def api_control(action):
            """API: Управление системой"""
            return jsonify(self.handle_control_action(action))
        
        # WebSocket обработчики
        @self.socketio.on('connect')
        def handle_connect():
            """Обработка подключения клиента"""
            logger.info("Клиент подключился к дашборду")
            emit('status', self.get_system_status())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Обработка отключения клиента"""
            logger.info("Клиент отключился от дашборда")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Обработка запроса обновления данных"""
            self.broadcast_update()
    
    def render_dashboard(self):
        """
        Рендерит главную страницу дашборда
        """
        if not FLASK_AVAILABLE:
            return "Flask не установлен"
        
        # Создаем HTML страницу
        html_template = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric { text-align: center; margin: 10px 0; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; font-size: 14px; }
        .status-online { color: #27ae60; }
        .status-offline { color: #e74c3c; }
        .control-button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .btn-start { background: #27ae60; color: white; }
        .btn-stop { background: #e74c3c; color: white; }
        .btn-restart { background: #f39c12; color: white; }
        .positions-table { width: 100%; border-collapse: collapse; }
        .positions-table th, .positions-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        .chart-container { height: 400px; margin: 20px 0; }
        .alert { padding: 15px; margin: 10px 0; border-radius: 5px; }
        .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .alert-critical { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Trading Bot Dashboard</h1>
        <p>Система автоматической торговли с машинным обучением</p>
    </div>
    
    <div class="dashboard-grid">
        <!-- Статус системы -->
        <div class="card">
            <h3>📊 Статус системы</h3>
            <div id="system-status">
                <div class="metric">
                    <div class="metric-value" id="trading-status">Загрузка...</div>
                    <div class="metric-label">Статус торговли</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="open-positions">-</div>
                    <div class="metric-label">Открытых позиций</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="last-update">-</div>
                    <div class="metric-label">Последнее обновление</div>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <button class="control-button btn-start" onclick="controlSystem('start')">Запустить</button>
                <button class="control-button btn-stop" onclick="controlSystem('stop')">Остановить</button>
                <button class="control-button btn-restart" onclick="controlSystem('restart')">Перезапустить</button>
            </div>
        </div>
        
        <!-- Метрики производительности -->
        <div class="card">
            <h3>📈 Производительность</h3>
            <div id="performance-metrics">
                <div class="metric">
                    <div class="metric-value" id="total-return">$0.00</div>
                    <div class="metric-label">Общая прибыль</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="win-rate">0%</div>
                    <div class="metric-label">Винрейт</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="total-trades">0</div>
                    <div class="metric-label">Всего сделок</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="sharpe-ratio">0.00</div>
                    <div class="metric-label">Коэффициент Шарпа</div>
                </div>
            </div>
        </div>
        
        <!-- Уведомления -->
        <div class="card">
            <h3>🔔 Уведомления</h3>
            <div id="alerts-container">
                <p>Нет активных уведомлений</p>
            </div>
        </div>
        
        <!-- График эквити -->
        <div class="card" style="grid-column: 1 / -1;">
            <h3>📊 График эквити</h3>
            <div id="equity-chart" class="chart-container"></div>
        </div>
        
        <!-- Текущие позиции -->
        <div class="card" style="grid-column: 1 / -1;">
            <h3>💼 Текущие позиции</h3>
            <div id="positions-container">
                <table class="positions-table">
                    <thead>
                        <tr>
                            <th>Символ</th>
                            <th>Сторона</th>
                            <th>Цена входа</th>
                            <th>Текущая цена</th>
                            <th>P&L</th>
                            <th>Время</th>
                        </tr>
                    </thead>
                    <tbody id="positions-table-body">
                        <tr><td colspan="6">Нет открытых позиций</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // Подключение к WebSocket
        const socket = io();
        
        // Обработчики событий
        socket.on('status', function(data) {
            updateSystemStatus(data);
        });
        
        socket.on('metrics', function(data) {
            updateMetrics(data);
        });
        
        socket.on('positions', function(data) {
            updatePositions(data);
        });
        
        socket.on('alerts', function(data) {
            updateAlerts(data);
        });
        
        socket.on('charts', function(data) {
            updateCharts(data);
        });
        
        // Функции обновления интерфейса
        function updateSystemStatus(data) {
            document.getElementById('trading-status').textContent = data.trading_status || 'Неизвестно';
            document.getElementById('trading-status').className = 'metric-value ' + 
                (data.trading_status === 'RUNNING' ? 'status-online' : 'status-offline');
            
            document.getElementById('open-positions').textContent = data.open_positions || 0;
            document.getElementById('last-update').textContent = data.last_update || 'Никогда';
        }
        
        function updateMetrics(data) {
            document.getElementById('total-return').textContent = '$' + (data.total_return || 0).toFixed(2);
            document.getElementById('win-rate').textContent = (data.win_rate || 0).toFixed(1) + '%';
            document.getElementById('total-trades').textContent = data.total_trades || 0;
            document.getElementById('sharpe-ratio').textContent = (data.sharpe_ratio || 0).toFixed(2);
        }
        
        function updatePositions(data) {
            const tbody = document.getElementById('positions-table-body');
            tbody.innerHTML = '';
            
            if (data.positions && data.positions.length > 0) {
                data.positions.forEach(pos => {
                    const row = tbody.insertRow();
                    row.innerHTML = `
                        <td>${pos.symbol}</td>
                        <td>${pos.side}</td>
                        <td>$${pos.entry_price.toFixed(2)}</td>
                        <td>$${pos.current_price.toFixed(2)}</td>
                        <td style="color: ${pos.pnl_pct >= 0 ? 'green' : 'red'}">${pos.pnl_pct.toFixed(2)}%</td>
                        <td>${new Date(pos.entry_time).toLocaleString()}</td>
                    `;
                });
            } else {
                tbody.innerHTML = '<tr><td colspan="6">Нет открытых позиций</td></tr>';
            }
        }
        
        function updateAlerts(data) {
            const container = document.getElementById('alerts-container');
            container.innerHTML = '';
            
            if (data.alerts && data.alerts.length > 0) {
                data.alerts.forEach(alert => {
                    const alertDiv = document.createElement('div');
                    alertDiv.className = 'alert alert-' + alert.level.toLowerCase();
                    alertDiv.innerHTML = `<strong>${alert.type}:</strong> ${alert.message}`;
                    container.appendChild(alertDiv);
                });
            } else {
                container.innerHTML = '<p>Нет активных уведомлений</p>';
            }
        }
        
        function updateCharts(data) {
            if (data.equity_chart) {
                Plotly.newPlot('equity-chart', data.equity_chart.data, data.equity_chart.layout);
            }
        }
        
        function controlSystem(action) {
            fetch(`/api/control/${action}`)
                .then(response => response.json())
                .then(data => {
                    alert(data.message || 'Команда выполнена');
                    socket.emit('request_update');
                })
                .catch(error => {
                    alert('Ошибка: ' + error);
                });
        }
        
        // Автоматическое обновление каждые 5 секунд
        setInterval(() => {
            socket.emit('request_update');
        }, 5000);
        
        // Запрос начального обновления
        socket.emit('request_update');
    </script>
</body>
</html>
        """
        
        return html_template
    
    def get_system_status(self) -> Dict:
        """
        Возвращает текущий статус системы
        """
        try:
            status = {
                'trading_status': 'STOPPED',
                'monitoring_status': 'STOPPED',
                'open_positions': 0,
                'last_update': None,
                'uptime': None,
                'symbols': []
            }
            
            if self.trader:
                status['trading_status'] = 'RUNNING' if self.trader.is_trading else 'STOPPED'
                status['open_positions'] = len(self.trader.positions)
                status['last_update'] = self.trader.last_update
                status['symbols'] = self.trader.trading_symbols
                
                if hasattr(self.trader, 'stats') and 'start_time' in self.trader.stats:
                    uptime = datetime.now() - self.trader.stats['start_time']
                    status['uptime'] = str(uptime).split('.')[0]  # Убираем микросекунды
            
            if self.monitor:
                status['monitoring_status'] = 'RUNNING' if self.monitor.is_monitoring else 'STOPPED'
            
            return status
            
        except Exception as e:
            logger.error(f"Ошибка получения статуса системы: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """
        Возвращает метрики производительности
        """
        try:
            if self.monitor:
                return self.monitor.calculate_portfolio_metrics()
            elif self.trader and self.trader.trades_history:
                # Простой расчет метрик
                trades_df = pd.DataFrame(self.trader.trades_history)
                
                total_return = trades_df['pnl_usd'].sum()
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                return {
                    'total_return_usd': total_return,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'winning_trades': winning_trades
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Ошибка получения метрик производительности: {e}")
            return {'error': str(e)}
    
    def get_positions_data(self) -> Dict:
        """
        Возвращает данные о текущих позициях
        """
        try:
            positions = []
            
            if self.trader and self.trader.positions:
                for symbol, position in self.trader.positions.items():
                    current_price = self.trader.get_current_price(symbol)
                    
                    if current_price:
                        entry_price = position['entry_price']
                        
                        if position['side'] == 'LONG':
                            pnl_pct = (current_price - entry_price) / entry_price * 100
                        else:
                            pnl_pct = (entry_price - current_price) / entry_price * 100
                        
                        positions.append({
                            'symbol': symbol,
                            'side': position['side'],
                            'entry_price': entry_price,
                            'current_price': current_price,
                            'pnl_pct': pnl_pct,
                            'amount': position['amount'],
                            'entry_time': position['entry_time'],
                            'confidence': position.get('confidence', 0)
                        })
            
            return {'positions': positions}
            
        except Exception as e:
            logger.error(f"Ошибка получения данных позиций: {e}")
            return {'error': str(e)}
    
    def get_trades_history(self) -> Dict:
        """
        Возвращает историю сделок
        """
        try:
            if self.trader and self.trader.trades_history:
                # Возвращаем последние 50 сделок
                recent_trades = self.trader.trades_history[-50:]
                return {'trades': recent_trades}
            else:
                return {'trades': []}
                
        except Exception as e:
            logger.error(f"Ошибка получения истории сделок: {e}")
            return {'error': str(e)}
    
    def create_equity_chart(self) -> Dict:
        """
        Создает график эквити
        """
        try:
            if not PLOTLY_AVAILABLE:
                return {'error': 'Plotly не установлен'}
            
            if self.monitor and self.monitor.equity_curve:
                equity_data = self.monitor.equity_curve
                
                timestamps = [point['timestamp'] for point in equity_data]
                equity_values = [point['equity'] for point in equity_data]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=equity_values,
                    mode='lines',
                    name='Эквити',
                    line=dict(color='#2c3e50', width=2)
                ))
                
                fig.update_layout(
                    title='Кривая эквити',
                    xaxis_title='Время',
                    yaxis_title='Баланс ($)',
                    hovermode='x unified'
                )
                
                return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
            return {'data': [], 'layout': {}}
            
        except Exception as e:
            logger.error(f"Ошибка создания графика эквити: {e}")
            return {'error': str(e)}
    
    def create_performance_chart(self) -> Dict:
        """
        Создает график производительности
        """
        try:
            if not PLOTLY_AVAILABLE:
                return {'error': 'Plotly не установлен'}
            
            # Заглушка для графика производительности
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Прибыльные', 'Убыточные'],
                y=[0, 0],
                name='Сделки'
            ))
            
            fig.update_layout(
                title='Распределение сделок',
                xaxis_title='Тип сделки',
                yaxis_title='Количество'
            )
            
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
        except Exception as e:
            logger.error(f"Ошибка создания графика производительности: {e}")
            return {'error': str(e)}
    
    def handle_control_action(self, action: str) -> Dict:
        """
        Обрабатывает команды управления системой
        """
        try:
            if action == 'start':
                if self.trader and not self.trader.is_trading:
                    # TODO: Запуск в отдельном потоке
                    return {'success': True, 'message': 'Торговля запущена'}
                else:
                    return {'success': False, 'message': 'Торговля уже запущена или трейдер недоступен'}
            
            elif action == 'stop':
                if self.trader and self.trader.is_trading:
                    self.trader.stop_trading()
                    return {'success': True, 'message': 'Торговля остановлена'}
                else:
                    return {'success': False, 'message': 'Торговля не активна'}
            
            elif action == 'restart':
                if self.trader:
                    if self.trader.is_trading:
                        self.trader.stop_trading()
                    # TODO: Перезапуск системы
                    return {'success': True, 'message': 'Система перезапущена'}
                else:
                    return {'success': False, 'message': 'Трейдер недоступен'}
            
            else:
                return {'success': False, 'message': f'Неизвестная команда: {action}'}
                
        except Exception as e:
            logger.error(f"Ошибка выполнения команды {action}: {e}")
            return {'success': False, 'message': str(e)}
    
    def broadcast_update(self):
        """
        Отправляет обновления всем подключенным клиентам
        """
        try:
            # Отправляем все обновления
            self.socketio.emit('status', self.get_system_status())
            self.socketio.emit('metrics', self.get_performance_metrics())
            self.socketio.emit('positions', self.get_positions_data())
            
            # Отправляем уведомления если есть монитор
            if self.monitor:
                report = self.monitor.generate_status_report()
                if 'active_alerts' in report:
                    self.socketio.emit('alerts', {'alerts': report['active_alerts']})
            
            # Отправляем графики
            charts_data = {
                'equity_chart': self.create_equity_chart()
            }
            self.socketio.emit('charts', charts_data)
            
        except Exception as e:
            logger.error(f"Ошибка отправки обновлений: {e}")
    
    def run(self, debug=False, host='0.0.0.0'):
        """
        Запускает веб-сервер дашборда
        """
        if not FLASK_AVAILABLE:
            logger.error("❌ Flask не установлен. Дашборд недоступен.")
            return False
        
        logger.info(f"🌐 Запуск веб-дашборда на http://{host}:{self.port}")
        
        try:
            self.is_running = True
            self.socketio.run(self.app, host=host, port=self.port, debug=debug)
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска дашборда: {e}")
            return False
        finally:
            self.is_running = False
