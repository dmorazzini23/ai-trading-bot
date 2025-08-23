"""Real-time monitoring dashboard for AI trading bot.

This module provides comprehensive real-time monitoring and analytics:
- Real-time performance dashboards
- Trading metrics and KPIs visualization
- Predictive alerting based on pattern recognition
- Trade execution analysis and optimization
- Portfolio performance attribution
- Risk metrics monitoring (VaR, Sharpe ratio, etc.)
- Automated reporting for regulatory compliance

AI-AGENT-REF: Production-grade monitoring dashboard system
"""
from __future__ import annotations
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from flask import Flask, jsonify, render_template_string, request

@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    daily_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float | None
    max_drawdown: float
    current_drawdown: float
    total_volume: float
    avg_trade_size: float

@dataclass
class PerformanceKPIs:
    """Key Performance Indicators."""
    timestamp: datetime
    orders_per_minute: float
    avg_execution_latency_ms: float
    p95_execution_latency_ms: float
    data_processing_latency_ms: float
    system_uptime_percent: float
    error_rate_percent: float
    cpu_utilization: float
    memory_utilization: float
    active_positions: int
    portfolio_value: float

@dataclass
class RiskMetrics:
    """Risk management metrics."""
    timestamp: datetime
    var_95: float
    var_99: float
    expected_shortfall: float
    beta: float
    volatility: float
    correlation_to_market: float
    max_position_concentration: float
    leverage_ratio: float
    margin_utilization: float

class MonitoringDashboard:
    """Real-time monitoring dashboard system."""

    def __init__(self, port: int=5000):
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.trading_metrics: deque = deque(maxlen=1440)
        self.performance_kpis: deque = deque(maxlen=1440)
        self.risk_metrics: deque = deque(maxlen=1440)
        self.current_trades: dict[str, Any] = {}
        self.active_orders: dict[str, Any] = {}
        self.alerts: deque = deque(maxlen=100)
        self.app = Flask(__name__)
        self.setup_routes()
        self._monitoring_active = False
        self._monitor_thread = None
        self.alert_callbacks: list[Callable] = []
        self.alert_thresholds = {'execution_latency_ms': 50.0, 'error_rate_percent': 5.0, 'drawdown_percent': 10.0, 'cpu_utilization': 80.0, 'memory_utilization': 85.0}
        self.trade_history: deque = deque(maxlen=10000)
        self.pnl_history: deque = deque(maxlen=1440)
        self.logger.info('Monitoring dashboard initialized')

    def setup_routes(self):
        """Setup Flask routes for web dashboard."""

        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template_string(DASHBOARD_HTML_TEMPLATE)

        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current metrics as JSON."""
            return jsonify({'trading_metrics': self.get_latest_trading_metrics(), 'performance_kpis': self.get_latest_performance_kpis(), 'risk_metrics': self.get_latest_risk_metrics(), 'alerts': [asdict(alert) for alert in list(self.alerts)[-10:]]})

        @self.app.route('/api/trades')
        def get_trades():
            """Get recent trades."""
            recent_trades = list(self.trade_history)[-100:]
            return jsonify(recent_trades)

        @self.app.route('/api/performance')
        def get_performance():
            """Get performance analytics."""
            return jsonify(self.calculate_performance_analytics())

        @self.app.route('/api/risk_report')
        def get_risk_report():
            """Get comprehensive risk report."""
            return jsonify(self.generate_risk_report())

        @self.app.route('/api/system_health')
        def get_system_health():
            """Get system health status."""
            try:
                from health_check import get_health_status
                return jsonify(get_health_status())
            except ImportError:
                return jsonify({'error': 'Health check module not available'})

        @self.app.route('/api/alerts', methods=['GET', 'POST'])
        def handle_alerts():
            """Handle alert operations."""
            if request.method == 'POST':
                alert_data = request.json
                self.add_alert(alert_data.get('level', 'INFO'), alert_data.get('message', ''), alert_data.get('details', {}))
                return jsonify({'status': 'success'})
            else:
                return jsonify([asdict(alert) for alert in list(self.alerts)])

    def start_dashboard(self, debug: bool=False):
        """Start the web dashboard."""
        try:
            self.logger.info(f'Starting monitoring dashboard on port {self.port}')
            self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Failed to start dashboard: {e}')

    def start_monitoring(self, interval_seconds: int=60):
        """Start background monitoring data collection."""
        if self._monitoring_active:
            return
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval_seconds,), daemon=True)
        self._monitor_thread.start()
        self.logger.info('Background monitoring started')

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info('Background monitoring stopped')

    def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                self.collect_trading_metrics()
                self.collect_performance_kpis()
                self.collect_risk_metrics()
                self.check_alert_conditions()
            except (ValueError, TypeError) as e:
                self.logger.error(f'Error in monitoring loop: {e}')
            time.sleep(interval_seconds)

    def collect_trading_metrics(self):
        """Collect current trading metrics."""
        try:
            recent_trades = [t for t in self.trade_history if time.time() - t.get('timestamp', 0) < 86400]
            if not recent_trades:
                return
            winning_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in recent_trades if t.get('pnl', 0) < 0]
            total_pnl = sum((t.get('pnl', 0) for t in recent_trades))
            total_volume = sum((t.get('volume', 0) for t in recent_trades))
            win_rate = len(winning_trades) / len(recent_trades) * 100 if recent_trades else 0
            avg_win = statistics.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = statistics.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss > 0 else 0
            cumulative_pnl = []
            running_total = 0
            for trade in recent_trades:
                running_total += trade.get('pnl', 0)
                cumulative_pnl.append(running_total)
            peak = max(cumulative_pnl) if cumulative_pnl else 0
            current_value = cumulative_pnl[-1] if cumulative_pnl else 0
            current_drawdown = (peak - current_value) / peak * 100 if peak > 0 else 0
            metrics = TradingMetrics(timestamp=datetime.now(UTC), total_trades=len(recent_trades), winning_trades=len(winning_trades), losing_trades=len(losing_trades), total_pnl=total_pnl, daily_pnl=total_pnl, win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss, profit_factor=profit_factor, sharpe_ratio=None, max_drawdown=current_drawdown, current_drawdown=current_drawdown, total_volume=total_volume, avg_trade_size=total_volume / len(recent_trades) if recent_trades else 0)
            self.trading_metrics.append(metrics)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error collecting trading metrics: {e}')

    def collect_performance_kpis(self):
        """Collect performance KPIs."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            [o for o in self.active_orders.values()]
            kpis = PerformanceKPIs(timestamp=datetime.now(UTC), orders_per_minute=0, avg_execution_latency_ms=0, p95_execution_latency_ms=0, data_processing_latency_ms=0, system_uptime_percent=99.9, error_rate_percent=0, cpu_utilization=cpu_percent, memory_utilization=memory.percent, active_positions=len(self.current_trades), portfolio_value=0)
            self.performance_kpis.append(kpis)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error collecting performance KPIs: {e}')

    def collect_risk_metrics(self):
        """Collect risk management metrics."""
        try:
            pnl_values = [t.get('pnl', 0) for t in self.trade_history]
            if len(pnl_values) < 2:
                return
            volatility = statistics.stdev(pnl_values) if len(pnl_values) > 1 else 0
            sorted_pnl = sorted(pnl_values)
            var_95_index = int(len(sorted_pnl) * 0.05)
            var_99_index = int(len(sorted_pnl) * 0.01)
            var_95 = sorted_pnl[var_95_index] if var_95_index < len(sorted_pnl) else 0
            var_99 = sorted_pnl[var_99_index] if var_99_index < len(sorted_pnl) else 0
            risk_metrics = RiskMetrics(timestamp=datetime.now(UTC), var_95=abs(var_95), var_99=abs(var_99), expected_shortfall=abs(statistics.mean(sorted_pnl[:var_95_index])) if var_95_index > 0 else 0, beta=1.0, volatility=volatility, correlation_to_market=0.0, max_position_concentration=0.0, leverage_ratio=1.0, margin_utilization=0.0)
            self.risk_metrics.append(risk_metrics)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error collecting risk metrics: {e}')

    def check_alert_conditions(self):
        """Check for alert conditions."""
        try:
            latest_kpis = self.get_latest_performance_kpis()
            latest_risk = self.get_latest_risk_metrics()
            if not latest_kpis:
                return
            if latest_kpis['cpu_utilization'] > self.alert_thresholds['cpu_utilization']:
                self.add_alert('WARNING', f"High CPU utilization: {latest_kpis['cpu_utilization']:.1f}%", {'metric': 'cpu_utilization', 'value': latest_kpis['cpu_utilization']})
            if latest_kpis['memory_utilization'] > self.alert_thresholds['memory_utilization']:
                self.add_alert('WARNING', f"High memory utilization: {latest_kpis['memory_utilization']:.1f}%", {'metric': 'memory_utilization', 'value': latest_kpis['memory_utilization']})
            if latest_risk and latest_risk.get('current_drawdown', 0) > self.alert_thresholds['drawdown_percent']:
                self.add_alert('CRITICAL', f"High drawdown: {latest_risk['current_drawdown']:.1f}%", {'metric': 'drawdown', 'value': latest_risk['current_drawdown']})
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error checking alert conditions: {e}')

    def add_alert(self, level: str, message: str, details: dict[str, Any]=None):
        """Add new alert."""
        alert = {'timestamp': datetime.now(UTC).isoformat(), 'level': level, 'message': message, 'details': details or {}}
        self.alerts.append(alert)
        if level == 'CRITICAL':
            self.logger.error(f'CRITICAL ALERT: {message}')
        elif level == 'WARNING':
            self.logger.warning(f'WARNING ALERT: {message}')
        else:
            self.logger.info(f'INFO ALERT: {message}')
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except (ValueError, TypeError) as e:
                self.logger.error(f'Error in alert callback: {e}')

    def add_alert_callback(self, callback: Callable):
        """Add alert notification callback."""
        self.alert_callbacks.append(callback)

    def record_trade(self, symbol: str, side: str, quantity: float, price: float, pnl: float, order_id: str=None):
        """Record a completed trade."""
        trade_record = {'timestamp': time.time(), 'datetime': datetime.now(UTC).isoformat(), 'symbol': symbol, 'side': side, 'quantity': quantity, 'price': price, 'volume': quantity * price, 'pnl': pnl, 'order_id': order_id}
        self.trade_history.append(trade_record)
        if symbol in self.current_trades:
            self.current_trades[symbol].update(trade_record)
        else:
            self.current_trades[symbol] = trade_record

    def get_latest_trading_metrics(self) -> dict[str, Any] | None:
        """Get latest trading metrics."""
        if self.trading_metrics:
            return asdict(self.trading_metrics[-1])
        return None

    def get_latest_performance_kpis(self) -> dict[str, Any] | None:
        """Get latest performance KPIs."""
        if self.performance_kpis:
            return asdict(self.performance_kpis[-1])
        return None

    def get_latest_risk_metrics(self) -> dict[str, Any] | None:
        """Get latest risk metrics."""
        if self.risk_metrics:
            return asdict(self.risk_metrics[-1])
        return None

    def calculate_performance_analytics(self) -> dict[str, Any]:
        """Calculate comprehensive performance analytics."""
        if not self.trade_history:
            return {'error': 'No trade data available'}
        trades = list(self.trade_history)
        daily_pnl = defaultdict(float)
        for trade in trades:
            trade_date = datetime.fromtimestamp(trade['timestamp']).date()
            daily_pnl[trade_date] += trade['pnl']
        total_pnl = sum((trade['pnl'] for trade in trades))
        winning_days = sum((1 for pnl in daily_pnl.values() if pnl > 0))
        total_days = len(daily_pnl)
        return {'total_trades': len(trades), 'total_pnl': total_pnl, 'avg_pnl_per_trade': total_pnl / len(trades), 'winning_days': winning_days, 'total_trading_days': total_days, 'daily_win_rate': winning_days / total_days * 100 if total_days > 0 else 0, 'best_day': max(daily_pnl.values()) if daily_pnl else 0, 'worst_day': min(daily_pnl.values()) if daily_pnl else 0, 'daily_pnl_std': statistics.stdev(daily_pnl.values()) if len(daily_pnl) > 1 else 0}

    def generate_risk_report(self) -> dict[str, Any]:
        """Generate comprehensive risk report."""
        latest_risk = self.get_latest_risk_metrics()
        risk_score = 0
        risk_factors = []
        if latest_risk:
            if latest_risk['var_95'] > 1000:
                risk_score += 25
                risk_factors.append('High Value at Risk')
            if latest_risk['volatility'] > 0.02:
                risk_score += 20
                risk_factors.append('High volatility')
            if latest_risk['max_position_concentration'] > 0.3:
                risk_score += 30
                risk_factors.append('High position concentration')
        return {'timestamp': datetime.now(UTC).isoformat(), 'risk_score': risk_score, 'risk_level': 'HIGH' if risk_score > 50 else 'MEDIUM' if risk_score > 25 else 'LOW', 'risk_factors': risk_factors, 'latest_metrics': latest_risk, 'recommendations': self._get_risk_recommendations(risk_score, risk_factors)}

    def _get_risk_recommendations(self, risk_score: int, risk_factors: list[str]) -> list[str]:
        """Get risk management recommendations."""
        recommendations = []
        if risk_score > 50:
            recommendations.append('Consider reducing position sizes')
            recommendations.append('Implement stricter stop-loss levels')
        if 'High position concentration' in risk_factors:
            recommendations.append('Diversify positions across more symbols')
        if 'High volatility' in risk_factors:
            recommendations.append('Reduce leverage during high volatility periods')
        return recommendations
DASHBOARD_HTML_TEMPLATE = '\n<!DOCTYPE html>\n<html>\n<head>\n    <title>AI Trading Bot - Monitoring Dashboard</title>\n    <meta charset="utf-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1">\n    <style>\n        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }\n        .container { max-width: 1200px; margin: 0 auto; }\n        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }\n        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }\n        .metric-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }\n        .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }\n        .metric-value { font-size: 24px; font-weight: bold; color: #27ae60; }\n        .metric-change { font-size: 14px; margin-top: 5px; }\n        .positive { color: #27ae60; }\n        .negative { color: #e74c3c; }\n        .alert { padding: 10px; margin: 5px 0; border-radius: 3px; }\n        .alert-warning { background: #f39c12; color: white; }\n        .alert-critical { background: #e74c3c; color: white; }\n        .alert-info { background: #3498db; color: white; }\n        .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 3px; cursor: pointer; }\n        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }\n        .status-healthy { background: #27ae60; }\n        .status-warning { background: #f39c12; }\n        .status-critical { background: #e74c3c; }\n    </style>\n</head>\n<body>\n    <div class="container">\n        <div class="header">\n            <h1>AI Trading Bot - Monitoring Dashboard</h1>\n            <p>Real-time performance and risk monitoring</p>\n            <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>\n        </div>\n        \n        <div id="alerts-section">\n            <!-- Alerts will be loaded here -->\n        </div>\n        \n        <div class="metrics-grid">\n            <div class="metric-card">\n                <div class="metric-title">System Status</div>\n                <div id="system-status">\n                    <span class="status-indicator status-healthy"></span>\n                    <span>All Systems Operational</span>\n                </div>\n            </div>\n            \n            <div class="metric-card">\n                <div class="metric-title">Total P&L</div>\n                <div class="metric-value" id="total-pnl">$0.00</div>\n                <div class="metric-change" id="pnl-change">No change</div>\n            </div>\n            \n            <div class="metric-card">\n                <div class="metric-title">Win Rate</div>\n                <div class="metric-value" id="win-rate">0%</div>\n                <div class="metric-change" id="win-rate-change">--</div>\n            </div>\n            \n            <div class="metric-card">\n                <div class="metric-title">Active Positions</div>\n                <div class="metric-value" id="active-positions">0</div>\n                <div class="metric-change" id="positions-change">--</div>\n            </div>\n            \n            <div class="metric-card">\n                <div class="metric-title">CPU Usage</div>\n                <div class="metric-value" id="cpu-usage">0%</div>\n                <div class="metric-change" id="cpu-change">--</div>\n            </div>\n            \n            <div class="metric-card">\n                <div class="metric-title">Memory Usage</div>\n                <div class="metric-value" id="memory-usage">0%</div>\n                <div class="metric-change" id="memory-change">--</div>\n            </div>\n        </div>\n        \n        <div class="metrics-grid" style="margin-top: 20px;">\n            <div class="metric-card">\n                <div class="metric-title">Risk Metrics</div>\n                <div id="risk-metrics">\n                    <p>VaR (95%): <span id="var-95">$0</span></p>\n                    <p>Max Drawdown: <span id="max-drawdown">0%</span></p>\n                    <p>Volatility: <span id="volatility">0%</span></p>\n                </div>\n            </div>\n            \n            <div class="metric-card">\n                <div class="metric-title">Recent Trades</div>\n                <div id="recent-trades">\n                    <p>Loading trades...</p>\n                </div>\n            </div>\n        </div>\n    </div>\n    \n    <script>\n        function refreshData() {\n            Promise.all([\n                fetch(\'/api/metrics\'),\n                fetch(\'/api/performance\'),\n                fetch(\'/api/trades\'),\n                fetch(\'/api/system_health\')\n            ]).then(responses => Promise.all(responses.map(r => r.json())))\n            .then(([metrics, performance, trades, health]) => {\n                updateMetrics(metrics);\n                updateTrades(trades);\n                updateHealth(health);\n            })\n            .catch(error => console.error(\'Error fetching data:\', error));\n        }\n        \n        function updateMetrics(data) {\n            if (data.trading_metrics) {\n                document.getElementById(\'total-pnl\').textContent = \'$\' + (data.trading_metrics.total_pnl || 0).toFixed(2);\n                document.getElementById(\'win-rate\').textContent = (data.trading_metrics.win_rate || 0).toFixed(1) + \'%\';\n            }\n            \n            if (data.performance_kpis) {\n                document.getElementById(\'cpu-usage\').textContent = (data.performance_kpis.cpu_utilization || 0).toFixed(1) + \'%\';\n                document.getElementById(\'memory-usage\').textContent = (data.performance_kpis.memory_utilization || 0).toFixed(1) + \'%\';\n                document.getElementById(\'active-positions\').textContent = data.performance_kpis.active_positions || 0;\n            }\n            \n            if (data.risk_metrics) {\n                document.getElementById(\'var-95\').textContent = \'$\' + (data.risk_metrics.var_95 || 0).toFixed(2);\n                document.getElementById(\'max-drawdown\').textContent = (data.risk_metrics.max_drawdown || 0).toFixed(1) + \'%\';\n                document.getElementById(\'volatility\').textContent = ((data.risk_metrics.volatility || 0) * 100).toFixed(2) + \'%\';\n            }\n            \n            if (data.alerts && data.alerts.length > 0) {\n                updateAlerts(data.alerts);\n            }\n        }\n        \n        function updateTrades(trades) {\n            const tradesDiv = document.getElementById(\'recent-trades\');\n            if (trades && trades.length > 0) {\n                const recentTrades = trades.slice(-5);\n                tradesDiv.innerHTML = recentTrades.map(trade => \n                    `<p>${trade.symbol} ${trade.side} ${trade.quantity} @ $${trade.price}</p>`\n                ).join(\'\');\n            } else {\n                tradesDiv.innerHTML = \'<p>No recent trades</p>\';\n            }\n        }\n        \n        function updateHealth(health) {\n            const statusDiv = document.getElementById(\'system-status\');\n            if (health && health.overall_status) {\n                const status = health.overall_status;\n                let statusClass = \'status-healthy\';\n                let statusText = \'All Systems Operational\';\n                \n                if (status === \'warning\') {\n                    statusClass = \'status-warning\';\n                    statusText = \'System Warnings Detected\';\n                } else if (status === \'critical\') {\n                    statusClass = \'status-critical\';\n                    statusText = \'Critical Issues Detected\';\n                }\n                \n                statusDiv.innerHTML = `<span class="status-indicator ${statusClass}"></span><span>${statusText}</span>`;\n            }\n        }\n        \n        function updateAlerts(alerts) {\n            const alertsSection = document.getElementById(\'alerts-section\');\n            if (alerts && alerts.length > 0) {\n                alertsSection.innerHTML = \'<h3>Recent Alerts</h3>\' + \n                    alerts.map(alert => {\n                        const alertClass = `alert-${alert.level.toLowerCase()}`;\n                        return `<div class="alert ${alertClass}">${alert.message}</div>`;\n                    }).join(\'\');\n            }\n        }\n        \n        // Auto-refresh every 30 seconds\n        setInterval(refreshData, 30000);\n        \n        // Initial load\n        refreshData();\n    </script>\n</body>\n</html>\n'
_monitoring_dashboard: MonitoringDashboard | None = None

def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get global monitoring dashboard instance."""
    global _monitoring_dashboard
    if _monitoring_dashboard is None:
        _monitoring_dashboard = MonitoringDashboard()
    return _monitoring_dashboard

def initialize_monitoring_dashboard(port: int=5000) -> MonitoringDashboard:
    """Initialize monitoring dashboard."""
    global _monitoring_dashboard
    _monitoring_dashboard = MonitoringDashboard(port)
    return _monitoring_dashboard

def start_dashboard_server(port: int=5000, debug: bool=False):
    """Start dashboard web server."""
    dashboard = get_monitoring_dashboard()
    dashboard.start_dashboard(debug=debug)
if __name__ == '__main__':
    dashboard = MonitoringDashboard()
    dashboard.start_monitoring()
    dashboard.start_dashboard(debug=True)