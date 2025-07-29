"""Performance metrics calculation and monitoring dashboard."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from flask import Flask, jsonify, render_template_string
import threading

from ..core.models import (
    PortfolioMetrics, StrategyPerformance, TradePosition,
    MarketData, ExecutionReport
)
from ..core.enums import StrategyType
from ..core.logging import get_trading_logger


logger = get_trading_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    # Returns
    total_return: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    yearly_return: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    current_drawdown: float
    drawdown_duration: int  # days
    average_drawdown: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Portfolio metrics
    total_pnl: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    cash_utilization: float
    
    # Risk metrics
    volatility: float
    var_95: float
    var_99: float
    beta: float
    alpha: float
    correlation_with_market: float
    
    # Efficiency metrics
    trades_per_day: float
    avg_holding_period: float  # hours
    turnover_ratio: float


class PerformanceCalculator:
    """Calculate institutional-grade performance metrics."""
    
    def __init__(self, benchmark_symbol: str = "SPY"):
        self.benchmark_symbol = benchmark_symbol
        self._portfolio_history: List[Tuple[datetime, float]] = []
        self._benchmark_history: List[Tuple[datetime, float]] = []
        self._trade_history: List[ExecutionReport] = []
        self._position_history: List[Tuple[datetime, Dict[str, TradePosition]]] = []
        
    def add_portfolio_snapshot(
        self,
        timestamp: datetime,
        portfolio_value: float,
        positions: Dict[str, TradePosition]
    ) -> None:
        """Add portfolio snapshot for performance calculation."""
        self._portfolio_history.append((timestamp, portfolio_value))
        self._position_history.append((timestamp, positions.copy()))
        
        # Keep only recent history (1 year)
        cutoff = timestamp - timedelta(days=365)
        self._portfolio_history = [
            (t, v) for t, v in self._portfolio_history if t >= cutoff
        ]
        self._position_history = [
            (t, p) for t, p in self._position_history if t >= cutoff
        ]
    
    def add_trade(self, trade: ExecutionReport) -> None:
        """Add completed trade for performance calculation."""
        self._trade_history.append(trade)
        
        # Keep only recent trades (1 year)
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        self._trade_history = [
            t for t in self._trade_history if t.execution_time >= cutoff
        ]
    
    def calculate_returns(self, period_days: int = 252) -> List[float]:
        """Calculate daily returns for given period."""
        if len(self._portfolio_history) < 2:
            return []
        
        # Get recent history
        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
        recent_history = [
            (t, v) for t, v in self._portfolio_history if t >= cutoff
        ]
        
        if len(recent_history) < 2:
            return []
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(recent_history)):
            prev_value = recent_history[i-1][1]
            curr_value = recent_history[i][1]
            
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)
        
        return returns
    
    def calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_sortino_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio (using downside deviation)."""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)
        
        # Calculate downside deviation
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) == 0:
            return float('inf')  # No downside
        
        downside_dev = np.std(negative_returns)
        if downside_dev == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_dev * np.sqrt(252)
    
    def calculate_max_drawdown(self, period_days: int = 252) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if len(self._portfolio_history) < 2:
            return 0.0, 0
        
        # Get recent history
        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
        recent_history = [
            (t, v) for t, v in self._portfolio_history if t >= cutoff
        ]
        
        if len(recent_history) < 2:
            return 0.0, 0
        
        values = [v for _, v in recent_history]
        peak = values[0]
        max_dd = 0.0
        current_dd_duration = 0
        max_dd_duration = 0
        
        for value in values[1:]:
            if value > peak:
                peak = value
                current_dd_duration = 0
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
        
        return max_dd, max_dd_duration
    
    def calculate_var(
        self,
        returns: List[float],
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Value at Risk."""
        if not returns or len(returns) < 10:
            return 0.0
        
        returns_array = np.array(returns)
        return -np.percentile(returns_array, (1 - confidence_level) * 100)
    
    def calculate_trade_metrics(self) -> Dict[str, Any]:
        """Calculate trading-specific metrics."""
        if not self._trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # Calculate P&L for each trade (simplified)
        trade_pnls = []
        for trade in self._trade_history:
            # This is simplified - in reality you'd track entry/exit prices
            if hasattr(trade, 'pnl'):
                trade_pnls.append(float(trade.pnl))
        
        if not trade_pnls:
            return {
                'total_trades': len(self._trade_history),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0
        
        return {
            'total_trades': len(trade_pnls),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trade_pnls) if trade_pnls else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'average_win': np.mean(winning_trades) if winning_trades else 0,
            'average_loss': np.mean([abs(l) for l in losing_trades]) if losing_trades else 0,
            'largest_win': max(winning_trades) if winning_trades else 0,
            'largest_loss': min(losing_trades) if losing_trades else 0
        }
    
    def calculate_comprehensive_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        # Calculate returns for different periods
        daily_returns = self.calculate_returns(30)  # Last 30 days
        weekly_returns = self.calculate_returns(7)   # Last week
        monthly_returns = self.calculate_returns(252) # Last year
        
        # Risk metrics
        sharpe = self.calculate_sharpe_ratio(daily_returns)
        sortino = self.calculate_sortino_ratio(daily_returns)
        max_dd, dd_duration = self.calculate_max_drawdown()
        var_95 = self.calculate_var(daily_returns, 0.95)
        var_99 = self.calculate_var(daily_returns, 0.99)
        
        # Trade metrics
        trade_metrics = self.calculate_trade_metrics()
        
        # Portfolio metrics
        current_portfolio = self._portfolio_history[-1][1] if self._portfolio_history else 0
        initial_portfolio = self._portfolio_history[0][1] if self._portfolio_history else current_portfolio
        
        total_return = (
            (current_portfolio - initial_portfolio) / initial_portfolio
            if initial_portfolio > 0 else 0
        )
        
        return PerformanceMetrics(
            # Returns
            total_return=total_return,
            daily_return=daily_returns[-1] if daily_returns else 0,
            weekly_return=sum(weekly_returns) if weekly_returns else 0,
            monthly_return=sum(monthly_returns[-30:]) if len(monthly_returns) >= 30 else 0,
            yearly_return=sum(monthly_returns) if monthly_returns else 0,
            
            # Risk-adjusted returns
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=total_return / max_dd if max_dd > 0 else 0,
            information_ratio=0,  # Would need benchmark data
            
            # Drawdown metrics
            max_drawdown=max_dd,
            current_drawdown=0,  # Would need current calculation
            drawdown_duration=dd_duration,
            average_drawdown=max_dd / 2,  # Simplified
            
            # Trading metrics
            **trade_metrics,
            
            # Portfolio metrics
            total_pnl=current_portfolio - initial_portfolio,
            gross_exposure=0,  # Would need position data
            net_exposure=0,    # Would need position data
            leverage=1.0,      # Would need position data
            cash_utilization=0.8,  # Simplified
            
            # Risk metrics
            volatility=np.std(daily_returns) * np.sqrt(252) if daily_returns else 0,
            var_95=var_95,
            var_99=var_99,
            beta=1.0,  # Would need benchmark correlation
            alpha=0.0, # Would need benchmark data
            correlation_with_market=0.0,  # Would need benchmark data
            
            # Efficiency metrics
            trades_per_day=len(self._trade_history) / 30 if self._trade_history else 0,
            avg_holding_period=24.0,  # Simplified
            turnover_ratio=2.0  # Simplified
        )


class MonitoringDashboard:
    """Real-time monitoring dashboard with web interface."""
    
    def __init__(
        self,
        performance_calculator: PerformanceCalculator,
        port: int = 8080
    ):
        self.performance_calculator = performance_calculator
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
        self._current_metrics: Optional[PerformanceMetrics] = None
        self._portfolio_metrics: Optional[PortfolioMetrics] = None
        self._strategy_performance: Dict[str, StrategyPerformance] = {}
        self._is_running = False
        
    def _setup_routes(self) -> None:
        """Setup Flask routes for dashboard."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template_string(DASHBOARD_TEMPLATE)
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for current metrics."""
            if not self._current_metrics:
                return jsonify({'error': 'No metrics available'})
            
            return jsonify({
                'performance': self._serialize_metrics(self._current_metrics),
                'portfolio': self._serialize_portfolio_metrics(),
                'strategies': self._serialize_strategy_metrics(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        @self.app.route('/api/performance/history')
        def api_performance_history():
            """API endpoint for performance history."""
            # Get portfolio history for chart
            history = self.performance_calculator._portfolio_history[-100:]  # Last 100 points
            
            return jsonify({
                'timestamps': [t.isoformat() for t, _ in history],
                'values': [v for _, v in history],
                'returns': self.performance_calculator.calculate_returns(30)
            })
        
        @self.app.route('/api/risk/summary')
        def api_risk_summary():
            """API endpoint for risk summary."""
            # This would integrate with risk monitor
            return jsonify({
                'status': 'monitoring',
                'alerts': [],
                'violations': 0
            })
    
    def _serialize_metrics(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Serialize performance metrics for JSON."""
        return {
            'total_return': f"{metrics.total_return:.2%}",
            'daily_return': f"{metrics.daily_return:.2%}",
            'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
            'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
            'max_drawdown': f"{metrics.max_drawdown:.2%}",
            'win_rate': f"{metrics.win_rate:.1%}",
            'profit_factor': f"{metrics.profit_factor:.2f}",
            'total_trades': metrics.total_trades,
            'volatility': f"{metrics.volatility:.2%}",
            'var_95': f"{metrics.var_95:.2%}",
            'var_99': f"{metrics.var_99:.2%}"
        }
    
    def _serialize_portfolio_metrics(self) -> Dict[str, Any]:
        """Serialize portfolio metrics for JSON."""
        if not self._portfolio_metrics:
            return {}
        
        return {
            'total_value': f"${self._portfolio_metrics.total_value:,.0f}",
            'daily_pnl': f"${self._portfolio_metrics.daily_pnl:,.0f}",
            'cash_balance': f"${self._portfolio_metrics.cash_balance:,.0f}",
            'leverage': f"{self._portfolio_metrics.leverage:.2f}x",
            'gross_exposure': f"${self._portfolio_metrics.gross_exposure:,.0f}",
            'net_exposure': f"${self._portfolio_metrics.net_exposure:,.0f}"
        }
    
    def _serialize_strategy_metrics(self) -> List[Dict[str, Any]]:
        """Serialize strategy metrics for JSON."""
        return [
            {
                'strategy_id': strategy.strategy_id,
                'strategy_type': strategy.strategy_type.value,
                'allocation': f"{strategy.allocation:.1%}",
                'total_pnl': f"${strategy.total_pnl:,.0f}",
                'daily_pnl': f"${strategy.daily_pnl:,.0f}",
                'win_rate': f"{strategy.win_rate:.1f}%",
                'active_positions': strategy.active_positions,
                'total_trades': strategy.total_trades
            }
            for strategy in self._strategy_performance.values()
        ]
    
    def update_metrics(
        self,
        portfolio_metrics: PortfolioMetrics,
        strategy_performance: Dict[str, StrategyPerformance]
    ) -> None:
        """Update dashboard with new metrics."""
        self._portfolio_metrics = portfolio_metrics
        self._strategy_performance = strategy_performance
        self._current_metrics = self.performance_calculator.calculate_comprehensive_metrics()
    
    def start_dashboard(self) -> None:
        """Start the dashboard server."""
        if self._is_running:
            logger.warning("Dashboard already running")
            return
        
        self._is_running = True
        
        def run_server():
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=False,
                use_reloader=False
            )
        
        dashboard_thread = threading.Thread(target=run_server, daemon=True)
        dashboard_thread.start()
        
        logger.info(f"Monitoring dashboard started on port {self.port}")
    
    def stop_dashboard(self) -> None:
        """Stop the dashboard server."""
        self._is_running = False
        logger.info("Dashboard stopped")


# Dashboard HTML template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Bot - Institutional Dashboard</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            margin: 0; padding: 20px; background: #f5f7fa; color: #2c3e50;
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;
        }
        .metrics-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px; margin-bottom: 20px;
        }
        .card { 
            background: white; padding: 20px; border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 4px solid #3498db;
        }
        .card h3 { margin-top: 0; color: #2c3e50; font-size: 1.2em; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-value { font-weight: bold; color: #27ae60; }
        .metric-value.negative { color: #e74c3c; }
        .chart-container { 
            background: white; padding: 20px; border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
        }
        .strategies-table { 
            background: white; border-radius: 10px; overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 15px; text-align: left; border-bottom: 1px solid #ecf0f1; }
        th { background: #34495e; color: white; font-weight: 600; }
        .status-indicator { 
            width: 10px; height: 10px; border-radius: 50%;
            background: #27ae60; display: inline-block; margin-right: 5px;
        }
        .last-updated { 
            text-align: center; color: #7f8c8d; margin-top: 20px; font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 AI Trading Bot - Institutional Dashboard</h1>
        <div style="display: flex; align-items: center;">
            <span class="status-indicator"></span>
            <span>System Active - Real-time Monitoring</span>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="card">
            <h3>📈 Performance Metrics</h3>
            <div id="performance-metrics">Loading...</div>
        </div>
        
        <div class="card">
            <h3>💰 Portfolio Summary</h3>
            <div id="portfolio-metrics">Loading...</div>
        </div>
        
        <div class="card">
            <h3>⚡ Risk Metrics</h3>
            <div id="risk-metrics">Loading...</div>
        </div>
    </div>

    <div class="chart-container">
        <h3>📊 Portfolio Value History</h3>
        <canvas id="portfolioChart" width="800" height="300"></canvas>
    </div>

    <div class="strategies-table">
        <h3 style="margin: 0; padding: 20px; background: #34495e; color: white;">🎯 Strategy Performance</h3>
        <table id="strategies-table">
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Type</th>
                    <th>Allocation</th>
                    <th>Daily P&L</th>
                    <th>Total P&L</th>
                    <th>Win Rate</th>
                    <th>Positions</th>
                    <th>Trades</th>
                </tr>
            </thead>
            <tbody id="strategies-body">
                <tr><td colspan="8" style="text-align: center; padding: 20px;">Loading strategies...</td></tr>
            </tbody>
        </table>
    </div>

    <div class="last-updated">
        Last Updated: <span id="last-updated">Loading...</span>
    </div>

    <script>
        // Auto-refresh dashboard data
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    updatePerformanceMetrics(data.performance);
                    updatePortfolioMetrics(data.portfolio);
                    updateStrategies(data.strategies);
                    document.getElementById('last-updated').textContent = new Date().toLocaleString();
                })
                .catch(error => console.error('Error:', error));
        }

        function updatePerformanceMetrics(metrics) {
            const container = document.getElementById('performance-metrics');
            container.innerHTML = `
                <div class="metric">
                    <span>Total Return:</span>
                    <span class="metric-value ${metrics.total_return.includes('-') ? 'negative' : ''}">${metrics.total_return}</span>
                </div>
                <div class="metric">
                    <span>Daily Return:</span>
                    <span class="metric-value ${metrics.daily_return.includes('-') ? 'negative' : ''}">${metrics.daily_return}</span>
                </div>
                <div class="metric">
                    <span>Sharpe Ratio:</span>
                    <span class="metric-value">${metrics.sharpe_ratio}</span>
                </div>
                <div class="metric">
                    <span>Max Drawdown:</span>
                    <span class="metric-value negative">${metrics.max_drawdown}</span>
                </div>
                <div class="metric">
                    <span>Win Rate:</span>
                    <span class="metric-value">${metrics.win_rate}</span>
                </div>
            `;
        }

        function updatePortfolioMetrics(metrics) {
            const container = document.getElementById('portfolio-metrics');
            container.innerHTML = `
                <div class="metric">
                    <span>Total Value:</span>
                    <span class="metric-value">${metrics.total_value}</span>
                </div>
                <div class="metric">
                    <span>Daily P&L:</span>
                    <span class="metric-value ${metrics.daily_pnl.includes('-') ? 'negative' : ''}">${metrics.daily_pnl}</span>
                </div>
                <div class="metric">
                    <span>Cash Balance:</span>
                    <span class="metric-value">${metrics.cash_balance}</span>
                </div>
                <div class="metric">
                    <span>Leverage:</span>
                    <span class="metric-value">${metrics.leverage}</span>
                </div>
            `;
        }

        function updateStrategies(strategies) {
            const tbody = document.getElementById('strategies-body');
            if (strategies.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center;">No strategies active</td></tr>';
                return;
            }
            
            tbody.innerHTML = strategies.map(strategy => `
                <tr>
                    <td><strong>${strategy.strategy_id}</strong></td>
                    <td><span style="background: #ecf0f1; padding: 4px 8px; border-radius: 4px; font-size: 0.8em;">${strategy.strategy_type}</span></td>
                    <td>${strategy.allocation}</td>
                    <td class="${strategy.daily_pnl.includes('-') ? 'negative' : 'metric-value'}">${strategy.daily_pnl}</td>
                    <td class="${strategy.total_pnl.includes('-') ? 'negative' : 'metric-value'}">${strategy.total_pnl}</td>
                    <td>${strategy.win_rate}</td>
                    <td>${strategy.active_positions}</td>
                    <td>${strategy.total_trades}</td>
                </tr>
            `).join('');
        }

        // Update dashboard every 5 seconds
        setInterval(updateDashboard, 5000);
        updateDashboard(); // Initial load
    </script>
</body>
</html>
"""