"""
Real-time performance monitoring dashboard for production trading.

Provides comprehensive performance tracking, real-time P&L monitoring,
risk metrics calculation, and anomaly detection for institutional trading.
"""
import statistics
import threading
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any
from ai_trading.logging import logger
from json import JSONDecodeError
try:
    import requests
    RequestException = requests.exceptions.RequestException
except ImportError:

    class RequestException(Exception):
        pass
COMMON_EXC = (TypeError, ValueError, KeyError, JSONDecodeError, RequestException, TimeoutError, ImportError)
from ..core.constants import DATA_PARAMETERS, PERFORMANCE_THRESHOLDS
from .alerting import AlertManager, AlertSeverity

class PerformanceMetrics:
    """
    Real-time performance metrics calculator.

    Calculates and tracks key trading performance metrics including
    Sharpe ratio, Sortino ratio, drawdown, and win rates.
    """

    def __init__(self, lookback_days: int=252):
        """Initialize performance metrics."""
        self.lookback_days = lookback_days
        self.returns = deque(maxlen=lookback_days)
        self.equity_curve = deque(maxlen=lookback_days)
        self.trades = deque(maxlen=1000)
        self.current_metrics = {}
        self.last_calculation = datetime.now(UTC)
        self.risk_free_rate = 0.02
        logger.info(f'PerformanceMetrics initialized with lookback_days={lookback_days}')

    def add_return(self, return_value: float, equity_value: float):
        """Add a new return data point."""
        try:
            self.returns.append(return_value)
            self.equity_curve.append(equity_value)
            self.last_calculation = datetime.now(UTC)
            if len(self.returns) >= 30:
                self._calculate_metrics()
        except COMMON_EXC as e:
            logger.error(f'Error adding return data: {e}')

    def add_trade(self, symbol: str, entry_time: datetime, exit_time: datetime, entry_price: float, exit_price: float, quantity: int, pnl: float, commission: float=0.0):
        """Add a completed trade."""
        try:
            trade = {'symbol': symbol, 'entry_time': entry_time, 'exit_time': exit_time, 'entry_price': entry_price, 'exit_price': exit_price, 'quantity': quantity, 'pnl': pnl, 'commission': commission, 'net_pnl': pnl - commission, 'return_pct': (exit_price - entry_price) / entry_price if entry_price > 0 else 0, 'hold_time': (exit_time - entry_time).total_seconds() / 3600}
            self.trades.append(trade)
            logger.debug(f'Trade added: {symbol} PnL=${pnl:.2f}')
        except COMMON_EXC as e:
            logger.error(f'Error adding trade: {e}')

    def calculate_sharpe_ratio(self, returns: list[float]=None) -> float:
        """Calculate Sharpe ratio."""
        try:
            returns_data = returns or list(self.returns)
            if len(returns_data) < 30:
                return 0.0
            mean_return = statistics.mean(returns_data)
            std_return = statistics.stdev(returns_data)
            if std_return == 0:
                return 0.0
            daily_risk_free = self.risk_free_rate / 252
            excess_return = mean_return - daily_risk_free
            sharpe = excess_return / std_return * 252 ** 0.5
            return sharpe
        except COMMON_EXC as e:
            logger.error(f'Error calculating Sharpe ratio: {e}')
            return 0.0

    def calculate_sortino_ratio(self, returns: list[float]=None) -> float:
        """Calculate Sortino ratio (using downside deviation)."""
        try:
            returns_data = returns or list(self.returns)
            if len(returns_data) < 30:
                return 0.0
            mean_return = statistics.mean(returns_data)
            negative_returns = [r for r in returns_data if r < 0]
            if not negative_returns:
                return float('inf') if mean_return > 0 else 0.0
            downside_std = statistics.stdev(negative_returns)
            if downside_std == 0:
                return 0.0
            daily_risk_free = self.risk_free_rate / 252
            excess_return = mean_return - daily_risk_free
            sortino = excess_return / downside_std * 252 ** 0.5
            return sortino
        except COMMON_EXC as e:
            logger.error(f'Error calculating Sortino ratio: {e}')
            return 0.0

    def calculate_max_drawdown(self, equity_curve: list[float]=None) -> tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        try:
            equity_data = equity_curve or list(self.equity_curve)
            if len(equity_data) < 2:
                return (0.0, 0)
            peak = equity_data[0]
            max_dd = 0.0
            max_dd_duration = 0
            current_dd_duration = 0
            for value in equity_data:
                if value > peak:
                    peak = value
                    current_dd_duration = 0
                else:
                    drawdown = (peak - value) / peak
                    max_dd = max(max_dd, drawdown)
                    current_dd_duration += 1
                    max_dd_duration = max(max_dd_duration, current_dd_duration)
            return (max_dd, max_dd_duration)
        except COMMON_EXC as e:
            logger.error(f'Error calculating max drawdown: {e}')
            return (0.0, 0)

    def calculate_win_rate(self) -> dict[str, float]:
        """Calculate win rate and related statistics."""
        try:
            if not self.trades:
                return {'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0}
            winning_trades = [t for t in self.trades if t['net_pnl'] > 0]
            losing_trades = [t for t in self.trades if t['net_pnl'] < 0]
            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
            avg_win = statistics.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0.0
            avg_loss = statistics.mean([abs(t['net_pnl']) for t in losing_trades]) if losing_trades else 0.0
            total_wins = sum((t['net_pnl'] for t in winning_trades))
            total_losses = sum((abs(t['net_pnl']) for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            return {'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': profit_factor, 'total_trades': len(self.trades), 'winning_trades': len(winning_trades), 'losing_trades': len(losing_trades)}
        except COMMON_EXC as e:
            logger.error(f'Error calculating win rate: {e}')
            return {'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0}

    def _calculate_metrics(self):
        """Calculate all performance metrics."""
        try:
            if len(self.returns) < 30:
                return
            sharpe = self.calculate_sharpe_ratio()
            sortino = self.calculate_sortino_ratio()
            max_dd, dd_duration = self.calculate_max_drawdown()
            win_stats = self.calculate_win_rate()
            total_return = self.equity_curve[-1] / self.equity_curve[0] - 1 if len(self.equity_curve) >= 2 else 0.0
            volatility = statistics.stdev(self.returns) * 252 ** 0.5 if len(self.returns) > 1 else 0.0
            self.current_metrics = {'sharpe_ratio': sharpe, 'sortino_ratio': sortino, 'max_drawdown': max_dd, 'drawdown_duration': dd_duration, 'total_return': total_return, 'annualized_volatility': volatility, 'win_rate': win_stats['win_rate'], 'profit_factor': win_stats['profit_factor'], 'total_trades': win_stats['total_trades'], 'avg_win': win_stats['avg_win'], 'avg_loss': win_stats['avg_loss'], 'last_updated': datetime.now(UTC)}
        except COMMON_EXC as e:
            logger.error(f'Error calculating performance metrics: {e}')

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return self.current_metrics.copy()

class RealTimePnLTracker:
    """
    Real-time P&L tracking and monitoring.

    Tracks unrealized and realized P&L, position-level returns,
    and provides real-time performance updates.
    """

    def __init__(self):
        """Initialize P&L tracker."""
        self.positions = {}
        self.daily_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.pnl_history = deque(maxlen=1000)
        self.daily_pnl_history = {}
        self.session_start_equity = 0.0
        self.session_high_equity = 0.0
        self.session_low_equity = 0.0
        self._lock = threading.RLock()
        logger.info('RealTimePnLTracker initialized')

    def update_position(self, symbol: str, quantity: int, avg_price: float, current_price: float, commission: float=0.0):
        """Update position with current market price."""
        try:
            with self._lock:
                if quantity == 0:
                    if symbol in self.positions:
                        del self.positions[symbol]
                else:
                    market_value = quantity * current_price
                    cost_basis = quantity * avg_price
                    unrealized_pnl = market_value - cost_basis - commission
                    self.positions[symbol] = {'symbol': symbol, 'quantity': quantity, 'avg_price': avg_price, 'current_price': current_price, 'market_value': market_value, 'cost_basis': cost_basis, 'unrealized_pnl': unrealized_pnl, 'commission': commission, 'last_updated': datetime.now(UTC)}
                self._calculate_unrealized_pnl()
        except COMMON_EXC as e:
            logger.error(f'Error updating position for {symbol}: {e}')

    def record_trade(self, symbol: str, quantity: int, price: float, commission: float, trade_type: str='unknown'):
        """Record a completed trade."""
        try:
            with self._lock:
                trade_pnl = 0.0
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    if trade_type == 'sell' or quantity < 0:
                        trade_quantity = abs(quantity)
                        trade_pnl = (price - pos['avg_price']) * trade_quantity - commission
                self.realized_pnl += trade_pnl
                self.daily_pnl += trade_pnl
                trade_record = {'timestamp': datetime.now(UTC), 'symbol': symbol, 'quantity': quantity, 'price': price, 'commission': commission, 'trade_pnl': trade_pnl, 'type': trade_type}
                self.pnl_history.append(trade_record)
                logger.debug(f'Trade recorded: {symbol} qty={quantity} pnl=${trade_pnl:.2f}')
        except COMMON_EXC as e:
            logger.error(f'Error recording trade: {e}')

    def start_new_session(self, starting_equity: float):
        """Start a new trading session."""
        try:
            with self._lock:
                self.session_start_equity = starting_equity
                self.session_high_equity = starting_equity
                self.session_low_equity = starting_equity
                self.daily_pnl = 0.0
                today = datetime.now(UTC).date()
                if self.daily_pnl != 0:
                    self.daily_pnl_history[today] = self.daily_pnl
                logger.info(f'New trading session started with equity ${starting_equity:,.2f}')
        except COMMON_EXC as e:
            logger.error(f'Error starting new session: {e}')

    def update_equity(self, current_equity: float):
        """Update current equity value."""
        try:
            with self._lock:
                self.session_high_equity = max(self.session_high_equity, current_equity)
                self.session_low_equity = min(self.session_low_equity, current_equity)
        except COMMON_EXC as e:
            logger.error(f'Error updating equity: {e}')

    def _calculate_unrealized_pnl(self):
        """Calculate total unrealized P&L from all positions."""
        try:
            self.unrealized_pnl = sum((pos['unrealized_pnl'] for pos in self.positions.values()))
        except COMMON_EXC as e:
            logger.error(f'Error calculating unrealized P&L: {e}')
            self.unrealized_pnl = 0.0

    def get_pnl_summary(self) -> dict[str, Any]:
        """Get comprehensive P&L summary."""
        try:
            with self._lock:
                total_pnl = self.realized_pnl + self.unrealized_pnl
                session_pnl = self.session_high_equity - self.session_start_equity if self.session_start_equity > 0 else 0
                session_drawdown = (self.session_high_equity - self.session_low_equity) / self.session_high_equity if self.session_high_equity > 0 else 0
                return {'realized_pnl': self.realized_pnl, 'unrealized_pnl': self.unrealized_pnl, 'total_pnl': total_pnl, 'daily_pnl': self.daily_pnl, 'session_pnl': session_pnl, 'session_drawdown': session_drawdown, 'session_high': self.session_high_equity, 'session_low': self.session_low_equity, 'open_positions': len(self.positions), 'total_trades': len(self.pnl_history), 'last_updated': datetime.now(UTC)}
        except COMMON_EXC as e:
            logger.error(f'Error getting P&L summary: {e}')
            return {'error': str(e)}

    def get_position_details(self) -> list[dict[str, Any]]:
        """Get detailed position information."""
        try:
            with self._lock:
                return list(self.positions.values())
        except COMMON_EXC as e:
            logger.error(f'Error getting position details: {e}')
            return []

class AnomalyDetector:
    """
    Performance anomaly detection system.

    Detects unusual patterns in trading performance, risk metrics,
    and system behavior to alert operators of potential issues.
    """

    def __init__(self, sensitivity: float=2.0):
        """Initialize anomaly detector."""
        self.sensitivity = sensitivity
        self.returns_history = deque(maxlen=252)
        self.pnl_history = deque(maxlen=252)
        self.volatility_history = deque(maxlen=50)
        self.return_threshold = 0.0
        self.pnl_threshold = 0.0
        self.volatility_threshold = 0.0
        self.recent_anomalies = deque(maxlen=100)
        logger.info(f'AnomalyDetector initialized with sensitivity={sensitivity}')

    def update_data(self, daily_return: float, daily_pnl: float, volatility: float):
        """Update historical data for anomaly detection."""
        try:
            self.returns_history.append(daily_return)
            self.pnl_history.append(daily_pnl)
            self.volatility_history.append(volatility)
            self._update_thresholds()
        except COMMON_EXC as e:
            logger.error(f'Error updating anomaly detector data: {e}')

    def detect_anomalies(self, current_return: float, current_pnl: float, current_volatility: float) -> list[dict[str, Any]]:
        """Detect performance anomalies."""
        try:
            anomalies = []
            if len(self.returns_history) >= 30:
                mean_return = statistics.mean(self.returns_history)
                std_return = statistics.stdev(self.returns_history)
                if abs(current_return - mean_return) > self.sensitivity * std_return:
                    anomalies.append({'type': 'unusual_return', 'severity': 'high' if abs(current_return) > 0.05 else 'medium', 'value': current_return, 'baseline': mean_return, 'threshold': self.sensitivity * std_return, 'description': f'Daily return {current_return:.2%} is {abs(current_return - mean_return) / std_return:.1f} std devs from mean'})
            if len(self.pnl_history) >= 30:
                mean_pnl = statistics.mean(self.pnl_history)
                std_pnl = statistics.stdev(self.pnl_history)
                if abs(current_pnl - mean_pnl) > self.sensitivity * std_pnl:
                    anomalies.append({'type': 'unusual_pnl', 'severity': 'high' if abs(current_pnl) > 10000 else 'medium', 'value': current_pnl, 'baseline': mean_pnl, 'threshold': self.sensitivity * std_pnl, 'description': f'Daily P&L ${current_pnl:,.2f} is unusual compared to recent performance'})
            if len(self.volatility_history) >= 20:
                mean_vol = statistics.mean(self.volatility_history)
                std_vol = statistics.stdev(self.volatility_history)
                if abs(current_volatility - mean_vol) > self.sensitivity * std_vol:
                    anomalies.append({'type': 'unusual_volatility', 'severity': 'high' if current_volatility > 0.5 else 'medium', 'value': current_volatility, 'baseline': mean_vol, 'threshold': self.sensitivity * std_vol, 'description': f'Volatility {current_volatility:.1%} is significantly different from recent levels'})
            for anomaly in anomalies:
                anomaly['timestamp'] = datetime.now(UTC)
                self.recent_anomalies.append(anomaly)
            return anomalies
        except COMMON_EXC as e:
            logger.error(f'Error detecting anomalies: {e}')
            return []

    def _update_thresholds(self):
        """Update anomaly detection thresholds."""
        try:
            if len(self.returns_history) >= 30:
                self.return_threshold = statistics.stdev(self.returns_history) * self.sensitivity
            if len(self.pnl_history) >= 30:
                self.pnl_threshold = statistics.stdev(self.pnl_history) * self.sensitivity
            if len(self.volatility_history) >= 20:
                self.volatility_threshold = statistics.stdev(self.volatility_history) * self.sensitivity
        except COMMON_EXC as e:
            logger.error(f'Error updating anomaly thresholds: {e}')

    def get_recent_anomalies(self, hours: int=24) -> list[dict[str, Any]]:
        """Get recent anomalies within specified time window."""
        try:
            cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
            return [anomaly for anomaly in self.recent_anomalies if anomaly['timestamp'] >= cutoff_time]
        except COMMON_EXC as e:
            logger.error(f'Error getting recent anomalies: {e}')
            return []

class PerformanceDashboard:
    """
    Comprehensive performance monitoring dashboard.

    Integrates all performance monitoring components and provides
    unified interface for real-time trading performance analysis.
    """

    def __init__(self, alert_manager: AlertManager=None):
        """Initialize performance dashboard."""
        self.metrics = PerformanceMetrics()
        self.pnl_tracker = RealTimePnLTracker()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = alert_manager
        self.last_update = datetime.now(UTC)
        self.update_interval = DATA_PARAMETERS['HEALTH_CHECK_INTERVAL']
        self.alert_thresholds = PERFORMANCE_THRESHOLDS.copy()
        logger.info('PerformanceDashboard initialized')

    def update_performance(self, equity_value: float, daily_return: float=None, volatility: float=None):
        """Update all performance metrics."""
        try:
            if daily_return is not None:
                self.metrics.add_return(daily_return, equity_value)
            self.pnl_tracker.update_equity(equity_value)
            if daily_return is not None and volatility is not None:
                pnl_summary = self.pnl_tracker.get_pnl_summary()
                daily_pnl = pnl_summary.get('daily_pnl', 0.0)
                anomalies = self.anomaly_detector.detect_anomalies(daily_return, daily_pnl, volatility)
                if anomalies and self.alert_manager:
                    for anomaly in anomalies:
                        severity = AlertSeverity.CRITICAL if anomaly['severity'] == 'high' else AlertSeverity.WARNING
                        self.alert_manager.send_performance_alert(anomaly['type'], anomaly['value'], anomaly['threshold'], severity)
            self._check_performance_thresholds()
            self.last_update = datetime.now(UTC)
        except COMMON_EXC as e:
            logger.error(f'Error updating performance dashboard: {e}')

    def _check_performance_thresholds(self):
        """Check performance metrics against alert thresholds."""
        try:
            if not self.alert_manager:
                return
            current_metrics = self.metrics.get_current_metrics()
            sharpe = current_metrics.get('sharpe_ratio', 0)
            if sharpe < self.alert_thresholds['MIN_SHARPE_RATIO']:
                self.alert_manager.send_performance_alert('Sharpe Ratio', sharpe, self.alert_thresholds['MIN_SHARPE_RATIO'], AlertSeverity.WARNING)
            max_dd = current_metrics.get('max_drawdown', 0)
            if max_dd > self.alert_thresholds['MAX_DRAWDOWN']:
                self.alert_manager.send_performance_alert('Maximum Drawdown', f'{max_dd:.2%}', f"{self.alert_thresholds['MAX_DRAWDOWN']:.2%}", AlertSeverity.CRITICAL)
            win_rate = current_metrics.get('win_rate', 0)
            if win_rate < self.alert_thresholds['MIN_WIN_RATE']:
                self.alert_manager.send_performance_alert('Win Rate', f'{win_rate:.2%}', f"{self.alert_thresholds['MIN_WIN_RATE']:.2%}", AlertSeverity.WARNING)
        except COMMON_EXC as e:
            logger.error(f'Error checking performance thresholds: {e}')

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Get comprehensive dashboard summary."""
        try:
            current_metrics = self.metrics.get_current_metrics()
            pnl_summary = self.pnl_tracker.get_pnl_summary()
            position_details = self.pnl_tracker.get_position_details()
            recent_anomalies = self.anomaly_detector.get_recent_anomalies(24)
            return {'timestamp': datetime.now(UTC), 'performance_metrics': current_metrics, 'pnl_summary': pnl_summary, 'position_count': len(position_details), 'recent_anomalies': len(recent_anomalies), 'anomaly_details': recent_anomalies[-5:] if recent_anomalies else [], 'system_status': {'last_update': self.last_update, 'update_interval': self.update_interval, 'alerts_configured': self.alert_manager is not None}}
        except COMMON_EXC as e:
            logger.error(f'Error getting dashboard summary: {e}')
            return {'error': str(e)}

    def get_detailed_positions(self) -> list[dict[str, Any]]:
        """Get detailed position information."""
        return self.pnl_tracker.get_position_details()

    def add_trade(self, symbol: str, entry_time: datetime, exit_time: datetime, entry_price: float, exit_price: float, quantity: int, pnl: float, commission: float=0.0):
        """Add a completed trade to tracking."""
        try:
            self.metrics.add_trade(symbol, entry_time, exit_time, entry_price, exit_price, quantity, pnl, commission)
            trade_type = 'sell' if quantity < 0 else 'buy'
            self.pnl_tracker.record_trade(symbol, quantity, exit_price, commission, trade_type)
        except COMMON_EXC as e:
            logger.error(f'Error adding trade to dashboard: {e}')

    def update_position(self, symbol: str, quantity: int, avg_price: float, current_price: float, commission: float=0.0):
        """Update position information."""
        try:
            self.pnl_tracker.update_position(symbol, quantity, avg_price, current_price, commission)
        except COMMON_EXC as e:
            logger.error(f'Error updating position in dashboard: {e}')
