"""
Dashboard data provider and real-time metrics.

Provides data aggregation and formatting for institutional
trading dashboards and real-time monitoring interfaces.
"""
from datetime import UTC, datetime, timedelta
from typing import Any
from ai_trading.logging import logger
from .alerts import AlertManager, AlertSeverity
from .metrics import MetricsCollector, PerformanceMonitor

class RealtimeMetrics:
    """
    Real-time metrics aggregation and calculation.

    Provides live calculation of key performance indicators
    and risk metrics for dashboard display.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize real-time metrics."""
        self.metrics_collector = metrics_collector
        self.cache_ttl = 30
        self._cache = {}
        self._cache_timestamps = {}
        logger.info('RealtimeMetrics initialized')

    def get_current_pnl(self) -> dict[str, float]:
        """Get current P&L metrics."""
        cache_key = 'current_pnl'
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        try:
            recent_trades = [m for m in self.metrics_collector.trade_metrics if (datetime.now(UTC) - m['timestamp']).total_seconds() < 3600]
            if not recent_trades:
                result = {'realized_pnl': 0.0, 'trade_count': 0, 'win_rate': 0.0}
            else:
                realized_pnl = sum((trade.get('pnl', 0) for trade in recent_trades))
                trade_count = len(recent_trades)
                winning_trades = len([t for t in recent_trades if t.get('pnl', 0) > 0])
                win_rate = winning_trades / trade_count if trade_count > 0 else 0
                result = {'realized_pnl': realized_pnl, 'trade_count': trade_count, 'win_rate': win_rate, 'avg_pnl_per_trade': realized_pnl / trade_count if trade_count > 0 else 0}
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = datetime.now(UTC)
            return result
        except (ValueError, TypeError) as e:
            logger.error(f'Error calculating current P&L: {e}')
            return {'realized_pnl': 0.0, 'trade_count': 0, 'win_rate': 0.0}

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary metrics."""
        cache_key = 'portfolio_summary'
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        try:
            portfolio_metrics = list(self.metrics_collector.portfolio_metrics)
            if not portfolio_metrics:
                result = {'total_value': 0.0, 'day_change': 0.0, 'day_change_pct': 0.0, 'unrealized_pnl': 0.0}
            else:
                latest = portfolio_metrics[-1]
                result = {'total_value': latest.get('total_value', 0), 'day_change': latest.get('day_change', 0), 'day_change_pct': latest.get('day_change_pct', 0), 'unrealized_pnl': latest.get('unrealized_pnl', 0), 'cash': latest.get('cash', 0), 'positions_count': latest.get('positions_count', 0)}
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = datetime.now(UTC)
            return result
        except (ValueError, TypeError) as e:
            logger.error(f'Error calculating portfolio summary: {e}')
            return {'total_value': 0.0, 'day_change': 0.0, 'day_change_pct': 0.0}

    def get_risk_summary(self) -> dict[str, float]:
        """Get risk summary metrics."""
        cache_key = 'risk_summary'
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        try:
            risk_metrics = list(self.metrics_collector.risk_metrics)
            if not risk_metrics:
                result = {'var_95': 0.0, 'max_drawdown': 0.0, 'current_drawdown': 0.0, 'sharpe_ratio': 0.0}
            else:
                latest = risk_metrics[-1]
                result = {'var_95': latest.get('var_95', 0), 'max_drawdown': latest.get('max_drawdown', 0), 'current_drawdown': latest.get('current_drawdown', 0), 'sharpe_ratio': latest.get('sharpe_ratio', 0), 'volatility': latest.get('volatility', 0)}
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = datetime.now(UTC)
            return result
        except (ValueError, TypeError) as e:
            logger.error(f'Error calculating risk summary: {e}')
            return {'var_95': 0.0, 'max_drawdown': 0.0, 'current_drawdown': 0.0}

    def get_execution_summary(self) -> dict[str, Any]:
        """Get execution summary metrics."""
        cache_key = 'execution_summary'
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        try:
            execution_metrics = list(self.metrics_collector.execution_metrics)
            if not execution_metrics:
                result = {'orders_submitted': 0, 'orders_filled': 0, 'fill_rate': 0.0, 'avg_fill_time': 0.0}
            else:
                latest = execution_metrics[-1]
                result = {'orders_submitted': latest.get('orders_submitted', 0), 'orders_filled': latest.get('orders_filled', 0), 'fill_rate': latest.get('fill_rate', 0), 'avg_fill_time': latest.get('average_fill_time', 0), 'total_volume': latest.get('total_volume', 0)}
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = datetime.now(UTC)
            return result
        except (ValueError, TypeError) as e:
            logger.error(f'Error calculating execution summary: {e}')
            return {'orders_submitted': 0, 'orders_filled': 0, 'fill_rate': 0.0}

    def _is_cached(self, cache_key: str) -> bool:
        """Check if metric is cached and still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        age = (datetime.now(UTC) - self._cache_timestamps[cache_key]).total_seconds()
        return age < self.cache_ttl

class DashboardDataProvider:
    """
    Comprehensive dashboard data provider.

    Aggregates and formats data from multiple sources for
    institutional trading dashboard display.
    """

    def __init__(self, metrics_collector: MetricsCollector, performance_monitor: PerformanceMonitor, alert_manager: AlertManager):
        """Initialize dashboard data provider."""
        self.metrics_collector = metrics_collector
        self.performance_monitor = performance_monitor
        self.alert_manager = alert_manager
        self.realtime_metrics = RealtimeMetrics(metrics_collector)
        logger.info('DashboardDataProvider initialized')

    def get_dashboard_data(self, time_range: timedelta=None) -> dict[str, Any]:
        """
        Get comprehensive dashboard data.

        Args:
            time_range: Time range for historical data

        Returns:
            Complete dashboard data dictionary
        """
        try:
            time_range = time_range or timedelta(hours=24)
            dashboard_data = {'timestamp': datetime.now(UTC).isoformat(), 'time_range_hours': time_range.total_seconds() / 3600, 'realtime': {'pnl': self.realtime_metrics.get_current_pnl(), 'portfolio': self.realtime_metrics.get_portfolio_summary(), 'risk': self.realtime_metrics.get_risk_summary(), 'execution': self.realtime_metrics.get_execution_summary()}, 'performance': self.performance_monitor.get_performance_report(time_range), 'alerts': self._get_alert_summary(), 'system_health': self._get_system_health(), 'charts': self._get_chart_data(time_range)}
            return dashboard_data
        except (ValueError, TypeError) as e:
            logger.error(f'Error generating dashboard data: {e}')
            return {'timestamp': datetime.now(UTC).isoformat(), 'error': str(e), 'realtime': {}, 'performance': {}, 'alerts': {}, 'system_health': {}, 'charts': {}}

    def get_trading_activity_summary(self, time_range: timedelta=None) -> dict[str, Any]:
        """Get trading activity summary."""
        try:
            time_range = time_range or timedelta(hours=24)
            cutoff_time = datetime.now(UTC) - time_range
            recent_trades = [m for m in self.metrics_collector.trade_metrics if m['timestamp'] >= cutoff_time]
            if not recent_trades:
                return {'total_trades': 0, 'total_volume': 0.0, 'symbols_traded': 0, 'strategies_active': 0}
            total_volume = sum((trade.get('notional_value', 0) for trade in recent_trades))
            symbols = {trade.get('symbol') for trade in recent_trades if trade.get('symbol')}
            strategies = {trade.get('strategy_id') for trade in recent_trades if trade.get('strategy_id')}
            pnl_values = [trade.get('pnl', 0) for trade in recent_trades]
            winning_trades = [pnl for pnl in pnl_values if pnl > 0]
            summary = {'total_trades': len(recent_trades), 'total_volume': total_volume, 'symbols_traded': len(symbols), 'strategies_active': len(strategies), 'gross_pnl': sum(pnl_values), 'winning_trades': len(winning_trades), 'win_rate': len(winning_trades) / len(recent_trades) if recent_trades else 0, 'avg_trade_size': total_volume / len(recent_trades) if recent_trades else 0, 'largest_win': max(pnl_values) if pnl_values else 0, 'largest_loss': min(pnl_values) if pnl_values else 0}
            return summary
        except (ValueError, TypeError) as e:
            logger.error(f'Error generating trading activity summary: {e}')
            return {'total_trades': 0, 'total_volume': 0.0}

    def _get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary for dashboard."""
        try:
            active_alerts = self.alert_manager.get_active_alerts()
            severity_counts = {}
            for severity in AlertSeverity:
                severity_counts[severity.value] = len([a for a in active_alerts if a.severity == severity])
            recent_cutoff = datetime.now(UTC) - timedelta(hours=1)
            recent_alerts = [a for a in self.alert_manager.alerts if a.timestamp >= recent_cutoff]
            return {'active_count': len(active_alerts), 'severity_breakdown': severity_counts, 'recent_count': len(recent_alerts), 'critical_alerts': [a.to_dict() for a in active_alerts if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]][:5]}
        except (ValueError, TypeError) as e:
            logger.error(f'Error generating alert summary: {e}')
            return {'active_count': 0, 'severity_breakdown': {}, 'recent_count': 0}

    def _get_system_health(self) -> dict[str, Any]:
        """Get system health metrics."""
        try:
            system_metrics = list(self.metrics_collector.system_metrics)
            if not system_metrics:
                return {'status': 'unknown', 'cpu_usage': 0.0, 'memory_usage': 0.0, 'latency': 0.0}
            latest = system_metrics[-1]
            cpu_usage = latest.get('cpu_usage', 0)
            memory_usage = latest.get('memory_usage', 0)
            latency = latest.get('latency', 0)
            status = 'healthy'
            if cpu_usage > 80 or memory_usage > 80 or latency > 1000:
                status = 'warning'
            if cpu_usage > 95 or memory_usage > 95 or latency > 5000:
                status = 'critical'
            return {'status': status, 'cpu_usage': cpu_usage, 'memory_usage': memory_usage, 'latency': latency, 'last_updated': latest['timestamp'].isoformat()}
        except (ValueError, TypeError) as e:
            logger.error(f'Error generating system health: {e}')
            return {'status': 'error', 'cpu_usage': 0, 'memory_usage': 0}

    def _get_chart_data(self, time_range: timedelta) -> dict[str, list]:
        """Get historical data for charts."""
        try:
            cutoff_time = datetime.now(UTC) - time_range
            portfolio_data = [{'timestamp': m['timestamp'].isoformat(), 'value': m.get('total_value', 0), 'pnl': m.get('unrealized_pnl', 0) + m.get('realized_pnl', 0)} for m in self.metrics_collector.portfolio_metrics if m['timestamp'] >= cutoff_time]
            risk_data = [{'timestamp': m['timestamp'].isoformat(), 'var_95': m.get('var_95', 0), 'drawdown': m.get('current_drawdown', 0), 'sharpe': m.get('sharpe_ratio', 0)} for m in self.metrics_collector.risk_metrics if m['timestamp'] >= cutoff_time]
            volume_data = []
            hourly_volume = {}
            for trade in self.metrics_collector.trade_metrics:
                if trade['timestamp'] >= cutoff_time:
                    hour_key = trade['timestamp'].replace(minute=0, second=0, microsecond=0)
                    if hour_key not in hourly_volume:
                        hourly_volume[hour_key] = {'volume': 0, 'count': 0}
                    hourly_volume[hour_key]['volume'] += trade.get('notional_value', 0)
                    hourly_volume[hour_key]['count'] += 1
            volume_data = [{'timestamp': hour.isoformat(), 'volume': data['volume'], 'trade_count': data['count']} for hour, data in sorted(hourly_volume.items())]
            return {'portfolio_value': portfolio_data, 'risk_metrics': risk_data, 'trading_volume': volume_data}
        except (ValueError, TypeError) as e:
            logger.error(f'Error generating chart data: {e}')
            return {'portfolio_value': [], 'risk_metrics': [], 'trading_volume': []}