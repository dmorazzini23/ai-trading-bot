"""Performance metrics for trading results with numerical stability."""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, TYPE_CHECKING
from importlib.util import find_spec

from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.logging import logger
HAS_PANDAS = find_spec("pandas") is not None
HAS_NUMPY = True

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

def compute_basic_metrics(df) -> dict[str, float]:
    """Return Sharpe ratio and max drawdown from ``df`` with a ``return`` column."""
    pd = load_pandas()
    import numpy as np
    if 'return' not in df:
        return {'sharpe': 0.0, 'max_drawdown': 0.0}
    ret = df['return'].astype(float)
    if ret.empty:
        return {'sharpe': 0.0, 'max_drawdown': 0.0}
    epsilon = 1e-08
    mean_return = ret.mean()
    std_return = ret.std()
    if std_return <= epsilon or pd.isna(std_return):
        sharpe = 0.0
    else:
        sharpe = mean_return / std_return * np.sqrt(252)
    cumulative = (1 + ret.fillna(0)).cumprod()
    if len(cumulative) == 0:
        max_dd = 0.0
    else:
        drawdown = cumulative.cummax() - cumulative
        max_dd = float(drawdown.max()) if not drawdown.empty else 0.0
    return {'sharpe': float(sharpe), 'max_drawdown': float(max_dd)}

def compute_advanced_metrics(df: 'pd.DataFrame') -> dict[str, float]:
    """Compute advanced performance metrics with numerical stability."""
    pd = load_pandas()
    import numpy as np

    if 'return' not in df:
        return {'sortino': 0.0, 'calmar': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0}
    ret = df['return'].astype(float).fillna(0)
    if ret.empty:
        return {'sortino': 0.0, 'calmar': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0}
    epsilon = 1e-08
    downside_returns = ret[ret < 0]
    if len(downside_returns) == 0:
        downside_std = epsilon
    else:
        downside_std = max(downside_returns.std(), epsilon)
    sortino = ret.mean() / downside_std * np.sqrt(252)
    annual_return = ret.mean() * 252
    basic_metrics = compute_basic_metrics(df)
    max_dd = max(basic_metrics['max_drawdown'], epsilon)
    calmar = annual_return / max_dd
    winning_trades = (ret > 0).sum()
    total_trades = len(ret)
    win_rate = winning_trades / max(total_trades, 1) * 100
    gross_profit = ret[ret > 0].sum()
    gross_loss = abs(ret[ret < 0].sum())
    profit_factor = gross_profit / max(gross_loss, epsilon)
    return {
        'sortino': float(sortino),
        'calmar': float(calmar),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
    }

def safe_divide(numerator: float, denominator: float, default: float=0.0) -> float:
    """Safe division with epsilon protection to prevent division by zero."""
    epsilon = 1e-08
    if abs(denominator) < epsilon:
        return default
    return numerator / denominator

def calculate_atr(df: 'pd.DataFrame', period: int = 14) -> 'pd.Series':
    """Calculate Average True Range with numerical stability."""
    pd = load_pandas()
    import numpy as np

    if df.empty or not all((col in df.columns for col in ['high', 'low', 'close'])):
        return pd.Series(dtype=float)
    epsilon = 1e-08
    high = df['high'].ffill()
    low = df['low'].ffill()
    close = df['close'].ffill()
    high = np.maximum(high, epsilon)
    low = np.maximum(low, epsilon)
    close = np.maximum(close, epsilon)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr.fillna(epsilon)

class MetricsCollector:
    """
    Comprehensive metrics collection for trading operations.
    
    Collects and tracks various performance metrics, system metrics,
    and trading-related measurements for monitoring and analysis.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.counters: dict[str, int] = defaultdict(int)
        self.histograms: dict[str, list] = defaultdict(list)
        self.gauges: dict[str, float] = {}
        self.start_time = time.time()
        logger.info('MetricsCollector initialized')

    def inc_counter(self, name: str, value: int=1, labels: dict[str, str] | None=None):
        """Increment a counter metric."""
        key = f"{name}_{hash(str(labels) if labels else '')}"
        self.counters[key] += value

    def observe_latency(self, name: str, latency_ms: float, labels: dict[str, str] | None=None):
        """Record a latency observation."""
        key = f"{name}_{hash(str(labels) if labels else '')}"
        self.histograms[key].append(latency_ms)
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-500:]

    def gauge_set(self, name: str, value: float, labels: dict[str, str] | None=None):
        """Set a gauge metric value."""
        key = f"{name}_{hash(str(labels) if labels else '')}"
        self.gauges[key] = value

    def record_trade_metrics(self, symbol: str, side: str, quantity: float, price: float, latency_ms: float, success: bool):
        """Record comprehensive trade execution metrics."""
        labels = {'symbol': symbol, 'side': side}
        if success:
            self.inc_counter('trades_executed', labels=labels)
            self.observe_latency('trade_latency', latency_ms, labels)
            self.gauge_set('last_trade_price', price, labels)
            self.gauge_set('last_trade_quantity', quantity, labels)
        else:
            self.inc_counter('trades_failed', labels=labels)
        self.gauge_set('last_trade_timestamp', time.time(), labels)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a summary of all collected metrics."""
        import numpy as np

        summary = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms_stats': {},
            'uptime_seconds': time.time() - self.start_time,
        }
        for name, values in self.histograms.items():
            if values:
                summary['histograms_stats'][name] = {
                    'count': len(values),
                    'avg': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                }
        return summary

class PerformanceMonitor:
    """
    Unified performance monitoring for trading operations.
    
    Combines metrics collection with performance analysis and alerting.
    This is the primary monitoring interface for the trading system.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics_collector = MetricsCollector()
        self.trade_history: list = []
        self.performance_cache: dict[str, Any] = {}
        self.last_cache_update = 0
        self.cache_ttl = 60
        logger.info('PerformanceMonitor initialized')

    def inc_counter(self, name: str, value: int=1, labels: dict[str, str] | None=None):
        """Increment a counter metric."""
        self.metrics_collector.inc_counter(name, value, labels)

    def observe_latency(self, name: str, latency_ms: float, labels: dict[str, str] | None=None):
        """Record a latency observation."""
        self.metrics_collector.observe_latency(name, latency_ms, labels)

    def gauge_set(self, name: str, value: float, labels: dict[str, str] | None=None):
        """Set a gauge metric value."""
        self.metrics_collector.gauge_set(name, value, labels)

    def record_trade(self, trade_data: dict[str, Any]):
        """Record a trade for performance analysis."""
        trade_data['timestamp'] = trade_data.get('timestamp', datetime.now(UTC))
        self.trade_history.append(trade_data)
        if len(self.trade_history) > 10000:
            self.trade_history = self.trade_history[-5000:]
        self.metrics_collector.record_trade_metrics(symbol=trade_data.get('symbol', 'UNKNOWN'), side=trade_data.get('side', 'UNKNOWN'), quantity=trade_data.get('quantity', 0), price=trade_data.get('price', 0), latency_ms=trade_data.get('latency_ms', 0), success=trade_data.get('success', False))

    def get_performance_metrics(self, force_refresh: bool=False) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        current_time = time.time()
        if not force_refresh and current_time - self.last_cache_update < self.cache_ttl:
            return self.performance_cache
        pd = load_pandas()
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            if 'return' in df.columns:
                basic_metrics = compute_basic_metrics(df)
                advanced_metrics = compute_advanced_metrics(df)
            else:
                basic_metrics = {'sharpe': 0.0, 'max_drawdown': 0.0}
                advanced_metrics = {'sortino': 0.0, 'calmar': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0}
        else:
            basic_metrics = {'sharpe': 0.0, 'max_drawdown': 0.0}
            advanced_metrics = {'sortino': 0.0, 'calmar': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0}
        system_metrics = self.metrics_collector.get_metrics_summary()
        self.performance_cache = {**basic_metrics, **advanced_metrics, **system_metrics, 'total_trades': len(self.trade_history), 'cache_timestamp': current_time}
        self.last_cache_update = current_time
        return self.performance_cache
