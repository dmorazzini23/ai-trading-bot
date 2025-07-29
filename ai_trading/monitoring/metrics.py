"""
Comprehensive metrics collection and performance monitoring.

Provides real-time metrics collection, performance analysis,
and institutional-grade monitoring capabilities.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import statistics
import threading
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..core.constants import DATA_PARAMETERS, PERFORMANCE_THRESHOLDS


class MetricsCollector:
    """
    Comprehensive metrics collection system.
    
    Collects, aggregates, and stores performance metrics
    for real-time monitoring and historical analysis.
    """
    
    def __init__(self, retention_days: int = None):
        """Initialize metrics collector."""
        # AI-AGENT-REF: Institutional metrics collection system
        self.retention_days = retention_days or DATA_PARAMETERS["METRICS_RETENTION_DAYS"]
        
        # Time-series storage for different metric types
        self.trade_metrics = deque(maxlen=10000)
        self.portfolio_metrics = deque(maxlen=10000)
        self.risk_metrics = deque(maxlen=10000)
        self.execution_metrics = deque(maxlen=10000)
        self.system_metrics = deque(maxlen=10000)
        
        # Real-time counters
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Cleanup thread
        self._cleanup_thread = None
        self._cleanup_running = False
        
        logger.info(f"MetricsCollector initialized with {self.retention_days} day retention")
    
    def record_trade_metric(self, symbol: str, side: str, quantity: int, price: float, 
                           pnl: float, execution_time: float, **kwargs):
        """Record trade execution metrics."""
        try:
            with self._lock:
                metric = {
                    "timestamp": datetime.now(),
                    "type": "trade",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "pnl": pnl,
                    "execution_time": execution_time,
                    "notional_value": abs(quantity * price),
                    "strategy_id": kwargs.get("strategy_id"),
                    "order_id": kwargs.get("order_id"),
                    "slippage": kwargs.get("slippage", 0.0),
                    "commission": kwargs.get("commission", 0.0),
                }
                
                self.trade_metrics.append(metric)
                
                # Update counters
                self.counters[f"trades_total"] += 1
                self.counters[f"trades_{side.lower()}"] += 1
                self.counters[f"trades_{symbol}"] += 1
                
                # Update gauges
                self.gauges["last_trade_pnl"] = pnl
                self.gauges["last_execution_time"] = execution_time
                
                # Update histograms
                self.histograms["trade_pnl"].append(pnl)
                self.histograms["execution_times"].append(execution_time)
                
                logger.debug(f"Trade metric recorded: {symbol} {side} {quantity}@{price}")
                
        except Exception as e:
            logger.error(f"Error recording trade metric: {e}")
    
    def record_portfolio_metric(self, total_value: float, unrealized_pnl: float, 
                               realized_pnl: float, cash: float, **kwargs):
        """Record portfolio performance metrics."""
        try:
            with self._lock:
                metric = {
                    "timestamp": datetime.now(),
                    "type": "portfolio",
                    "total_value": total_value,
                    "unrealized_pnl": unrealized_pnl,
                    "realized_pnl": realized_pnl,
                    "cash": cash,
                    "invested_value": total_value - cash,
                    "day_change": kwargs.get("day_change", 0.0),
                    "day_change_pct": kwargs.get("day_change_pct", 0.0),
                    "positions_count": kwargs.get("positions_count", 0),
                }
                
                self.portfolio_metrics.append(metric)
                
                # Update gauges
                self.gauges["portfolio_value"] = total_value
                self.gauges["unrealized_pnl"] = unrealized_pnl
                self.gauges["realized_pnl"] = realized_pnl
                self.gauges["cash_balance"] = cash
                
                logger.debug(f"Portfolio metric recorded: ${total_value:,.2f}")
                
        except Exception as e:
            logger.error(f"Error recording portfolio metric: {e}")
    
    def record_risk_metric(self, var_95: float, var_99: float, max_drawdown: float, 
                          current_drawdown: float, **kwargs):
        """Record risk management metrics."""
        try:
            with self._lock:
                metric = {
                    "timestamp": datetime.now(),
                    "type": "risk",
                    "var_95": var_95,
                    "var_99": var_99,
                    "max_drawdown": max_drawdown,
                    "current_drawdown": current_drawdown,
                    "sharpe_ratio": kwargs.get("sharpe_ratio", 0.0),
                    "sortino_ratio": kwargs.get("sortino_ratio", 0.0),
                    "beta": kwargs.get("beta", 1.0),
                    "volatility": kwargs.get("volatility", 0.0),
                }
                
                self.risk_metrics.append(metric)
                
                # Update gauges
                self.gauges["var_95"] = var_95
                self.gauges["max_drawdown"] = max_drawdown
                self.gauges["current_drawdown"] = current_drawdown
                
                logger.debug(f"Risk metric recorded: VaR95={var_95:.4f}, DD={current_drawdown:.4f}")
                
        except Exception as e:
            logger.error(f"Error recording risk metric: {e}")
    
    def record_execution_metric(self, orders_submitted: int, orders_filled: int, 
                               orders_cancelled: int, average_fill_time: float, **kwargs):
        """Record execution engine metrics."""
        try:
            with self._lock:
                metric = {
                    "timestamp": datetime.now(),
                    "type": "execution",
                    "orders_submitted": orders_submitted,
                    "orders_filled": orders_filled,
                    "orders_cancelled": orders_cancelled,
                    "orders_rejected": kwargs.get("orders_rejected", 0),
                    "average_fill_time": average_fill_time,
                    "fill_rate": orders_filled / orders_submitted if orders_submitted > 0 else 0,
                    "total_volume": kwargs.get("total_volume", 0.0),
                }
                
                self.execution_metrics.append(metric)
                
                # Update gauges
                self.gauges["orders_submitted"] = orders_submitted
                self.gauges["orders_filled"] = orders_filled
                self.gauges["average_fill_time"] = average_fill_time
                
                logger.debug(f"Execution metric recorded: {orders_filled}/{orders_submitted} filled")
                
        except Exception as e:
            logger.error(f"Error recording execution metric: {e}")
    
    def record_system_metric(self, cpu_usage: float, memory_usage: float, 
                            latency: float, **kwargs):
        """Record system performance metrics."""
        try:
            with self._lock:
                metric = {
                    "timestamp": datetime.now(),
                    "type": "system",
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "latency": latency,
                    "disk_usage": kwargs.get("disk_usage", 0.0),
                    "network_io": kwargs.get("network_io", 0.0),
                    "active_connections": kwargs.get("active_connections", 0),
                }
                
                self.system_metrics.append(metric)
                
                # Update gauges
                self.gauges["cpu_usage"] = cpu_usage
                self.gauges["memory_usage"] = memory_usage
                self.gauges["latency"] = latency
                
                logger.debug(f"System metric recorded: CPU={cpu_usage:.1f}%, MEM={memory_usage:.1f}%")
                
        except Exception as e:
            logger.error(f"Error recording system metric: {e}")
    
    def get_metrics_summary(self, metric_type: str = None, time_range: timedelta = None) -> Dict:
        """
        Get summary of collected metrics.
        
        Args:
            metric_type: Type of metrics to summarize (trade, portfolio, risk, execution, system)
            time_range: Time range for metrics (default: last 24 hours)
            
        Returns:
            Dictionary containing metrics summary
        """
        try:
            time_range = time_range or timedelta(hours=24)
            cutoff_time = datetime.now() - time_range
            
            with self._lock:
                # Select appropriate metric collection
                if metric_type == "trade":
                    metrics = [m for m in self.trade_metrics if m["timestamp"] >= cutoff_time]
                elif metric_type == "portfolio":
                    metrics = [m for m in self.portfolio_metrics if m["timestamp"] >= cutoff_time]
                elif metric_type == "risk":
                    metrics = [m for m in self.risk_metrics if m["timestamp"] >= cutoff_time]
                elif metric_type == "execution":
                    metrics = [m for m in self.execution_metrics if m["timestamp"] >= cutoff_time]
                elif metric_type == "system":
                    metrics = [m for m in self.system_metrics if m["timestamp"] >= cutoff_time]
                else:
                    # All metrics
                    all_metrics = (list(self.trade_metrics) + list(self.portfolio_metrics) + 
                                 list(self.risk_metrics) + list(self.execution_metrics) + 
                                 list(self.system_metrics))
                    metrics = [m for m in all_metrics if m["timestamp"] >= cutoff_time]
                
                summary = {
                    "metric_type": metric_type or "all",
                    "time_range_hours": time_range.total_seconds() / 3600,
                    "total_metrics": len(metrics),
                    "start_time": cutoff_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                }
                
                if metrics:
                    summary["first_metric"] = metrics[0]["timestamp"].isoformat()
                    summary["last_metric"] = metrics[-1]["timestamp"].isoformat()
                
                # Add current gauge values
                summary["current_gauges"] = dict(self.gauges)
                summary["counters"] = dict(self.counters)
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {"error": str(e)}
    
    def start_cleanup_thread(self):
        """Start background thread for metrics cleanup."""
        if self._cleanup_running:
            return
        
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self._cleanup_thread.start()
        logger.info("Metrics cleanup thread started")
    
    def stop_cleanup_thread(self):
        """Stop background cleanup thread."""
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        logger.info("Metrics cleanup thread stopped")
    
    def _cleanup_old_metrics(self):
        """Background thread to cleanup old metrics."""
        while self._cleanup_running:
            try:
                cutoff_time = datetime.now() - timedelta(days=self.retention_days)
                
                with self._lock:
                    # Clean up each metric collection
                    for metrics_collection in [self.trade_metrics, self.portfolio_metrics,
                                             self.risk_metrics, self.execution_metrics,
                                             self.system_metrics]:
                        # Remove old metrics
                        while metrics_collection and metrics_collection[0]["timestamp"] < cutoff_time:
                            metrics_collection.popleft()
                    
                    # Clean up histograms (keep only recent values)
                    for histogram in self.histograms.values():
                        if len(histogram) > 1000:
                            histogram[:] = histogram[-1000:]
                
                # Sleep for an hour before next cleanup
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
                time.sleep(300)  # Sleep 5 minutes on error


class PerformanceMonitor:
    """
    Advanced performance monitoring and analysis.
    
    Provides real-time performance analysis, benchmarking,
    and institutional-grade performance reporting.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance monitor."""
        # AI-AGENT-REF: Institutional performance monitoring
        self.metrics_collector = metrics_collector
        self.benchmark_data = {}
        self.performance_thresholds = PERFORMANCE_THRESHOLDS
        
        # Performance calculation cache
        self._performance_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 60  # Cache for 60 seconds
        
        logger.info("PerformanceMonitor initialized")
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from returns."""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
            
            if not excess_returns:
                return 0.0
            
            mean_excess = statistics.mean(excess_returns)
            
            if len(excess_returns) < 2:
                return 0.0
            
            std_excess = statistics.stdev(excess_returns)
            
            if std_excess == 0:
                return 0.0
            
            sharpe = mean_excess / std_excess * (252 ** 0.5)  # Annualized
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, values: List[float]) -> Tuple[float, Dict]:
        """Calculate maximum drawdown and related statistics."""
        try:
            if not values:
                return 0.0, {}
            
            peak = values[0]
            max_dd = 0.0
            dd_start_idx = 0
            dd_end_idx = 0
            current_dd_start = 0
            
            for i, value in enumerate(values):
                if value > peak:
                    peak = value
                    current_dd_start = i
                
                drawdown = (peak - value) / peak if peak > 0 else 0
                
                if drawdown > max_dd:
                    max_dd = drawdown
                    dd_start_idx = current_dd_start
                    dd_end_idx = i
            
            stats = {
                "max_drawdown": max_dd,
                "drawdown_start_index": dd_start_idx,
                "drawdown_end_index": dd_end_idx,
                "drawdown_duration": dd_end_idx - dd_start_idx,
                "peak_value": peak,
            }
            
            return max_dd, stats
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0, {}
    
    def get_performance_report(self, time_range: timedelta = None) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            time_range: Time range for analysis (default: last 30 days)
            
        Returns:
            Comprehensive performance report
        """
        try:
            # Check cache
            current_time = datetime.now()
            if (self._cache_timestamp and 
                (current_time - self._cache_timestamp).total_seconds() < self._cache_ttl):
                return self._performance_cache
            
            time_range = time_range or timedelta(days=30)
            cutoff_time = current_time - time_range
            
            # Get metrics
            trade_summary = self.metrics_collector.get_metrics_summary("trade", time_range)
            portfolio_summary = self.metrics_collector.get_metrics_summary("portfolio", time_range)
            risk_summary = self.metrics_collector.get_metrics_summary("risk", time_range)
            
            # Extract trade metrics
            trade_metrics = [m for m in self.metrics_collector.trade_metrics 
                           if m["timestamp"] >= cutoff_time]
            portfolio_metrics = [m for m in self.metrics_collector.portfolio_metrics 
                               if m["timestamp"] >= cutoff_time]
            
            report = {
                "report_date": current_time.isoformat(),
                "analysis_period": {
                    "start_date": cutoff_time.isoformat(),
                    "end_date": current_time.isoformat(),
                    "days": time_range.days,
                },
                "trading_activity": {},
                "portfolio_performance": {},
                "risk_metrics": {},
                "benchmark_comparison": {},
                "alerts": [],
            }
            
            # Trading activity analysis
            if trade_metrics:
                total_trades = len(trade_metrics)
                winning_trades = len([t for t in trade_metrics if t.get("pnl", 0) > 0])
                total_pnl = sum(t.get("pnl", 0) for t in trade_metrics)
                total_volume = sum(t.get("notional_value", 0) for t in trade_metrics)
                
                report["trading_activity"] = {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": total_trades - winning_trades,
                    "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
                    "total_pnl": total_pnl,
                    "total_volume": total_volume,
                    "average_trade_pnl": total_pnl / total_trades if total_trades > 0 else 0,
                    "average_trade_size": total_volume / total_trades if total_trades > 0 else 0,
                }
            
            # Portfolio performance analysis
            if portfolio_metrics:
                values = [p["total_value"] for p in portfolio_metrics]
                returns = []
                for i in range(1, len(values)):
                    if values[i-1] != 0:
                        returns.append((values[i] - values[i-1]) / values[i-1])
                
                max_dd, dd_stats = self.calculate_max_drawdown(values)
                sharpe = self.calculate_sharpe_ratio(returns) if returns else 0
                
                report["portfolio_performance"] = {
                    "starting_value": values[0] if values else 0,
                    "ending_value": values[-1] if values else 0,
                    "total_return": (values[-1] - values[0]) / values[0] if values and values[0] != 0 else 0,
                    "max_drawdown": max_dd,
                    "sharpe_ratio": sharpe,
                    "volatility": statistics.stdev(returns) if len(returns) > 1 else 0,
                    "best_day": max(returns) if returns else 0,
                    "worst_day": min(returns) if returns else 0,
                }
            
            # Performance alerts
            portfolio_perf = report["portfolio_performance"]
            if portfolio_perf.get("sharpe_ratio", 0) < self.performance_thresholds["MIN_SHARPE_RATIO"]:
                report["alerts"].append("Sharpe ratio below minimum threshold")
            
            if portfolio_perf.get("max_drawdown", 0) > self.performance_thresholds["MAX_DRAWDOWN"]:
                report["alerts"].append("Maximum drawdown exceeded")
            
            trading_activity = report["trading_activity"]
            if trading_activity.get("win_rate", 0) < self.performance_thresholds["MIN_WIN_RATE"]:
                report["alerts"].append("Win rate below minimum threshold")
            
            # Cache the report
            self._performance_cache = report
            self._cache_timestamp = current_time
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}