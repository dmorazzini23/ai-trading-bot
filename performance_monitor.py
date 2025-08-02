#!/usr/bin/env python3
"""
Performance Monitoring System for AI Trading Bot
Real-time monitoring of system resources, trading performance, and bottleneck detection.
Uses only built-in Python modules for maximum compatibility.
"""

import os
import sys
import time
import threading
import subprocess
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any
import logging
from collections import deque

# AI-AGENT-REF: Performance monitoring and alerting system

class ResourceMonitor:
    """Monitor system resources and performance metrics."""
    
    def __init__(self, monitoring_interval: int = 30):
        self.monitoring_interval = monitoring_interval
        self.logger = self._setup_logger()
        self.metrics_history = deque(maxlen=1000)  # Store last 1000 measurements
        self.alerts_history = deque(maxlen=100)    # Store last 100 alerts
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.alert_thresholds = self._default_thresholds()
        self.alert_callbacks = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup performance monitor logger."""
        logger = logging.getLogger('performance_monitor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _default_thresholds(self) -> Dict:
        """Default alert thresholds."""
        return {
            'memory_usage_percent': 80,
            'swap_usage_mb': 500,
            'cpu_usage_percent': 90,
            'disk_usage_percent': 85,
            'file_descriptors': 500,
            'thread_count': 50,
            'response_time_ms': 5000
        }
    
    def get_system_metrics(self) -> Dict:
        """Collect comprehensive system metrics."""
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'collection_time_ms': 0
        }
        
        start_time = time.time()
        
        try:
            # Memory metrics
            metrics['memory'] = self._get_memory_metrics()
            
            # CPU metrics  
            metrics['cpu'] = self._get_cpu_metrics()
            
            # Disk metrics
            metrics['disk'] = self._get_disk_metrics()
            
            # Process metrics
            metrics['process'] = self._get_process_metrics()
            
            # Network metrics (basic)
            metrics['network'] = self._get_network_metrics()
            
            # Python-specific metrics
            metrics['python'] = self._get_python_metrics()
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            metrics['error'] = str(e)
        
        metrics['collection_time_ms'] = (time.time() - start_time) * 1000
        
        return metrics
    
    def _get_memory_metrics(self) -> Dict:
        """Get memory-related metrics."""
        memory_metrics = {}
        
        try:
            # Read /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    key, value = line.split(':', 1)
                    meminfo[key.strip()] = int(value.strip().split()[0])
                
                total_mb = meminfo['MemTotal'] / 1024
                available_mb = meminfo['MemAvailable'] / 1024
                free_mb = meminfo['MemFree'] / 1024
                cached_mb = meminfo['Cached'] / 1024
                buffers_mb = meminfo['Buffers'] / 1024
                
                swap_total_mb = meminfo['SwapTotal'] / 1024
                swap_free_mb = meminfo['SwapFree'] / 1024
                swap_used_mb = swap_total_mb - swap_free_mb
                
                memory_metrics = {
                    'total_mb': total_mb,
                    'available_mb': available_mb,
                    'free_mb': free_mb,
                    'used_mb': total_mb - available_mb,
                    'usage_percent': ((total_mb - available_mb) / total_mb) * 100,
                    'cached_mb': cached_mb,
                    'buffers_mb': buffers_mb,
                    'swap_total_mb': swap_total_mb,
                    'swap_used_mb': swap_used_mb,
                    'swap_usage_percent': (swap_used_mb / max(swap_total_mb, 1)) * 100
                }
                
        except Exception as e:
            memory_metrics['error'] = str(e)
        
        return memory_metrics
    
    def _get_cpu_metrics(self) -> Dict:
        """Get CPU-related metrics."""
        cpu_metrics = {}
        
        try:
            # Read /proc/loadavg
            with open('/proc/loadavg', 'r') as f:
                loadavg = f.read().strip().split()
                cpu_metrics['load_1min'] = float(loadavg[0])
                cpu_metrics['load_5min'] = float(loadavg[1])
                cpu_metrics['load_15min'] = float(loadavg[2])
            
            # Get CPU count
            cpu_metrics['cpu_count'] = os.cpu_count() or 1
            
            # Calculate CPU usage percentage (simplified)
            cpu_metrics['usage_percent'] = min(cpu_metrics['load_1min'] / cpu_metrics['cpu_count'] * 100, 100)
            
        except Exception as e:
            cpu_metrics['error'] = str(e)
        
        return cpu_metrics
    
    def _get_disk_metrics(self) -> Dict:
        """Get disk usage metrics."""
        disk_metrics = {}
        
        try:
            import shutil
            
            # Get disk usage for current directory
            total, used, free = shutil.disk_usage('.')
            
            disk_metrics = {
                'total_gb': total / (1024**3),
                'used_gb': used / (1024**3),
                'free_gb': free / (1024**3),
                'usage_percent': (used / total) * 100
            }
            
            # Check for large files that might indicate issues
            large_files = []
            for root, dirs, files in os.walk('.'):
                # Skip version control directories
                dirs[:] = [d for d in dirs if not d.startswith('.git')]
                
                for file in files[:20]:  # Limit to prevent long scans
                    try:
                        filepath = os.path.join(root, file)
                        size = os.path.getsize(filepath)
                        if size > 50 * 1024 * 1024:  # Files > 50MB
                            large_files.append({
                                'path': filepath,
                                'size_mb': size / (1024**2)
                            })
                    except OSError:
                        continue
            
            disk_metrics['large_files_count'] = len(large_files)
            
        except Exception as e:
            disk_metrics['error'] = str(e)
        
        return disk_metrics
    
    def _get_process_metrics(self) -> Dict:
        """Get process-related metrics."""
        process_metrics = {}
        
        try:
            # Get current process info
            pid = os.getpid()
            
            # Process memory info
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                process_metrics['memory_mb'] = usage.ru_maxrss / 1024 / 1024
                process_metrics['user_time'] = usage.ru_utime
                process_metrics['system_time'] = usage.ru_stime
                process_metrics['voluntary_ctx_switches'] = usage.ru_nvcsw
                process_metrics['involuntary_ctx_switches'] = usage.ru_nivcsw
            except ImportError:
                process_metrics['memory_mb'] = 0
            
            # File descriptor count
            proc_fd = f'/proc/{pid}/fd'
            if os.path.exists(proc_fd):
                process_metrics['file_descriptors'] = len(os.listdir(proc_fd))
            
            # Thread count
            import threading
            process_metrics['thread_count'] = threading.active_count()
            
            # Python processes count
            try:
                result = subprocess.run(['pgrep', '-f', 'python'], 
                                      capture_output=True, text=True)
                python_pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
                process_metrics['python_processes'] = len([p for p in python_pids if p])
            except subprocess.SubprocessError:
                process_metrics['python_processes'] = 1
            
        except Exception as e:
            process_metrics['error'] = str(e)
        
        return process_metrics
    
    def _get_network_metrics(self) -> Dict:
        """Get basic network metrics."""
        network_metrics = {}
        
        try:
            # Check for established connections
            result = subprocess.run(['netstat', '-tn'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                established = sum(1 for line in lines if 'ESTABLISHED' in line)
                network_metrics['established_connections'] = established
            
        except subprocess.SubprocessError:
            network_metrics['established_connections'] = 0
        except Exception as e:
            network_metrics['error'] = str(e)
        
        return network_metrics
    
    def _get_python_metrics(self) -> Dict:
        """Get Python-specific metrics."""
        python_metrics = {}
        
        try:
            import gc
            
            python_metrics['loaded_modules'] = len(sys.modules)
            python_metrics['gc_counts'] = gc.get_count()
            python_metrics['gc_thresholds'] = gc.get_threshold()
            python_metrics['total_objects'] = len(gc.get_objects())
            
            # Check for AI trading specific modules
            trading_modules = [name for name in sys.modules.keys() 
                             if any(keyword in name.lower() for keyword in 
                                  ['trading', 'alpaca', 'ai_trading'])]
            python_metrics['trading_modules_loaded'] = len(trading_modules)
            
        except Exception as e:
            python_metrics['error'] = str(e)
        
        return python_metrics
    
    def check_alert_conditions(self, metrics: Dict) -> List[Dict]:
        """Check metrics against alert thresholds."""
        alerts = []
        
        # Memory alerts
        if 'memory' in metrics and isinstance(metrics['memory'], dict):
            mem = metrics['memory']
            
            if 'usage_percent' in mem and mem['usage_percent'] > self.alert_thresholds['memory_usage_percent']:
                alerts.append({
                    'type': 'memory_high',
                    'severity': 'high',
                    'message': f"Memory usage at {mem['usage_percent']:.1f}%",
                    'value': mem['usage_percent'],
                    'threshold': self.alert_thresholds['memory_usage_percent']
                })
            
            if 'swap_used_mb' in mem and mem['swap_used_mb'] > self.alert_thresholds['swap_usage_mb']:
                alerts.append({
                    'type': 'swap_high',
                    'severity': 'critical',
                    'message': f"Swap usage at {mem['swap_used_mb']:.1f}MB",
                    'value': mem['swap_used_mb'],
                    'threshold': self.alert_thresholds['swap_usage_mb']
                })
        
        # CPU alerts
        if 'cpu' in metrics and isinstance(metrics['cpu'], dict):
            cpu = metrics['cpu']
            
            if 'usage_percent' in cpu and cpu['usage_percent'] > self.alert_thresholds['cpu_usage_percent']:
                alerts.append({
                    'type': 'cpu_high',
                    'severity': 'medium',
                    'message': f"CPU usage at {cpu['usage_percent']:.1f}%",
                    'value': cpu['usage_percent'],
                    'threshold': self.alert_thresholds['cpu_usage_percent']
                })
        
        # Process alerts
        if 'process' in metrics and isinstance(metrics['process'], dict):
            proc = metrics['process']
            
            if 'file_descriptors' in proc and proc['file_descriptors'] > self.alert_thresholds['file_descriptors']:
                alerts.append({
                    'type': 'file_descriptors_high',
                    'severity': 'medium',
                    'message': f"File descriptors: {proc['file_descriptors']}",
                    'value': proc['file_descriptors'],
                    'threshold': self.alert_thresholds['file_descriptors']
                })
            
            if 'thread_count' in proc and proc['thread_count'] > self.alert_thresholds['thread_count']:
                alerts.append({
                    'type': 'thread_count_high',
                    'severity': 'low',
                    'message': f"Thread count: {proc['thread_count']}",
                    'value': proc['thread_count'],
                    'threshold': self.alert_thresholds['thread_count']
                })
            
            if 'python_processes' in proc and proc['python_processes'] > 1:
                alerts.append({
                    'type': 'multiple_python_processes',
                    'severity': 'medium',
                    'message': f"Multiple Python processes: {proc['python_processes']}",
                    'value': proc['python_processes'],
                    'threshold': 1
                })
        
        # Add timestamp to alerts
        for alert in alerts:
            alert['timestamp'] = metrics['timestamp']
        
        return alerts
    
    def monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Collect metrics
                metrics = self.get_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                alerts = self.check_alert_conditions(metrics)
                
                if alerts:
                    for alert in alerts:
                        self.alerts_history.append(alert)
                        self.logger.warning(f"ALERT: {alert['message']}")
                        
                        # Call alert callbacks
                        for callback in self.alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                self.logger.error(f"Error in alert callback: {e}")
                
                # Log summary periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 cycles
                    self._log_summary(metrics)
                
                # Wait for next cycle
                self.stop_monitoring.wait(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.stop_monitoring.wait(60)  # Wait longer on error
    
    def _log_summary(self, metrics: Dict):
        """Log periodic summary of system state."""
        summary_parts = []
        
        if 'memory' in metrics and 'usage_percent' in metrics['memory']:
            summary_parts.append(f"MEM: {metrics['memory']['usage_percent']:.1f}%")
        
        if 'cpu' in metrics and 'usage_percent' in metrics['cpu']:
            summary_parts.append(f"CPU: {metrics['cpu']['usage_percent']:.1f}%")
        
        if 'process' in metrics and 'thread_count' in metrics['process']:
            summary_parts.append(f"Threads: {metrics['process']['thread_count']}")
        
        if summary_parts:
            self.logger.info(f"System Summary: {' | '.join(summary_parts)}")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(
                target=self.monitoring_loop,
                name="ResourceMonitor",
                daemon=True
            )
            self.monitoring_thread.start()
            self.logger.info("Resource monitoring started")
    
    def stop_monitoring_gracefully(self):
        """Stop resource monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=10)
            self.logger.info("Resource monitoring stopped")
    
    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {'status': 'no_data', 'message': 'No metrics collected yet'}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        current_metrics = recent_metrics[-1]
        
        # Calculate trends
        trends = {}
        if len(recent_metrics) >= 2:
            first = recent_metrics[0]
            last = recent_metrics[-1]
            
            if 'memory' in first and 'memory' in last:
                if 'usage_percent' in first['memory'] and 'usage_percent' in last['memory']:
                    trends['memory_trend'] = last['memory']['usage_percent'] - first['memory']['usage_percent']
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'current_metrics': current_metrics,
            'trends': trends,
            'recent_alerts': list(self.alerts_history)[-20:],  # Last 20 alerts
            'monitoring_status': {
                'active': self.monitoring_thread and self.monitoring_thread.is_alive(),
                'metrics_collected': len(self.metrics_history),
                'alerts_generated': len(self.alerts_history)
            },
            'alert_thresholds': self.alert_thresholds
        }
        
        return report


class TradingPerformanceMonitor:
    """Monitor trading-specific performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger('trading_performance')
        self.trade_metrics = deque(maxlen=1000)
        self.api_response_times = deque(maxlen=100)
        
    def record_trade_execution_time(self, execution_time_ms: float, trade_type: str):
        """Record trade execution time."""
        self.trade_metrics.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'execution_time_ms': execution_time_ms,
            'trade_type': trade_type
        })
    
    def record_api_response_time(self, endpoint: str, response_time_ms: float):
        """Record API response time."""
        self.api_response_times.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'endpoint': endpoint,
            'response_time_ms': response_time_ms
        })
    
    def get_trading_performance_report(self) -> Dict:
        """Get trading performance report."""
        report = {
            'trade_execution': {
                'total_trades': len(self.trade_metrics),
                'average_execution_time_ms': 0,
                'max_execution_time_ms': 0
            },
            'api_performance': {
                'total_calls': len(self.api_response_times),
                'average_response_time_ms': 0,
                'max_response_time_ms': 0
            }
        }
        
        if self.trade_metrics:
            execution_times = [m['execution_time_ms'] for m in self.trade_metrics]
            report['trade_execution']['average_execution_time_ms'] = sum(execution_times) / len(execution_times)
            report['trade_execution']['max_execution_time_ms'] = max(execution_times)
        
        if self.api_response_times:
            response_times = [m['response_time_ms'] for m in self.api_response_times]
            report['api_performance']['average_response_time_ms'] = sum(response_times) / len(response_times)
            report['api_performance']['max_response_time_ms'] = max(response_times)
        
        return report


# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> ResourceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = ResourceMonitor()
    return _performance_monitor

def start_performance_monitoring():
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()

def stop_performance_monitoring():
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring_gracefully()


if __name__ == "__main__":
    # Test the performance monitor
    monitor = ResourceMonitor(monitoring_interval=5)
    
    print("Performance Monitor Test")
    print("=" * 30)
    
    # Get system metrics
    metrics = monitor.get_system_metrics()
    print(f"Collected metrics in {metrics['collection_time_ms']:.2f}ms")
    
    # Check for alerts
    alerts = monitor.check_alert_conditions(metrics)
    print(f"Generated {len(alerts)} alerts")
    
    # Print summary
    if 'memory' in metrics:
        print(f"Memory usage: {metrics['memory'].get('usage_percent', 0):.1f}%")
    if 'cpu' in metrics:
        print(f"CPU usage: {metrics['cpu'].get('usage_percent', 0):.1f}%")
    
    # Generate report
    report = monitor.get_performance_report()
    print(f"Performance report generated with status: {report.get('status', 'ok')}")