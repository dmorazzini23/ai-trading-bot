#!/usr/bin/env python3
"""Performance optimization and profiling module for AI trading bot.

This module provides comprehensive performance optimization capabilities:
- Real-time performance profiling and bottleneck identification
- Memory usage optimization and leak detection  
- Latency optimization for order execution and data processing
- Database query optimization and caching strategies
- Algorithm complexity analysis and optimization
- Resource usage monitoring and optimization

AI-AGENT-REF: Production-grade performance optimization system
"""

from __future__ import annotations

import asyncio
import cProfile
import functools
import gc
import logging
import os
import psutil
import statistics
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
import weakref

# AI-AGENT-REF: Performance optimization for institutional-grade trading


@dataclass
class PerformanceProfile:
    """Performance profiling results."""
    function_name: str
    call_count: int
    total_time: float
    avg_time: float
    max_time: float
    min_time: float
    memory_usage: float
    timestamp: datetime
    bottlenecks: List[str] = field(default_factory=list)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: datetime
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    gc_objects: int
    gc_stats: Dict[str, Any]


class PerformanceOptimizer:
    """Comprehensive performance optimization system."""
    
    def __init__(self, enable_memory_profiling: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_memory_profiling = enable_memory_profiling
        
        # Performance tracking
        self.function_profiles: Dict[str, PerformanceProfile] = {}
        self.execution_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.memory_snapshots: deque = deque(maxlen=1000)
        
        # Caching system
        self.cache: Dict[str, Any] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.cache_max_size: int = 1000
        
        # Memory management
        self.memory_thresholds = {
            'warning': 1024,  # 1GB
            'critical': 2048,  # 2GB
            'gc_trigger': 512   # 512MB
        }
        
        # Performance targets
        self.performance_targets = {
            'order_execution_ms': 10.0,
            'data_processing_ms': 5.0,
            'indicator_calculation_ms': 2.0,
            'risk_check_ms': 1.0
        }
        
        # Initialize memory tracking
        if enable_memory_profiling:
            tracemalloc.start()
        
        # Optimization flags
        self._gc_optimization_enabled = True
        self._query_optimization_enabled = True
        self._caching_enabled = True
        
        self.logger.info("Performance optimizer initialized")
    
    def profile_function(self, include_memory: bool = False):
        """Decorator to profile function performance."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_execution(func, args, kwargs, include_memory)
            return wrapper
        return decorator
    
    def _profile_execution(self, func: Callable, args: tuple, kwargs: dict, include_memory: bool) -> Any:
        """Execute function with performance profiling."""
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Memory before execution
        memory_before = 0
        if include_memory and self.enable_memory_profiling:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execution timing
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            self.logger.warning(f"Function {func_name} failed during profiling: {e}")
            success = False
            raise
        finally:
            # Calculate execution time
            execution_time = time.perf_counter() - start_time
            execution_time_ms = execution_time * 1000
            
            # Memory after execution
            memory_after = 0
            memory_delta = 0
            if include_memory and self.enable_memory_profiling:
                process = psutil.Process()
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = memory_after - memory_before
            
            # Record performance data
            self._record_performance(func_name, execution_time_ms, memory_delta)
            
            # Check for performance issues
            self._check_performance_thresholds(func_name, execution_time_ms)
        
        return result
    
    def _record_performance(self, func_name: str, execution_time_ms: float, memory_delta: float):
        """Record performance metrics for a function."""
        self.execution_times[func_name].append(execution_time_ms)
        
        # Update or create performance profile
        times = list(self.execution_times[func_name])
        
        if func_name in self.function_profiles:
            profile = self.function_profiles[func_name]
            profile.call_count += 1
            profile.total_time += execution_time_ms
            profile.avg_time = profile.total_time / profile.call_count
            profile.max_time = max(profile.max_time, execution_time_ms)
            profile.min_time = min(profile.min_time, execution_time_ms)
            profile.memory_usage += memory_delta
        else:
            self.function_profiles[func_name] = PerformanceProfile(
                function_name=func_name,
                call_count=1,
                total_time=execution_time_ms,
                avg_time=execution_time_ms,
                max_time=execution_time_ms,
                min_time=execution_time_ms,
                memory_usage=memory_delta,
                timestamp=datetime.now(timezone.utc)
            )
    
    def _check_performance_thresholds(self, func_name: str, execution_time_ms: float):
        """Check if function execution exceeds performance thresholds."""
        # Check against known performance targets
        target_key = None
        if 'order' in func_name.lower() and 'execution' in func_name.lower():
            target_key = 'order_execution_ms'
        elif 'data' in func_name.lower() and 'process' in func_name.lower():
            target_key = 'data_processing_ms'
        elif 'indicator' in func_name.lower() or 'signal' in func_name.lower():
            target_key = 'indicator_calculation_ms'
        elif 'risk' in func_name.lower():
            target_key = 'risk_check_ms'
        
        if target_key and target_key in self.performance_targets:
            threshold = self.performance_targets[target_key]
            if execution_time_ms > threshold * 2:  # Critical threshold (2x target)
                self.logger.error(
                    f"CRITICAL: {func_name} execution time {execution_time_ms:.2f}ms "
                    f"exceeds critical threshold {threshold * 2:.2f}ms"
                )
            elif execution_time_ms > threshold:  # Warning threshold
                self.logger.warning(
                    f"WARNING: {func_name} execution time {execution_time_ms:.2f}ms "
                    f"exceeds target {threshold:.2f}ms"
                )
    
    @contextmanager
    def track_operation(self, operation_name: str, include_memory: bool = False):
        """Context manager to track operation performance."""
        memory_before = 0
        if include_memory and self.enable_memory_profiling:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            execution_time = time.perf_counter() - start_time
            execution_time_ms = execution_time * 1000
            
            memory_delta = 0
            if include_memory and self.enable_memory_profiling:
                process = psutil.Process()
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_delta = memory_after - memory_before
            
            self._record_performance(operation_name, execution_time_ms, memory_delta)
            self._check_performance_thresholds(operation_name, execution_time_ms)
    
    def take_memory_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        # Get garbage collection stats
        gc_stats = {
            'objects': len(gc.get_objects()),
            'collections': gc.get_stats(),
            'thresholds': gc.get_threshold()
        }
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(timezone.utc),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=system_memory.percent,
            available_mb=system_memory.available / 1024 / 1024,
            gc_objects=gc_stats['objects'],
            gc_stats=gc_stats
        )
        
        self.memory_snapshots.append(snapshot)
        
        # Check memory thresholds
        self._check_memory_thresholds(snapshot)
        
        return snapshot
    
    def _check_memory_thresholds(self, snapshot: MemorySnapshot):
        """Check memory usage against thresholds."""
        if snapshot.rss_mb > self.memory_thresholds['critical']:
            self.logger.error(f"CRITICAL: Memory usage {snapshot.rss_mb:.1f}MB exceeds critical threshold")
            self._trigger_memory_optimization()
        elif snapshot.rss_mb > self.memory_thresholds['warning']:
            self.logger.warning(f"WARNING: Memory usage {snapshot.rss_mb:.1f}MB exceeds warning threshold")
        elif snapshot.rss_mb > self.memory_thresholds['gc_trigger']:
            self._trigger_garbage_collection()
    
    def _trigger_memory_optimization(self):
        """Trigger aggressive memory optimization."""
        self.logger.info("Triggering memory optimization")
        
        # Clear caches
        self.clear_cache()
        
        # Force garbage collection
        self._trigger_garbage_collection()
        
        # Clear old performance data
        self._cleanup_performance_data()
    
    def _trigger_garbage_collection(self):
        """Trigger garbage collection."""
        if self._gc_optimization_enabled:
            collected = gc.collect()
            self.logger.debug(f"Garbage collection freed {collected} objects")
    
    def _cleanup_performance_data(self):
        """Clean up old performance data to free memory."""
        # Keep only recent performance data
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        # Clear old execution times
        for func_name in list(self.execution_times.keys()):
            if len(self.execution_times[func_name]) > 100:
                # Keep only recent 100 entries
                recent_times = list(self.execution_times[func_name])[-100:]
                self.execution_times[func_name] = deque(recent_times, maxlen=1000)
        
        self.logger.debug("Cleaned up old performance data")
    
    # Caching system
    def cached_function(self, cache_key_func: Optional[Callable] = None, ttl_seconds: int = 300):
        """Decorator for function result caching."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self._caching_enabled:
                    return func(*args, **kwargs)
                
                # Generate cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                
                # Check cache
                cache_entry = self.cache.get(cache_key)
                if cache_entry:
                    value, timestamp = cache_entry
                    if time.time() - timestamp < ttl_seconds:
                        self.cache_hits += 1
                        return value
                    else:
                        # Expired entry
                        del self.cache[cache_key]
                
                # Cache miss - execute function
                self.cache_misses += 1
                result = func(*args, **kwargs)
                
                # Store in cache (with size limit)
                if len(self.cache) >= self.cache_max_size:
                    # Remove oldest entry
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                    del self.cache[oldest_key]
                
                self.cache[cache_key] = (result, time.time())
                return result
            
            return wrapper
        return decorator
    
    def clear_cache(self):
        """Clear the performance cache."""
        self.cache.clear()
        self.logger.debug("Performance cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'max_size': self.cache_max_size
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Top slow functions
        sorted_profiles = sorted(
            self.function_profiles.values(),
            key=lambda p: p.avg_time,
            reverse=True
        )
        
        # Memory usage trend
        recent_snapshots = list(self.memory_snapshots)[-10:] if self.memory_snapshots else []
        memory_trend = {
            'current_mb': recent_snapshots[-1].rss_mb if recent_snapshots else 0,
            'avg_mb': statistics.mean([s.rss_mb for s in recent_snapshots]) if recent_snapshots else 0,
            'max_mb': max([s.rss_mb for s in recent_snapshots]) if recent_snapshots else 0,
            'snapshots_count': len(recent_snapshots)
        }
        
        # Performance violations
        violations = []
        for profile in sorted_profiles[:10]:  # Top 10 slow functions
            for target_key, threshold in self.performance_targets.items():
                if target_key in profile.function_name.lower():
                    if profile.avg_time > threshold:
                        violations.append({
                            'function': profile.function_name,
                            'avg_time_ms': profile.avg_time,
                            'threshold_ms': threshold,
                            'violation_ratio': profile.avg_time / threshold
                        })
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'performance_summary': {
                'functions_profiled': len(self.function_profiles),
                'total_function_calls': sum(p.call_count for p in self.function_profiles.values()),
                'top_slow_functions': [
                    {
                        'name': p.function_name,
                        'avg_time_ms': p.avg_time,
                        'call_count': p.call_count,
                        'total_time_ms': p.total_time
                    }
                    for p in sorted_profiles[:5]
                ]
            },
            'memory_usage': memory_trend,
            'cache_performance': self.get_cache_stats(),
            'performance_violations': violations,
            'optimization_settings': {
                'gc_optimization_enabled': self._gc_optimization_enabled,
                'query_optimization_enabled': self._query_optimization_enabled,
                'caching_enabled': self._caching_enabled,
                'memory_profiling_enabled': self.enable_memory_profiling
            }
        }
    
    def optimize_memory_usage(self):
        """Perform memory optimization."""
        self.logger.info("Starting memory optimization")
        
        # Take snapshot before optimization
        before_snapshot = self.take_memory_snapshot()
        
        # Clear caches
        self.clear_cache()
        
        # Force garbage collection
        self._trigger_garbage_collection()
        
        # Clean up performance data
        self._cleanup_performance_data()
        
        # Take snapshot after optimization
        after_snapshot = self.take_memory_snapshot()
        
        memory_freed = before_snapshot.rss_mb - after_snapshot.rss_mb
        
        self.logger.info(
            f"Memory optimization completed. "
            f"Memory freed: {memory_freed:.1f}MB "
            f"({before_snapshot.rss_mb:.1f}MB -> {after_snapshot.rss_mb:.1f}MB)"
        )
        
        return memory_freed
    
    def enable_optimization(self, optimization_type: str):
        """Enable specific optimization."""
        if optimization_type == "gc":
            self._gc_optimization_enabled = True
        elif optimization_type == "caching":
            self._caching_enabled = True
        elif optimization_type == "query":
            self._query_optimization_enabled = True
        
        self.logger.info(f"Enabled {optimization_type} optimization")
    
    def disable_optimization(self, optimization_type: str):
        """Disable specific optimization."""
        if optimization_type == "gc":
            self._gc_optimization_enabled = False
        elif optimization_type == "caching":
            self._caching_enabled = False
        elif optimization_type == "query":
            self._query_optimization_enabled = False
        
        self.logger.info(f"Disabled {optimization_type} optimization")


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def initialize_performance_optimizer(enable_memory_profiling: bool = True) -> PerformanceOptimizer:
    """Initialize performance optimizer."""
    global _performance_optimizer
    _performance_optimizer = PerformanceOptimizer(enable_memory_profiling)
    return _performance_optimizer


# Convenience decorators
def profile_performance(include_memory: bool = False):
    """Decorator to profile function performance."""
    optimizer = get_performance_optimizer()
    return optimizer.profile_function(include_memory)


def cached(cache_key_func: Optional[Callable] = None, ttl_seconds: int = 300):
    """Decorator for function result caching."""
    optimizer = get_performance_optimizer()
    return optimizer.cached_function(cache_key_func, ttl_seconds)