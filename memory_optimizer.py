#!/usr/bin/env python3
"""
Memory Optimization Module for AI Trading Bot
Implements garbage collection optimization, memory monitoring, and leak detection.
Uses only built-in Python modules for maximum compatibility.
"""

import gc
import os
import sys
import time
import threading
import weakref
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
import logging

# AI-AGENT-REF: Memory optimization and leak detection module

class MemoryOptimizer:
    """Memory optimization and monitoring system."""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.logger = self._setup_logger()
        self.memory_snapshots = []
        self.gc_stats = []
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self._weak_refs = {}
        
        # Configure garbage collection for optimal performance
        self.configure_garbage_collection()
        
        if enable_monitoring:
            self.start_memory_monitoring()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup memory optimizer logger."""
        logger = logging.getLogger('memory_optimizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def configure_garbage_collection(self):
        """Configure garbage collection for optimal performance."""
        # Enable garbage collection
        gc.enable()
        
        # Set more aggressive garbage collection thresholds
        # Default is (700, 10, 10), we use more frequent collection
        gc.set_threshold(500, 8, 8)
        
        # Enable debugging for garbage collection (in development)
        if os.getenv('TRADING_ENV') == 'development':
            gc.set_debug(gc.DEBUG_STATS)
        
        self.logger.info(f"Garbage collection configured with thresholds: {gc.get_threshold()}")
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics."""
        memory_info = {}
        
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_info['rss_mb'] = usage.ru_maxrss / 1024 / 1024  # Convert to MB
            memory_info['user_time'] = usage.ru_utime
            memory_info['system_time'] = usage.ru_stime
        except ImportError:
            memory_info['rss_mb'] = 0
        
        # Get garbage collection stats
        memory_info['gc_counts'] = gc.get_count()
        memory_info['gc_threshold'] = gc.get_threshold()
        
        # Count objects by type
        all_objects = gc.get_objects()
        memory_info['total_objects'] = len(all_objects)
        
        # Count specific object types that commonly cause leaks
        object_counts = {}
        for obj in all_objects:
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Report top object types
        memory_info['top_object_types'] = dict(
            sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        memory_info['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return memory_info
    
    def force_garbage_collection(self) -> Dict:
        """Force garbage collection and return statistics."""
        start_time = time.time()
        
        # Force collection for all generations
        collected = []
        for generation in range(3):
            n_collected = gc.collect(generation)
            collected.append(n_collected)
        
        gc_time = time.time() - start_time
        
        result = {
            'objects_collected': sum(collected),
            'by_generation': collected,
            'collection_time_ms': gc_time * 1000,
            'objects_after_gc': len(gc.get_objects()),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.gc_stats.append(result)
        self.logger.info(f"Garbage collection completed: {result['objects_collected']} objects collected in {result['collection_time_ms']:.2f}ms")
        
        return result
    
    def detect_memory_leaks(self) -> Dict:
        """Detect potential memory leaks by analyzing object growth."""
        if len(self.memory_snapshots) < 2:
            return {'status': 'insufficient_data', 'message': 'Need at least 2 snapshots'}
        
        current = self.memory_snapshots[-1]
        previous = self.memory_snapshots[-2]
        
        leak_indicators = {}
        
        # Check for significant object count increases
        for obj_type, count in current['top_object_types'].items():
            prev_count = previous['top_object_types'].get(obj_type, 0)
            if prev_count > 0:
                growth_rate = (count - prev_count) / prev_count
                if growth_rate > 0.5:  # 50% growth threshold
                    leak_indicators[obj_type] = {
                        'previous_count': prev_count,
                        'current_count': count,
                        'growth_rate': growth_rate
                    }
        
        # Check for memory growth
        rss_growth = current['rss_mb'] - previous['rss_mb']
        
        result = {
            'memory_growth_mb': rss_growth,
            'potential_leaks': leak_indicators,
            'total_objects_growth': current['total_objects'] - previous['total_objects'],
            'analysis_time': datetime.now(timezone.utc).isoformat()
        }
        
        if leak_indicators or rss_growth > 50:  # 50MB growth threshold
            self.logger.warning(f"Potential memory leak detected: {result}")
        
        return result
    
    def cleanup_circular_references(self):
        """Clean up circular references using weak references."""
        try:
            # This is a simplified approach - in practice, you'd identify
            # specific circular reference patterns in your codebase
            
            # Force garbage collection to clean up cycles
            collected = gc.collect()
            
            # Clean up any tracked weak references
            dead_refs = []
            for key, ref in self._weak_refs.items():
                if ref() is None:
                    dead_refs.append(key)
            
            for key in dead_refs:
                del self._weak_refs[key]
            
            self.logger.info(f"Cleaned up {collected} objects and {len(dead_refs)} dead references")
            
        except Exception as e:
            self.logger.error(f"Error during circular reference cleanup: {e}")
    
    def optimize_pandas_memory(self):
        """Optimize pandas memory usage if pandas is loaded."""
        try:
            if 'pandas' in sys.modules:
                pandas = sys.modules['pandas']
                
                # Set pandas options for memory optimization
                pandas.set_option('mode.chained_assignment', None)
                
                # Force garbage collection after pandas operations
                gc.collect()
                
                self.logger.info("Pandas memory optimization applied")
        except Exception as e:
            self.logger.warning(f"Could not optimize pandas memory: {e}")
    
    def memory_monitoring_loop(self):
        """Background thread for continuous memory monitoring."""
        while not self.stop_monitoring.is_set():
            try:
                # Take memory snapshot
                snapshot = self.get_memory_usage()
                self.memory_snapshots.append(snapshot)
                
                # Keep only last 100 snapshots to prevent memory buildup
                if len(self.memory_snapshots) > 100:
                    self.memory_snapshots = self.memory_snapshots[-100:]
                
                # Check for leaks every 10 snapshots
                if len(self.memory_snapshots) % 10 == 0:
                    leak_analysis = self.detect_memory_leaks()
                    
                    # Force GC if significant growth detected
                    if leak_analysis.get('memory_growth_mb', 0) > 100:
                        self.force_garbage_collection()
                        self.cleanup_circular_references()
                
                # Wait for next monitoring cycle
                self.stop_monitoring.wait(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                self.stop_monitoring.wait(60)  # Wait longer on error
    
    def start_memory_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(
                target=self.memory_monitoring_loop,
                name="MemoryMonitor",
                daemon=True
            )
            self.monitoring_thread.start()
            self.logger.info("Memory monitoring started")
    
    def stop_memory_monitoring(self):
        """Stop background memory monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
            self.logger.info("Memory monitoring stopped")
    
    def get_memory_report(self) -> Dict:
        """Generate comprehensive memory report."""
        current_memory = self.get_memory_usage()
        
        report = {
            'current_memory': current_memory,
            'gc_stats_history': self.gc_stats[-10:],  # Last 10 GC cycles
            'monitoring_snapshots': len(self.memory_snapshots),
            'thread_status': {
                'monitoring_active': self.monitoring_thread and self.monitoring_thread.is_alive(),
                'stop_event_set': self.stop_monitoring.is_set()
            }
        }
        
        # Add leak detection if we have enough data
        if len(self.memory_snapshots) >= 2:
            report['leak_analysis'] = self.detect_memory_leaks()
        
        return report
    
    def emergency_cleanup(self):
        """Emergency memory cleanup procedure."""
        self.logger.warning("Performing emergency memory cleanup")
        
        # Force aggressive garbage collection
        for generation in range(5):  # Multiple passes
            collected = gc.collect()
            self.logger.info(f"Emergency GC pass {generation + 1}: {collected} objects collected")
        
        # Clean up circular references
        self.cleanup_circular_references()
        
        # Optimize pandas if available
        self.optimize_pandas_memory()
        
        # Clear old snapshots to free memory
        if len(self.memory_snapshots) > 10:
            self.memory_snapshots = self.memory_snapshots[-10:]
        
        # Clear old GC stats
        if len(self.gc_stats) > 10:
            self.gc_stats = self.gc_stats[-10:]
        
        final_memory = self.get_memory_usage()
        self.logger.warning(f"Emergency cleanup completed. Current memory: {final_memory['rss_mb']:.2f}MB")
        
        return final_memory


class MemoryProfiler:
    """Simple memory profiler for function-level memory tracking."""
    
    def __init__(self):
        self.profiles = {}
        self.logger = logging.getLogger('memory_profiler')
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile memory usage of a function."""
        def wrapper(*args, **kwargs):
            # Get memory before function execution
            try:
                import resource
                mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
            except ImportError:
                mem_before = 0
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Get memory after function execution
                try:
                    import resource
                    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
                except ImportError:
                    mem_after = 0
                
                execution_time = time.time() - start_time
                memory_delta = mem_after - mem_before
                
                # Store profile data
                func_name = f"{func.__module__}.{func.__name__}"
                if func_name not in self.profiles:
                    self.profiles[func_name] = []
                
                self.profiles[func_name].append({
                    'execution_time_ms': execution_time * 1000,
                    'memory_delta_mb': memory_delta,
                    'memory_before_mb': mem_before,
                    'memory_after_mb': mem_after,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                # Log if significant memory increase
                if memory_delta > 10:  # 10MB threshold
                    self.logger.warning(f"Function {func_name} used {memory_delta:.2f}MB of memory")
            
            return result
        
        return wrapper
    
    def get_profile_report(self) -> Dict:
        """Get profiling report for all tracked functions."""
        report = {}
        
        for func_name, profiles in self.profiles.items():
            if profiles:
                total_memory = sum(p['memory_delta_mb'] for p in profiles)
                avg_memory = total_memory / len(profiles)
                max_memory = max(p['memory_delta_mb'] for p in profiles)
                
                report[func_name] = {
                    'call_count': len(profiles),
                    'total_memory_mb': total_memory,
                    'average_memory_mb': avg_memory,
                    'max_memory_mb': max_memory,
                    'last_called': profiles[-1]['timestamp']
                }
        
        return report


# Global memory optimizer instance
_memory_optimizer = None

def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer

def optimize_memory():
    """Convenience function for memory optimization."""
    optimizer = get_memory_optimizer()
    return optimizer.force_garbage_collection()

def emergency_memory_cleanup():
    """Convenience function for emergency memory cleanup."""
    optimizer = get_memory_optimizer()
    return optimizer.emergency_cleanup()

def memory_profile(func: Callable) -> Callable:
    """Decorator for memory profiling functions."""
    profiler = MemoryProfiler()
    return profiler.profile_function(func)


if __name__ == "__main__":
    # Test the memory optimizer
    optimizer = MemoryOptimizer(enable_monitoring=False)
    
    print("Memory Optimizer Test")
    print("=" * 30)
    
    # Get initial memory stats
    initial_memory = optimizer.get_memory_usage()
    print(f"Initial memory: {initial_memory['rss_mb']:.2f}MB")
    print(f"Initial objects: {initial_memory['total_objects']}")
    
    # Force garbage collection
    gc_result = optimizer.force_garbage_collection()
    print(f"GC collected: {gc_result['objects_collected']} objects")
    
    # Get final memory stats
    final_memory = optimizer.get_memory_usage()
    print(f"Final memory: {final_memory['rss_mb']:.2f}MB")
    print(f"Final objects: {final_memory['total_objects']}")
    
    # Generate report
    report = optimizer.get_memory_report()
    print(f"Memory report generated with {len(report)} sections")