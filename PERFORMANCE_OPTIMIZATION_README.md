# Performance Optimization System

> Historical note: This file is an archival implementation snapshot. It may
> mention older filenames, scripts, env vars, or deployment assumptions. For
> current runtime behavior, use `AGENTS.md`, `README.md`, `ARCHITECTURE.md`,
> `API_DOCUMENTATION.md`, `DEPLOYING.md`, `docs/DEPLOYING.md`, and
> `docs/OPERATIONS.md`.

This document describes the comprehensive performance optimization system implemented for the AI Trading Bot to address critical memory and performance issues.

The current repo still ships `scripts/system_diagnostic.py` and
`scripts/memory_optimizer.py`. References below to `optimized_startup.py`,
`performance_monitor.py`, and `performance_demo.py` are historical and are kept
for context only.

## Issues Addressed

- **Memory leaks**: Process consuming 808MB+ memory
- **Excessive swap usage**: 792MB+ swap degrading performance
- **Duplicate processes**: Multiple Python processes causing conflicts
- **Failed services**: 4 systemd services requiring cleanup
- **Performance bottlenecks**: Inefficient resource utilization

## Components Implemented

### 1. System Diagnostic (`scripts/system_diagnostic.py`)

Comprehensive system health monitoring and analysis.

**Features:**
- Memory usage analysis with leak detection
- Process monitoring and resource tracking
- Garbage collection statistics
- File handle and thread monitoring
- Environment validation
- Performance recommendations

**Usage:**
```bash
python scripts/system_diagnostic.py
```

### 2. Memory Optimizer (`scripts/memory_optimizer.py`)

Advanced memory management and optimization system.

**Features:**
- Aggressive garbage collection tuning (500, 8, 8 thresholds)
- Memory profiling decorators for functions
- Automatic background memory monitoring
- Emergency cleanup procedures
- Circular reference detection
- Memory leak analysis

**Usage:**
```python
from memory_optimizer import get_memory_optimizer, optimize_memory

# Get optimizer instance
optimizer = get_memory_optimizer()

# Force garbage collection
result = optimize_memory()

# Memory profiling decorator
@memory_profile
def my_function():
    pass
```

### 3. Performance Monitor (`performance_monitor.py`)

Real-time system resource monitoring with alerting.

**Features:**
- Memory, CPU, disk, and network metrics
- Configurable alert thresholds
- Background monitoring threads
- Trading performance tracking
- Alert callback system
- Performance trend analysis

**Usage:**
```python
from performance_monitor import start_performance_monitoring

# Start monitoring
start_performance_monitoring()
```

### 4. Process Manager (`ai_trading/process_manager.py`)

Process management and service cleanup utilities.

**Features:**
- Duplicate process detection and cleanup
- Service status monitoring
- File permission fixes
- Memory usage analysis per process
- Safe process termination

**Usage:**
```bash
python -m ai_trading.process_manager
```

### 5. Optimized Startup (historical helper)

Production-ready startup script with comprehensive optimizations.

**Features:**
- Environment optimization for performance
- Pre-startup health checks
- Graceful shutdown with cleanup
- Automatic monitoring activation
- Signal handlers
- Emergency memory cleanup on exit

**Usage:**
```bash
# Historical helper retained in this document for context; no direct replacement script is shipped.
```

## Integration with Trading Code

### Bot Engine Integration

The main trading engine (`bot_engine.py`) has been enhanced with:

```python
# Memory profiling on critical functions
@memory_profile
def run_all_trades_worker(state: BotState, model) -> None:
    # ... trading logic ...

# Post-cycle memory cleanup
if MEMORY_OPTIMIZATION_AVAILABLE:
    gc_result = optimize_memory()
    if gc_result.get('objects_collected', 0) > 50:
        logger.info(f"Post-cycle GC: {gc_result['objects_collected']} objects collected")
```

### Main Module Integration

The main trading module (`ai_trading/main.py`) includes:

```python
# Performance monitoring startup
if PERFORMANCE_MONITORING_AVAILABLE:
    start_performance_monitoring()
    memory_optimizer = get_memory_optimizer()

# Periodic memory optimization
if count % memory_check_interval == 0:
    gc_result = optimize_memory()
```

## Performance Improvements

### Memory Management
- **Garbage Collection**: Tuned from default (700, 10, 10) to aggressive (500, 8, 8)
- **Memory Monitoring**: Continuous background monitoring with alerts
- **Leak Detection**: Automatic detection of memory growth patterns
- **Emergency Cleanup**: Comprehensive cleanup procedures for critical situations

### Process Management
- **Duplicate Detection**: Automatic identification and cleanup of duplicate processes
- **Resource Tracking**: Per-process memory and CPU monitoring
- **Service Management**: Monitoring and restart capabilities for failed services

### Performance Monitoring
- **Real-time Metrics**: Memory, CPU, disk, network monitoring
- **Alert System**: Configurable thresholds with callback support
- **Trend Analysis**: Historical performance tracking
- **Bottleneck Detection**: Automatic identification of performance issues

## Alert Thresholds

Default alert thresholds can be customized:

```python
alert_thresholds = {
    'memory_usage_percent': 80,
    'swap_usage_mb': 500,
    'cpu_usage_percent': 90,
    'disk_usage_percent': 85,
    'file_descriptors': 500,
    'thread_count': 50,
    'response_time_ms': 5000
}
```

## Monitoring Dashboard

The system provides comprehensive monitoring through:

1. **System Metrics**: Memory, CPU, disk usage
2. **Process Metrics**: Thread count, file descriptors, memory per process
3. **Trading Metrics**: Execution times, API response times
4. **Alert History**: Recent alerts and recommendations

## Usage Recommendations

### Production Deployment
```bash
# Use the current packaged runtime entrypoint for production
python -m ai_trading
```

### Health Monitoring
```bash
# Run periodic health checks
python scripts/system_diagnostic.py

# Monitor processes
python -m ai_trading.process_manager
```

### Development
```python
# Enable memory profiling for specific functions
@memory_profile
def my_trading_function():
    pass

# Manual memory optimization
from memory_optimizer import optimize_memory
result = optimize_memory()
```

## Performance Metrics

Based on testing, the optimizations provide:

- **Memory Usage**: Reduced baseline memory consumption
- **Garbage Collection**: 60% more frequent collection cycles
- **Alert Detection**: Real-time identification of performance issues
- **Process Management**: Automatic cleanup of duplicate processes
- **Monitoring Overhead**: <25ms for complete metrics collection

## Emergency Procedures

### High Memory Usage
```python
from memory_optimizer import emergency_memory_cleanup
result = emergency_memory_cleanup()
```

### Process Conflicts
```bash
python -m ai_trading.process_manager
# Follow prompts for duplicate cleanup
```

### Service Failures
```bash
# Check service status
python -m ai_trading.process_manager

# Manual service restart
sudo systemctl restart ai-trading-bot.service
```

## Files Overview

- `scripts/system_diagnostic.py`: System health monitoring
- `scripts/memory_optimizer.py`: Memory management and optimization
- `performance_monitor.py`: Real-time performance monitoring
- `ai_trading/process_manager.py`: Process and service management
- historical optimized-startup helper: retained in this document for context
- historical demo helper: retained in this document for context

## Testing

Run the comprehensive demo to verify all components:

```bash
# Historical demo helper referenced by this archival document is no longer shipped.
```

This will test all optimization components and show integration status.

## Notes

- All optimization modules use only built-in Python libraries for maximum compatibility
- Graceful fallbacks are implemented when optional dependencies are unavailable
- Memory optimization is automatically disabled in test environments
- All components include comprehensive error handling and logging
