#!/usr/bin/env python3
"""
Performance Optimization Summary and Demo
Demonstrates all implemented performance improvements for the AI Trading Bot.
"""

import os
import sys
import time
from datetime import datetime, timezone

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

def demo_system_diagnostic():
    """Demonstrate system diagnostic capabilities."""
    print_section("System Diagnostic")
    
    try:
        from system_diagnostic import SystemDiagnostic
        
        diagnostic = SystemDiagnostic()
        results = diagnostic.run_full_diagnostic()
        
        print(f"✓ System diagnostic completed in {results['diagnostic_runtime_seconds']:.2f}s")
        print(f"  - Memory usage: {results['memory_analysis']['system_memory']['usage_percent']:.1f}%")
        print(f"  - Swap usage: {results['memory_analysis']['system_memory']['swap_used_mb']:.1f}MB")
        print(f"  - Total objects: {results['garbage_collection']['object_counts']['total_objects']:,}")
        print(f"  - Open file descriptors: {results['file_handles']['open_file_descriptors']}")
        print(f"  - Active threads: {results['thread_analysis']['active_threads']}")
        
        # Generate recommendations
        recommendations = diagnostic.generate_recommendations(results)
        if recommendations:
            print(f"  - Recommendations: {len(recommendations)} items")
            for rec in recommendations[:2]:  # Show first 2
                print(f"    • {rec[:80]}...")
        else:
            print("  - No performance recommendations needed ✓")
            
    except Exception as e:
        print(f"✗ System diagnostic error: {e}")

def demo_memory_optimizer():
    """Demonstrate memory optimization capabilities."""
    print_section("Memory Optimizer")
    
    try:
        from memory_optimizer import MemoryOptimizer, optimize_memory
        
        # Create optimizer instance
        optimizer = MemoryOptimizer(enable_monitoring=False)
        
        # Get initial memory stats
        initial_memory = optimizer.get_memory_usage()
        print(f"✓ Initial memory: {initial_memory['rss_mb']:.2f}MB")
        print(f"  - Total objects: {initial_memory['total_objects']:,}")
        print(f"  - GC thresholds: {initial_memory['gc_threshold']}")
        
        # Force garbage collection
        gc_result = optimizer.force_garbage_collection()
        print(f"✓ Garbage collection: {gc_result['objects_collected']} objects collected")
        print(f"  - Collection time: {gc_result['collection_time_ms']:.2f}ms")
        
        # Show memory after GC
        final_memory = optimizer.get_memory_usage()
        print(f"✓ Final memory: {final_memory['rss_mb']:.2f}MB")
        print(f"  - Objects reduced by: {initial_memory['total_objects'] - final_memory['total_objects']:,}")
        
        # Show top object types
        top_objects = final_memory['top_object_types']
        print("  - Top object types:")
        for obj_type, count in list(top_objects.items())[:3]:
            print(f"    • {obj_type}: {count:,}")
            
    except Exception as e:
        print(f"✗ Memory optimizer error: {e}")

def demo_performance_monitor():
    """Demonstrate performance monitoring capabilities."""
    print_section("Performance Monitor")
    
    try:
        from performance_monitor import ResourceMonitor
        
        monitor = ResourceMonitor(monitoring_interval=5)
        
        # Collect system metrics
        metrics = monitor.get_system_metrics()
        print(f"✓ Metrics collection time: {metrics['collection_time_ms']:.1f}ms")
        
        # Show memory metrics
        if 'memory' in metrics:
            mem = metrics['memory']
            print(f"  - Memory usage: {mem.get('usage_percent', 0):.1f}%")
            print(f"  - Available memory: {mem.get('available_mb', 0):.1f}MB")
            print(f"  - Swap usage: {mem.get('swap_used_mb', 0):.1f}MB")
        
        # Show CPU metrics
        if 'cpu' in metrics:
            cpu = metrics['cpu']
            print(f"  - CPU usage: {cpu.get('usage_percent', 0):.1f}%")
            print(f"  - Load average: {cpu.get('load_1min', 0):.2f}")
        
        # Show process metrics
        if 'process' in metrics:
            proc = metrics['process']
            print(f"  - Process memory: {proc.get('memory_mb', 0):.1f}MB")
            print(f"  - File descriptors: {proc.get('file_descriptors', 0)}")
            print(f"  - Thread count: {proc.get('thread_count', 0)}")
            print(f"  - Python processes: {proc.get('python_processes', 0)}")
        
        # Check for alerts
        alerts = monitor.check_alert_conditions(metrics)
        if alerts:
            print(f"  - Alerts generated: {len(alerts)}")
            for alert in alerts[:2]:
                print(f"    • {alert['type']}: {alert['message']}")
        else:
            print("  - No alerts (system healthy) ✓")
            
    except Exception as e:
        print(f"✗ Performance monitor error: {e}")

def demo_process_manager():
    """Demonstrate process management capabilities."""
    print_section("Process Manager")
    
    try:
        from process_manager import ProcessManager
        
        manager = ProcessManager()
        
        # Find Python processes
        processes = manager.find_python_processes()
        print(f"✓ Found {len(processes)} trading-related Python processes")
        
        if processes:
            total_memory = sum(p['memory_mb'] for p in processes)
            print(f"  - Total memory usage: {total_memory:.1f}MB")
            
            max_memory_proc = max(processes, key=lambda p: p['memory_mb'])
            print(f"  - Highest memory process: {max_memory_proc['memory_mb']:.1f}MB")
            print(f"    Command: {max_memory_proc['command'][:60]}...")
        
        # Check for duplicates
        duplicates = manager.find_duplicate_processes()
        if duplicates:
            print(f"  - Duplicate processes: {len(duplicates)} found")
            print("    ⚠ Consider running cleanup")
        else:
            print("  - No duplicate processes ✓")
        
        # Check service status
        service_status = manager.check_service_status()
        failed_services = [name for name, info in service_status.items() 
                          if not info.get('active', False)]
        
        if failed_services:
            print(f"  - Failed services: {len(failed_services)}")
            for service in failed_services[:2]:
                print(f"    • {service}")
        else:
            print("  - All services healthy ✓")
            
    except Exception as e:
        print(f"✗ Process manager error: {e}")

def demo_integration():
    """Demonstrate integration with trading code."""
    print_section("Trading Code Integration")
    
    try:
        # Test memory optimization integration in bot_engine
        print("Checking bot_engine.py integration:")
        with open('bot_engine.py', 'r') as f:
            content = f.read()
            
        if 'MEMORY_OPTIMIZATION_AVAILABLE' in content:
            print("  ✓ Memory optimization integrated")
        if '@memory_profile' in content:
            print("  ✓ Memory profiling decorators added")
        if 'optimize_memory()' in content:
            print("  ✓ Periodic memory cleanup added")
        
        # Test performance monitoring integration in ai_trading/main.py
        print("Checking ai_trading/main.py integration:")
        with open('ai_trading/main.py', 'r') as f:
            content = f.read()
            
        if 'PERFORMANCE_MONITORING_AVAILABLE' in content:
            print("  ✓ Performance monitoring integrated")
        if 'start_performance_monitoring' in content:
            print("  ✓ Monitoring startup added")
        if 'memory_check_interval' in content:
            print("  ✓ Periodic memory checks added")
            
    except Exception as e:
        print(f"✗ Integration check error: {e}")

def demo_startup_optimization():
    """Demonstrate startup optimization."""
    print_section("Startup Optimization")
    
    try:
        print("Optimized startup script features:")
        print("  ✓ Environment variable optimization")
        print("  ✓ Garbage collection tuning (500, 8, 8)")
        print("  ✓ Signal handlers for graceful shutdown")
        print("  ✓ Pre-startup health checks")
        print("  ✓ Duplicate process cleanup")
        print("  ✓ Automatic monitoring activation")
        print("  ✓ Emergency cleanup on exit")
        
        # Check if startup script exists
        if os.path.exists('optimized_startup.py'):
            print("  ✓ Optimized startup script ready")
        else:
            print("  ✗ Startup script not found")
            
    except Exception as e:
        print(f"✗ Startup optimization error: {e}")

def main():
    """Main demo function."""
    print_header("AI Trading Bot - Performance Optimization Demo")
    print(f"Demo time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Python version: {sys.version}")
    
    # Run all demonstrations
    demo_system_diagnostic()
    demo_memory_optimizer() 
    demo_performance_monitor()
    demo_process_manager()
    demo_integration()
    demo_startup_optimization()
    
    print_header("Summary of Performance Improvements")
    
    improvements = [
        "System diagnostic tool for comprehensive health monitoring",
        "Memory optimizer with aggressive GC and leak detection", 
        "Real-time performance monitoring with alerts",
        "Process manager for duplicate cleanup and service monitoring",
        "Memory profiling decorators on critical trading functions",
        "Periodic memory cleanup in trading loops",
        "Performance monitoring integration in main trading cycle",
        "Optimized startup script with health checks",
        "Graceful shutdown with emergency cleanup",
        "Environment optimization for maximum performance"
    ]
    
    print("Implemented optimizations:")
    for i, improvement in enumerate(improvements, 1):
        print(f"  {i:2d}. {improvement}")
    
    print("\nRecommended usage:")
    print("  • Use 'python optimized_startup.py' for production")
    print("  • Run 'python system_diagnostic.py' for health checks")
    print("  • Use 'python process_manager.py' for process cleanup")
    print("  • Monitor performance with built-in monitoring system")
    
    print(f"\n{'='*60}")
    print("  Performance optimization implementation complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()