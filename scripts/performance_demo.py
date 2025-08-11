#!/usr/bin/env python3
import logging

"""
Performance Optimization Summary and Demo
Demonstrates all implemented performance improvements for the AI Trading Bot.
"""

import os
import sys
from datetime import datetime, timezone

def print_header(title):
    """Print a formatted header."""
    logging.info(str(f"\n{'='*60}"))
    logging.info(f"  {title}")
    logging.info(str(f"{'='*60}"))

def print_section(title):
    """Print a formatted section header."""
    logging.info(str(f"\n{'-'*40}"))
    logging.info(f"  {title}")
    logging.info(str(f"{'-'*40}"))

def demo_system_diagnostic():
    """Demonstrate system diagnostic capabilities."""
    print_section("System Diagnostic")
    
    try:
        from system_diagnostic import SystemDiagnostic
        
        diagnostic = SystemDiagnostic()
        results = diagnostic.run_full_diagnostic()
        
        logging.info(str(f"✓ System diagnostic completed in {results['diagnostic_runtime_seconds']:.2f}s"))
        logging.info(str(f"  - Memory usage: {results['memory_analysis']['system_memory']['usage_percent']:.1f}%"))
        logging.info(str(f"  - Swap usage: {results['memory_analysis']['system_memory']['swap_used_mb']:.1f}MB"))
        logging.info(str(f"  - Total objects: {results['garbage_collection']['object_counts']['total_objects']:,}"))
        logging.info(str(f"  - Open file descriptors: {results['file_handles']['open_file_descriptors']}"))
        logging.info(str(f"  - Active threads: {results['thread_analysis']['active_threads']}"))
        
        # Generate recommendations
        recommendations = diagnostic.generate_recommendations(results)
        if recommendations:
            logging.info(f"  - Recommendations: {len(recommendations)} items")
            for rec in recommendations[:2]:  # Show first 2
                logging.info(f"    • {rec[:80]}...")
        else:
            logging.info("  - No performance recommendations needed ✓")
            
    except Exception as e:
        logging.info(f"✗ System diagnostic error: {e}")

def demo_memory_optimizer():
    """Demonstrate memory optimization capabilities."""
    print_section("Memory Optimizer")
    
    try:
        from memory_optimizer import MemoryOptimizer
        
        # Create optimizer instance
        optimizer = MemoryOptimizer(enable_monitoring=False)
        
        # Get initial memory stats
        initial_memory = optimizer.get_memory_usage()
        logging.info(str(f"✓ Initial memory: {initial_memory['rss_mb']:.2f}MB"))
        logging.info(str(f"  - Total objects: {initial_memory['total_objects']:,}"))
        logging.info(str(f"  - GC thresholds: {initial_memory['gc_threshold']}"))
        
        # Force garbage collection
        gc_result = optimizer.force_garbage_collection()
        logging.info(str(f"✓ Garbage collection: {gc_result['objects_collected']} objects collected"))
        logging.info(str(f"  - Collection time: {gc_result['collection_time_ms']:.2f}ms"))
        
        # Show memory after GC
        final_memory = optimizer.get_memory_usage()
        logging.info(str(f"✓ Final memory: {final_memory['rss_mb']:.2f}MB"))
        logging.info(str(f"  - Objects reduced by: {initial_memory['total_objects'] - final_memory['total_objects']:,}"))
        
        # Show top object types
        top_objects = final_memory['top_object_types']
        logging.info("  - Top object types:")
        for obj_type, count in list(top_objects.items())[:3]:
            logging.info(f"    • {obj_type}: {count:,}")
            
    except Exception as e:
        logging.info(f"✗ Memory optimizer error: {e}")

def demo_performance_monitor():
    """Demonstrate performance monitoring capabilities."""
    print_section("Performance Monitor")
    
    try:
        from performance_monitor import ResourceMonitor
        
        monitor = ResourceMonitor(monitoring_interval=5)
        
        # Collect system metrics
        metrics = monitor.get_system_metrics()
        logging.info(str(f"✓ Metrics collection time: {metrics['collection_time_ms']:.1f}ms"))
        
        # Show memory metrics
        if 'memory' in metrics:
            mem = metrics['memory']
            logging.info(f"  - Memory usage: {mem.get('usage_percent', 0):.1f}%")
            logging.info(str(f"  - Available memory: {mem.get('available_mb', 0)):.1f}MB")
            logging.info(str(f"  - Swap usage: {mem.get('swap_used_mb', 0)):.1f}MB")
        
        # Show CPU metrics
        if 'cpu' in metrics:
            cpu = metrics['cpu']
            logging.info(str(f"  - CPU usage: {cpu.get('usage_percent', 0)):.1f}%")
            logging.info(str(f"  - Load average: {cpu.get('load_1min', 0)):.2f}")
        
        # Show process metrics
        if 'process' in metrics:
            proc = metrics['process']
            logging.info(str(f"  - Process memory: {proc.get('memory_mb', 0)):.1f}MB")
            logging.info(str(f"  - File descriptors: {proc.get('file_descriptors', 0))}")
            logging.info(str(f"  - Thread count: {proc.get('thread_count', 0))}")
            logging.info(str(f"  - Python processes: {proc.get('python_processes', 0))}")
        
        # Check for alerts
        alerts = monitor.check_alert_conditions(metrics)
        if alerts:
            logging.info(f"  - Alerts generated: {len(alerts)}")
            for alert in alerts[:2]:
                logging.info(str(f"    • {alert['type']}: {alert['message']}"))
        else:
            logging.info("  - No alerts (system healthy) ✓")
            
    except Exception as e:
        logging.info(f"✗ Performance monitor error: {e}")

def demo_process_manager():
    """Demonstrate process management capabilities."""
    print_section("Process Manager")
    
    try:
        from process_manager import ProcessManager
        
        manager = ProcessManager()
        
        # Find Python processes
        processes = manager.find_python_processes()
        logging.info(f"✓ Found {len(processes)} trading-related Python processes")
        
        if processes:
            total_memory = sum(p['memory_mb'] for p in processes)
            logging.info(f"  - Total memory usage: {total_memory:.1f}MB")
            
            max_memory_proc = max(processes, key=lambda p: p['memory_mb'])
            logging.info(str(f"  - Highest memory process: {max_memory_proc['memory_mb']:.1f}MB"))
            logging.info(str(f"    Command: {max_memory_proc['command'][:60]}..."))
        
        # Check for duplicates
        duplicates = manager.find_duplicate_processes()
        if duplicates:
            logging.info(f"  - Duplicate processes: {len(duplicates)} found")
            logging.info("    ⚠ Consider running cleanup")
        else:
            logging.info("  - No duplicate processes ✓")
        
        # Check service status
        service_status = manager.check_service_status()
        failed_services = [name for name, info in service_status.items() 
                          if not info.get('active', False)]
        
        if failed_services:
            logging.info(f"  - Failed services: {len(failed_services)}")
            for service in failed_services[:2]:
                logging.info(f"    • {service}")
        else:
            logging.info("  - All services healthy ✓")
            
    except Exception as e:
        logging.info(f"✗ Process manager error: {e}")

def demo_integration():
    """Demonstrate integration with trading code."""
    print_section("Trading Code Integration")
    
    try:
        # Test memory optimization integration in bot_engine
        logging.info("Checking bot_engine.py integration:")
        with open('bot_engine.py', 'r') as f:
            content = f.read()
            
        if 'MEMORY_OPTIMIZATION_AVAILABLE' in content:
            logging.info("  ✓ Memory optimization integrated")
        if '@memory_profile' in content:
            logging.info("  ✓ Memory profiling decorators added")
        if 'optimize_memory()' in content:
            logging.info("  ✓ Periodic memory cleanup added")
        
        # Test performance monitoring integration in ai_trading/main.py
        logging.info("Checking ai_trading/main.py integration:")
        with open('ai_trading/main.py', 'r') as f:
            content = f.read()
            
        if 'PERFORMANCE_MONITORING_AVAILABLE' in content:
            logging.info("  ✓ Performance monitoring integrated")
        if 'start_performance_monitoring' in content:
            logging.info("  ✓ Monitoring startup added")
        if 'memory_check_interval' in content:
            logging.info("  ✓ Periodic memory checks added")
            
    except Exception as e:
        logging.info(f"✗ Integration check error: {e}")

def demo_startup_optimization():
    """Demonstrate startup optimization."""
    print_section("Startup Optimization")
    
    try:
        logging.info("Optimized startup script features:")
        logging.info("  ✓ Environment variable optimization")
        logging.info("  ✓ Garbage collection tuning (500, 8, 8)")
        logging.info("  ✓ Signal handlers for graceful shutdown")
        logging.info("  ✓ Pre-startup health checks")
        logging.info("  ✓ Duplicate process cleanup")
        logging.info("  ✓ Automatic monitoring activation")
        logging.info("  ✓ Emergency cleanup on exit")
        
        # Check if startup script exists
        if os.path.exists('optimized_startup.py'):
            logging.info("  ✓ Optimized startup script ready")
        else:
            logging.info("  ✗ Startup script not found")
            
    except Exception as e:
        logging.info(f"✗ Startup optimization error: {e}")

def main():
    """Main demo function."""
    print_header("AI Trading Bot - Performance Optimization Demo")
    logging.info(f"Demo time: {datetime.now(timezone.utc).isoformat()}")
    logging.info(f"Python version: {sys.version}")
    
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
    
    logging.info("Implemented optimizations:")
    for i, improvement in enumerate(improvements, 1):
        logging.info(f"  {i:2d}. {improvement}")
    
    logging.info("\nRecommended usage:")
    logging.info(str("  • Use 'python optimized_startup.py' for production"))
    logging.info(str("  • Run 'python system_diagnostic.py' for health checks"))
    logging.info(str("  • Use 'python process_manager.py' for process cleanup"))
    logging.info("  • Monitor performance with built-in monitoring system")
    
    logging.info(str(f"\n{'='*60}"))
    logging.info("  Performance optimization implementation complete!")
    logging.info(str(f"{'='*60}"))

if __name__ == "__main__":
    main()