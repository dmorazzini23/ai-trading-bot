#!/usr/bin/env python3
"""
Optimized Trading Bot Startup Script
Integrates memory optimization, performance monitoring, and process management.
"""

import os
import sys
import logging
import signal
import atexit
from datetime import datetime, timezone

# AI-AGENT-REF: Optimized startup script with performance monitoring

def setup_optimized_environment():
    """Setup environment for optimal performance."""
    
    # Set environment variables for memory optimization
    os.environ['PYTHONHASHSEED'] = '0'  # Consistent hash seed
    os.environ['PYTHONUNBUFFERED'] = '1'  # Unbuffered output
    
    # Memory optimization settings
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'  # 128KB
    os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'  # 128KB
    
    # Enable garbage collection optimization
    import gc
    gc.enable()
    gc.set_threshold(500, 8, 8)  # More aggressive GC
    
    print("Environment optimized for performance")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        
        # Stop performance monitoring
        try:
            from performance_monitor import stop_performance_monitoring
            stop_performance_monitoring()
        except ImportError:
            pass
        
        # Stop memory monitoring
        try:
            from memory_optimizer import get_memory_optimizer
            optimizer = get_memory_optimizer()
            if optimizer:
                optimizer.stop_memory_monitoring()
        except ImportError:
            pass
        
        # Emergency memory cleanup
        try:
            from memory_optimizer import emergency_memory_cleanup
            emergency_memory_cleanup()
        except ImportError:
            pass
        
        print("Graceful shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def check_system_health():
    """Perform system health check before startup."""
    print("Performing system health check...")
    
    try:
        from system_diagnostic import SystemDiagnostic
        diagnostic = SystemDiagnostic()
        results = diagnostic.run_full_diagnostic()
        
        # Check for critical issues
        recommendations = diagnostic.generate_recommendations(results)
        critical_issues = [r for r in recommendations if 'HIGH PRIORITY' in r or 'CRITICAL' in r]
        
        if critical_issues:
            print("CRITICAL ISSUES DETECTED:")
            for issue in critical_issues:
                print(f"  - {issue}")
            
            response = input("Continue startup despite critical issues? (y/N): ")
            if response.lower() != 'y':
                print("Startup aborted due to critical issues")
                return False
        
        print(f"Health check complete: {len(recommendations)} recommendations")
        return True
        
    except ImportError:
        print("System diagnostic not available, skipping health check")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def cleanup_duplicate_processes():
    """Clean up any duplicate processes."""
    print("Checking for duplicate processes...")
    
    try:
        from process_manager import ProcessManager
        manager = ProcessManager()
        
        duplicates = manager.find_duplicate_processes()
        if duplicates:
            print(f"Found {len(duplicates)} duplicate processes")
            cleanup_result = manager.cleanup_duplicate_processes(dry_run=False)
            print(f"Cleaned up {len(cleanup_result['processes_killed'])} processes")
        else:
            print("No duplicate processes found")
            
    except ImportError:
        print("Process manager not available, skipping duplicate cleanup")
    except Exception as e:
        print(f"Process cleanup failed: {e}")


def start_monitoring():
    """Start performance and memory monitoring."""
    print("Starting performance monitoring...")
    
    try:
        from performance_monitor import start_performance_monitoring
        from memory_optimizer import get_memory_optimizer
        
        # Start performance monitoring
        start_performance_monitoring()
        
        # Initialize memory optimizer
        optimizer = get_memory_optimizer()
        if optimizer:
            print("Memory optimization enabled")
        
        print("Monitoring systems active")
        return True
        
    except ImportError:
        print("Monitoring systems not available")
        return False
    except Exception as e:
        print(f"Failed to start monitoring: {e}")
        return False


def cleanup_on_exit():
    """Cleanup function called on exit."""
    print("Performing cleanup on exit...")
    
    try:
        from memory_optimizer import emergency_memory_cleanup
        result = emergency_memory_cleanup()
        print(f"Emergency cleanup: {result.get('rss_mb', 0):.1f}MB memory")
    except:
        pass


def main():
    """Main optimized startup function."""
    print("AI Trading Bot - Optimized Startup")
    print("=" * 40)
    print(f"Startup time: {datetime.now(timezone.utc).isoformat()}")
    
    # Register cleanup function
    atexit.register(cleanup_on_exit)
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Optimize environment
    setup_optimized_environment()
    
    # System health check
    if not check_system_health():
        return 1
    
    # Clean up duplicate processes
    cleanup_duplicate_processes()
    
    # Start monitoring
    monitoring_started = start_monitoring()
    
    # Import and start the main trading application
    try:
        print("Starting main trading application...")
        
        # Choose startup method based on available modules
        if os.path.exists('ai_trading/main.py'):
            # Use new modular approach
            from ai_trading.main import main as trading_main
            print("Using ai_trading.main module")
            trading_main()
        elif os.path.exists('bot_engine.py'):
            # Fallback to legacy bot_engine
            print("Using legacy bot_engine module")
            import bot_engine
            if hasattr(bot_engine, 'main'):
                bot_engine.main()
            else:
                print("No main function found in bot_engine")
                return 1
        else:
            print("No trading module found")
            return 1
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Trading application failed: {e}")
        logging.exception("Trading application error")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())