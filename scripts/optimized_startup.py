#!/usr/bin/env python3
"""
Optimized Trading Bot Startup Script
Integrates memory optimization, performance monitoring, and process management.
"""

import atexit
import logging
import os
import signal
import sys
from datetime import UTC, datetime

from ai_trading.core import bot_engine  # AI-AGENT-REF: use packaged bot_engine

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

    logging.info("Environment optimized for performance")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        logging.info(f"\nReceived signal {signum}, shutting down gracefully...")

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

        logging.info("Graceful shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def check_system_health():
    """Perform system health check before startup."""
    logging.info("Performing system health check...")

    try:
        from system_diagnostic import SystemDiagnostic
        diagnostic = SystemDiagnostic()
        results = diagnostic.run_full_diagnostic()

        # Check for critical issues
        recommendations = diagnostic.generate_recommendations(results)
        critical_issues = [r for r in recommendations if 'HIGH PRIORITY' in r or 'CRITICAL' in r]

        if critical_issues:
            logging.info("CRITICAL ISSUES DETECTED:")
            for issue in critical_issues:
                logging.info(f"  - {issue}")

            response = input("Continue startup despite critical issues? (y/N): ")
            if response.lower() != 'y':
                logging.info("Startup aborted due to critical issues")
                return False

        logging.info(f"Health check complete: {len(recommendations)} recommendations")
        return True

    except ImportError:
        logging.info("System diagnostic not available, skipping health check")
        return True
    except (ValueError, TypeError) as e:
        logging.info(f"Health check failed: {e}")
        return False


def cleanup_duplicate_processes():
    """Clean up any duplicate processes."""
    logging.info("Checking for duplicate processes...")

    try:
        from process_manager import ProcessManager
        manager = ProcessManager()

        duplicates = manager.find_duplicate_processes()
        if duplicates:
            logging.info(f"Found {len(duplicates)} duplicate processes")
            cleanup_result = manager.cleanup_duplicate_processes(dry_run=False)
            logging.info(f"Cleaned up {len(cleanup_result['processes_killed'])} processes")
        else:
            logging.info("No duplicate processes found")

    except ImportError:
        logging.info("Process manager not available, skipping duplicate cleanup")
    except (ValueError, TypeError) as e:
        logging.info(f"Process cleanup failed: {e}")


def start_monitoring():
    """Start performance and memory monitoring."""
    logging.info("Starting performance monitoring...")

    try:
        from memory_optimizer import get_memory_optimizer
        from performance_monitor import start_performance_monitoring

        # Start performance monitoring
        start_performance_monitoring()

        # Initialize memory optimizer
        optimizer = get_memory_optimizer()
        if optimizer:
            logging.info("Memory optimization enabled")

        logging.info("Monitoring systems active")
        return True

    except ImportError:
        logging.info("Monitoring systems not available")
        return False
    except (ValueError, TypeError) as e:
        logging.info(f"Failed to start monitoring: {e}")
        return False


def cleanup_on_exit():
    """Cleanup function called on exit."""
    logging.info("Performing cleanup on exit...")

    try:
        from memory_optimizer import emergency_memory_cleanup
        result = emergency_memory_cleanup()
        logging.info(f"Emergency cleanup: {result.get('rss_mb', 0):.1f}MB memory")
    except ImportError:
        logging.debug("Memory optimizer not available, skipping emergency cleanup")
    except (ValueError, TypeError) as e:
        logging.warning(f"Emergency memory cleanup failed: {e}")


def main():
    """Main optimized startup function."""
    logging.info("AI Trading Bot - Optimized Startup")
    logging.info(str("=" * 40))
    logging.info(f"Startup time: {datetime.now(UTC).isoformat()}")

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
    start_monitoring()

    # Import and start the main trading application
    try:
        logging.info("Starting main trading application...")

        # Choose startup method based on available modules
        if os.path.exists('ai_trading/main.py'):
            from ai_trading.main import main as trading_main
            logging.info("Using ai_trading.main module")
            trading_main()
        else:
            logging.info("Using ai_trading.core.bot_engine module")
            bot_engine.main()

    except KeyboardInterrupt:
        logging.info("\nShutdown requested by user")
        return 0
    except (ValueError, TypeError) as e:
        logging.info(f"Trading application failed: {e}")
        logging.exception("Trading application error")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
