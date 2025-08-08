from __future__ import annotations

import logging
import os
import time
from threading import Thread
import threading

# AI-AGENT-REF: Load .env BEFORE importing any heavy modules or Settings
from dotenv import load_dotenv
load_dotenv()

# AI-AGENT-REF: Import Settings AFTER .env is loaded to prevent import-time crashes
from ai_trading.config import Settings
import ai_trading.app as app
from ai_trading.runner import run_cycle
from ai_trading.utils import set_random_seeds, ensure_deterministic_training, get_pid_on_port, get_free_port

# AI-AGENT-REF: Import memory optimization and performance monitoring
try:
    from memory_optimizer import get_memory_optimizer, optimize_memory
    from performance_monitor import get_performance_monitor, start_performance_monitoring
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    # Fallback if modules not available
    PERFORMANCE_MONITORING_AVAILABLE = False
    def get_memory_optimizer():
        return None
    def optimize_memory():
        return {}
    def get_performance_monitor():
        return None
    def start_performance_monitoring():
        pass

# AI-AGENT-REF: Create global config AFTER .env loading and Settings import
config = Settings()

logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """Ensure required environment variables are present."""
    if not config.WEBHOOK_SECRET:
        raise RuntimeError("WEBHOOK_SECRET is required")
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")


def run_bot(*_a, **_k) -> int:
    """
    Compatibility wrapper to execute one trading cycle.
    
    This function imports runner components after .env is guaranteed to be loaded
    to prevent import-time crashes.
    """
    # AI-AGENT-REF: Ensure .env is loaded before importing runner components
    load_dotenv()
    
    # AI-AGENT-REF: Import runner after .env is guaranteed loaded
    from ai_trading.runner import run_cycle
    
    # AI-AGENT-REF: run cycle directly instead of spawning subprocesses
    run_cycle()
    return 0


def run_flask_app(port: int = 5000, ready_signal: threading.Event = None) -> None:
    """Launch Flask API on an available port."""
    # AI-AGENT-REF: simplified port fallback logic with get_free_port fallback
    max_attempts = 10
    original_port = port

    for attempt in range(max_attempts):
        if not get_pid_on_port(port):
            break
        port += 1
    else:
        # If consecutive ports are all occupied, use get_free_port as fallback
        free_port = get_free_port()
        if free_port is None:
            raise RuntimeError(f"Could not find available port starting from {original_port}")
        port = free_port

    application = app.create_app()
    
    # AI-AGENT-REF: Signal ready immediately after Flask app creation for faster startup
    if ready_signal is not None:
        logger.info(f"Flask app created successfully, signaling ready on port {port}")
        ready_signal.set()
    
    logger.info(f"Starting Flask app on 0.0.0.0:{port}")
    application.run(host="0.0.0.0", port=port)


def start_api(ready_signal: threading.Event = None) -> None:
    """Spin up the Flask API server."""
    run_flask_app(int(os.getenv("API_PORT", 9001)), ready_signal)


def main() -> None:
    """Start the API thread and repeatedly run trading cycles."""
    # AI-AGENT-REF: Ensure .env is loaded before any Settings usage
    load_dotenv()
    
    # AI-AGENT-REF: Set up deterministic behavior early
    set_random_seeds()
    ensure_deterministic_training()
    
    validate_environment()

    # AI-AGENT-REF: Initialize performance monitoring
    if PERFORMANCE_MONITORING_AVAILABLE:
        logger.info("Starting performance monitoring...")
        start_performance_monitoring()
        
        # Configure memory optimizer for better performance
        memory_optimizer = get_memory_optimizer()
        if memory_optimizer:
            logger.info("Memory optimizer initialized")

    # Ensure API is ready before starting trading cycles
    api_ready = threading.Event()
    api_error = threading.Event()
    api_exception = None

    def start_api_with_signal():
        try:
            start_api(api_ready)  # Pass the ready signal to be set before blocking run
        except Exception as e:
            # AI-AGENT-REF: Add proper timeout error handling for API startup synchronization
            nonlocal api_exception
            api_exception = e
            logger.error("Failed to start API: %s", e)
            api_error.set()

    t = Thread(target=start_api_with_signal, daemon=True)
    t.start()

    # Wait for API to be ready with proper error handling
    # AI-AGENT-REF: Improved timeout handling with more granular checks for test environments
    try:
        # Check for immediate startup errors first
        if api_error.wait(timeout=2):  # Quick check for startup errors
            raise RuntimeError(f"API failed to start: {api_exception}")
        
        # Wait for API ready signal with reasonable timeout
        if not api_ready.wait(timeout=10):  # Reduced timeout for test environments
            # Check if thread is still alive - if not, there might be an unhandled exception
            if not t.is_alive():
                raise RuntimeError("API thread terminated unexpectedly during startup")
            else:
                # Thread is alive but not ready - this is a true timeout
                logger.warning("API startup taking longer than expected, proceeding with degraded functionality")
                # In test environments, we might want to continue without the API
                test_mode = os.getenv("SCHEDULER_ITERATIONS", "0") != "0"
                if not test_mode:
                    raise RuntimeError("API startup timeout - trading cannot proceed without API ready")
    except RuntimeError:
        # Re-raise runtime errors as-is
        raise
    except Exception as e:
        # Handle any other synchronization errors
        logger.error("Unexpected error during API startup synchronization: %s", e)
        raise RuntimeError(f"API startup synchronization failed: {e}")

    interval = int(os.getenv("SCHEDULER_SLEEP_SECONDS", 30))
    iterations = int(os.getenv("SCHEDULER_ITERATIONS", 0))  # AI-AGENT-REF: test hook
    count = 0
    
    # AI-AGENT-REF: Track memory optimization cycles
    memory_check_interval = 10  # Check every 10 cycles
    
    while iterations <= 0 or count < iterations:
        try:
            # AI-AGENT-REF: Periodic memory optimization
            if PERFORMANCE_MONITORING_AVAILABLE and count % memory_check_interval == 0:
                gc_result = optimize_memory()
                if gc_result.get('objects_collected', 0) > 100:
                    logger.info(f"Cycle {count}: Garbage collected {gc_result['objects_collected']} objects")
            
            run_cycle()
        except Exception:  # pragma: no cover - log unexpected errors
            logger.exception("run_cycle failed")
        count += 1
        time.sleep(interval)


if __name__ == "__main__":
    main()
