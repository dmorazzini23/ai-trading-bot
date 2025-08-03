from __future__ import annotations

import logging
import os
import time
from threading import Thread
import threading

from dotenv import load_dotenv
from validate_env import Settings

import ai_trading.app as app
from ai_trading.runner import run_cycle
import utils

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

config = Settings()

logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """Ensure required environment variables are present."""
    if not config.WEBHOOK_SECRET:
        raise RuntimeError("WEBHOOK_SECRET is required")
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")


def run_bot(*_a, **_k) -> int:
    """Compatibility wrapper to execute one trading cycle."""
    # AI-AGENT-REF: run cycle directly instead of spawning subprocesses
    run_cycle()
    return 0


def run_flask_app(port: int = 5000) -> None:
    """Launch Flask API on an available port."""
    # AI-AGENT-REF: simplified port fallback logic with get_free_port fallback
    max_attempts = 10
    original_port = port

    for attempt in range(max_attempts):
        if not utils.get_pid_on_port(port):
            break
        port += 1
    else:
        # If consecutive ports are all occupied, use get_free_port as fallback
        free_port = utils.get_free_port()
        if free_port is None:
            raise RuntimeError(f"Could not find available port starting from {original_port}")
        port = free_port

    application = app.create_app()
    application.run(host="0.0.0.0", port=port)


def start_api() -> None:
    """Spin up the Flask API server."""
    run_flask_app(int(os.getenv("API_PORT", 9001)))


def main() -> None:
    """Start the API thread and repeatedly run trading cycles."""
    load_dotenv()
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
            start_api()
            api_ready.set()
        except Exception as e:
            # AI-AGENT-REF: Add proper timeout error handling for API startup synchronization
            nonlocal api_exception
            api_exception = e
            logger.error("Failed to start API: %s", e)
            api_error.set()

    t = Thread(target=start_api_with_signal, daemon=True)
    t.start()

    # Wait for API to be ready with proper error handling
    if api_error.wait(timeout=10):
        raise RuntimeError(f"API failed to start: {api_exception}")
    elif not api_ready.wait(timeout=10):
        raise RuntimeError("API startup timeout - trading cannot proceed without API ready")

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
