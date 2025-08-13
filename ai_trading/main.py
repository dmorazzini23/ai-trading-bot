from __future__ import annotations

import logging
import threading
import time
from threading import Thread

# AI-AGENT-REF: Load .env BEFORE importing any heavy modules or Settings
from dotenv import load_dotenv

load_dotenv()

# AI-AGENT-REF: Import only essential modules at top level for import-light entrypoint
from ai_trading.config import Settings, get_settings
from ai_trading.utils import get_free_port, get_pid_on_port

# AI-AGENT-REF: Import memory optimization and performance monitoring
def get_memory_optimizer():
    from ai_trading.config import get_settings
    S = get_settings()
    if not S.enable_memory_optimization:
        return None

    from ai_trading.utils import memory_optimizer  # AI-AGENT-REF: stable import path
    return memory_optimizer


def optimize_memory():
    from ai_trading.config import get_settings
    S = get_settings()
    if not S.enable_memory_optimization:
        return {}

    from ai_trading.utils import memory_optimizer  # AI-AGENT-REF: stable import path
    return memory_optimizer.report_memory_use()


def get_performance_monitor():
    from ai_trading.config import get_settings
    S = get_settings()
    if not S.enable_memory_optimization:  # Reuse same flag for both features
        return None

    from performance_monitor import (  # noqa: F401 - external optional dep
        get_performance_monitor as _get_performance_monitor,
    )
    return _get_performance_monitor()


def start_performance_monitoring():
    from ai_trading.config import get_settings
    S = get_settings()
    if not S.enable_memory_optimization:
        return

    from performance_monitor import (  # noqa: F401 - external optional dep
        start_performance_monitoring as _start_performance_monitoring,
    )
    _start_performance_monitoring()
# AI-AGENT-REF: Create global config AFTER .env loading and Settings import
config: Settings | None = None

logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """Ensure required environment variables are present and dependencies are available."""
    # Check critical environment variables
    if not S.webhook_secret:
        raise RuntimeError("WEBHOOK_SECRET is required")
    if not S.alpaca_api_key or not S.alpaca_secret_key:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")

    # Check optional but important dependencies
    import alpaca_trade_api
    import alpaca
    # Validate data directories exist
    import os

    data_dir = "data"
    if not os.path.exists(data_dir):
        logger.info("Creating data directory: %s", data_dir)
        os.makedirs(data_dir, exist_ok=True)

    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        logger.info("Creating logs directory: %s", logs_dir)
        os.makedirs(logs_dir, exist_ok=True)


def run_bot(*_a, **_k) -> int:
    """
    Main entry point for the trading bot.

    Sets up logging, validates configuration, and starts the bot.
    """
    global config

    # AI-AGENT-REF: Setup logging exactly once at startup
    from ai_trading.logging import setup_logging, validate_logging_setup

    logger = setup_logging(log_file="logs/bot.log", debug=False)

    # Validate single handler setup to detect duplicates
    validation_result = validate_logging_setup()
    if not validation_result["validation_passed"]:
        logger.error("Logging validation failed: %s", validation_result["issues"])
        # Don't fail startup, just warn about potential duplicates

    logger.info("Application startup - logging configured once")

    try:
        # Load configuration
        config = get_settings()
        validate_environment()

        # Memory optimization and performance monitoring
        memory_optimizer = get_memory_optimizer()
        performance_monitor = get_performance_monitor()

        if memory_optimizer:
            memory_optimizer.enable_low_memory_mode()
            logger.info("Memory optimization enabled")

        if performance_monitor:
            start_performance_monitoring()
            logger.info("Performance monitoring started")

        logger.info("Bot startup complete - entering main loop")
        # Defer runner import to avoid import-time side effects
        from ai_trading.runner import run_cycle
        return run_cycle()

    except Exception as e:
        logger.error("Bot startup failed: %s", e, exc_info=True)
        return 1


def run_flask_app(port: int = 5000, ready_signal: threading.Event = None) -> None:
    """Launch Flask API on an available port."""
    # AI-AGENT-REF: simplified port fallback logic with get_free_port fallback
    max_attempts = 10
    original_port = port

    for _attempt in range(max_attempts):
        if not get_pid_on_port(port):
            break
        port += 1
    else:
        # If consecutive ports are all occupied, use get_free_port as fallback
        free_port = get_free_port()
        if free_port is None:
            raise RuntimeError(
                f"Could not find available port starting from {original_port}"
            )
        port = free_port

    # Defer app import to avoid import-time side effects
    import ai_trading.app as app
    application = app.create_app()

    # AI-AGENT-REF: Signal ready immediately after Flask app creation for faster startup
    if ready_signal is not None:
        logger.info(f"Flask app created successfully, signaling ready on port {port}")
        ready_signal.set()

    logger.info(f"Starting Flask app on 0.0.0.0:{port}")
    application.run(host="0.0.0.0", port=port)


def start_api(ready_signal: threading.Event = None) -> None:
    """Spin up the Flask API server."""
    settings = get_settings()
    run_flask_app(settings.api_port, ready_signal)


def main() -> None:
    """Start the API thread and repeatedly run trading cycles."""
    load_dotenv()
    global config
    config = get_settings()
    # Defer runner import to avoid import-time side effects
    from ai_trading.runner import run_cycle
    run_cycle()

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
                logger.warning(
                    "API startup taking longer than expected, proceeding with degraded functionality"
                )
                # In test environments, we might want to continue without the API
                test_mode = config.scheduler_iterations != 0
                if not test_mode:
                    raise RuntimeError(
                        "API startup timeout - trading cannot proceed without API ready"
                    )
    except RuntimeError:
        # Re-raise runtime errors as-is
        raise
    except Exception as e:
        # Handle any other synchronization errors
        logger.error("Unexpected error during API startup synchronization: %s", e)
        raise RuntimeError(f"API startup synchronization failed: {e}")

    interval = config.scheduler_sleep_seconds
    iterations = config.scheduler_iterations  # AI-AGENT-REF: test hook
    count = 0

    # AI-AGENT-REF: Track memory optimization cycles
    memory_check_interval = 10  # Check every 10 cycles

    while iterations <= 0 or count < iterations:
        try:
            # AI-AGENT-REF: Periodic memory optimization
            if count % memory_check_interval == 0:
                gc_result = optimize_memory()
                if gc_result.get("objects_collected", 0) > 100:
                    logger.info(
                        f"Cycle {count}: Garbage collected {gc_result['objects_collected']} objects"
                    )

            # Defer runner import to avoid import-time side effects
            from ai_trading.runner import run_cycle
            run_cycle()
        except Exception:  # pragma: no cover - log unexpected errors
            logger.exception("run_cycle failed")
        count += 1
        time.sleep(interval)


if __name__ == "__main__":
    main()


from importlib.util import find_spec
from ai_trading.config import get_settings
S = get_settings()
if getattr(S, "enable_performance_monitoring", False):
    if find_spec("performance_monitor") is None:
        raise RuntimeError("Feature enabled but module 'performance_monitor' not installed")
    from performance_monitor import *  # noqa: F401
