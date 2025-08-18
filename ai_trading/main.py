from __future__ import annotations

import argparse
import logging
import os
import threading
import time  # AI-AGENT-REF: tests patch main.time.sleep
from threading import Thread

# AI-AGENT-REF: Load .env BEFORE importing any heavy modules or Settings
from dotenv import load_dotenv

# AI-AGENT-REF: Import only essential modules at top level for import-light entrypoint
from ai_trading.config import get_settings as get_config
from ai_trading.settings import (
    get_seed_int,
    get_settings,
)  # AI-AGENT-REF: runtime env settings
from ai_trading.utils import get_free_port, get_pid_on_port
from ai_trading.utils import sleep as psleep
from ai_trading.utils.prof import StageTimer, SoftBudget
from ai_trading.utils import http as http_utils


# AI-AGENT-REF: expose run_cycle for monkeypatching
def _default_run_cycle():
    return None


run_cycle = _default_run_cycle


def _get_run_cycle():
    global run_cycle
    if run_cycle is _default_run_cycle:
        from ai_trading.runner import run_cycle as _runner_run_cycle

        run_cycle = _runner_run_cycle
    return run_cycle


# AI-AGENT-REF: Import memory optimization only
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
    # AI-AGENT-REF: shim removed; feature currently disabled
    return None


def start_performance_monitoring():
    # AI-AGENT-REF: shim removed; no-op
    return None


# AI-AGENT-REF: Create global config AFTER .env loading and Settings import
from typing import Any

config: Any | None = None


logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """Ensure required environment variables are present and dependencies are available."""
    cfg = get_config()
    # Check critical environment variables
    if not cfg.webhook_secret:
        raise RuntimeError("WEBHOOK_SECRET is required")
    if not cfg.alpaca_api_key or not cfg.alpaca_secret_key_plain:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY are required"
        )  # AI-AGENT-REF: check plain secret

    # Check optional but important dependencies
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
        config = get_config()
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
        rc = _get_run_cycle()
        return rc()

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
    # AI-AGENT-REF: disable debug mode in production server
    application.run(host="0.0.0.0", port=port, debug=False)


def start_api(ready_signal: threading.Event = None) -> None:
    """Spin up the Flask API server."""
    settings = get_config()
    port = int(settings.api_port or 9001)  # AI-AGENT-REF: default API port fallback
    run_flask_app(port, ready_signal)


def parse_cli(argv: list[str] | None = None):
    """Parse CLI arguments, tolerating unknown flags."""  # AI-AGENT-REF: tolerant parser
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument("--iterations")
    parser.add_argument("--interval")
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    """Start the API thread and repeatedly run trading cycles."""
    args = parse_cli(argv)

    load_dotenv()
    global config
    config = get_config()
    rc = _get_run_cycle()
    rc()

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
            # Thread is alive but not ready - log and continue in degraded mode
            logger.warning(
                "API startup taking longer than expected, proceeding with degraded functionality"
            )
    except RuntimeError:
        # Re-raise runtime errors as-is
        raise
    except Exception as e:
        # Handle any other synchronization errors
        logger.error("Unexpected error during API startup synchronization: %s", e)
        raise RuntimeError(f"API startup synchronization failed: {e}")

    import os  # AI-AGENT-REF: scheduler config

    S = get_settings()
    from ai_trading.utils.device import pick_torch_device  # AI-AGENT-REF: ML device log

    pick_torch_device()
    raw_tick = os.getenv("HEALTH_TICK_SECONDS") or getattr(
        S, "health_tick_seconds", 300
    )
    try:
        health_tick_seconds = int(raw_tick)
    except Exception:
        health_tick_seconds = 300
    last_health = time.monotonic()

    # CLI takes precedence; then settings
    raw_iter = (
        args.iterations if args.iterations is not None else getattr(S, "iterations", 0)
    )
    try:
        iterations = int(raw_iter)
    except Exception:
        iterations = 0

    raw_interval = (
        args.interval if args.interval is not None else getattr(S, "interval", 60)
    )
    try:
        interval = int(raw_interval)
    except Exception:
        interval = 60

    seed = get_seed_int()

    # AI-AGENT-REF: log resolved runtime defaults
    logger.info(
        "Runtime defaults resolved",
        extra={"iterations": iterations, "interval": interval, "seed": seed},
    )

    count = 0

    # AI-AGENT-REF: Track memory optimization cycles
    memory_check_interval = 10  # Check every 10 cycles

    try:
        while iterations <= 0 or count < iterations:
            budget = SoftBudget(
                interval_sec=float(interval),
                fraction=float(os.getenv("CYCLE_BUDGET_FRACTION", 0.8)),
            )
            try:
                # AI-AGENT-REF: Periodic memory optimization
                if count % memory_check_interval == 0:
                    gc_result = optimize_memory()
                    if gc_result.get("objects_collected", 0) > 100:
                        logger.info(
                            f"Cycle {count}: Garbage collected {gc_result['objects_collected']} objects"
                        )

                with StageTimer(logger, "CYCLE_FETCH"):
                    pass
                if budget.over():
                    logger.warning("BUDGET_OVER", extra={"stage": "CYCLE_FETCH"})

                with StageTimer(logger, "CYCLE_COMPUTE"):
                    rc()
                if budget.over():
                    logger.warning("BUDGET_OVER", extra={"stage": "CYCLE_COMPUTE"})

                with StageTimer(logger, "CYCLE_EXECUTE"):
                    pass
                if budget.over():
                    logger.warning("BUDGET_OVER", extra={"stage": "CYCLE_EXECUTE"})
            except Exception:  # pragma: no cover - log unexpected errors
                logger.exception("run_cycle failed")
            count += 1

            logger.info("HTTP_POOL_STATS", extra=http_utils.pool_stats())
            logger.info(
                "CYCLE_TIMING",
                extra={
                    "elapsed_ms": budget.elapsed_ms(),
                    "within_budget": not budget.over(),
                },
            )

            now_mono = time.monotonic()
            if now_mono - last_health >= max(30, health_tick_seconds):
                logger.info(
                    "HEALTH_TICK", extra={"iteration": count, "interval": interval}
                )
                last_health = now_mono

            psleep(int(max(1, interval)))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down gracefully")
        return


if __name__ == "__main__":  # pragma: no cover
    main()
