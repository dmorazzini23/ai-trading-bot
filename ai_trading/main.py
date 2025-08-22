# ruff: noqa: I001, E402
from __future__ import annotations

import argparse
import os
import threading
import time  # AI-AGENT-REF: tests patch main.time.sleep
from threading import Thread
import signal  # AI-AGENT-REF: handle graceful shutdown
from datetime import datetime, UTC  # AI-AGENT-REF: structured signal logs

# AI-AGENT-REF: Load .env BEFORE importing any heavy modules or Settings
from dotenv import load_dotenv, find_dotenv

# AI-AGENT-REF: .env values override inherited environment
_DOTENV_PATH = find_dotenv(usecwd=True)
load_dotenv(_DOTENV_PATH or None, override=True)

from ai_trading.logging import get_logger  # AI-AGENT-REF: early structured logging

logger = get_logger(__name__)
if _DOTENV_PATH:
    logger.info(
        "ENV_LOADED_FROM override=True", extra={"dotenv_path": _DOTENV_PATH}
    )
else:
    logger.info("ENV_LOADED_DEFAULT override=True")

# AI-AGENT-REF: Import only essential modules after env load for import-light entrypoint
from ai_trading.settings import get_seed_int
from ai_trading.config import get_settings
from ai_trading.utils import get_free_port, get_pid_on_port
from ai_trading.utils.prof import StageTimer, SoftBudget
from ai_trading.logging.redact import redact as _redact  # AI-AGENT-REF: startup banner redaction
from ai_trading.net.http import (
    build_retrying_session,
    set_global_session,
)  # AI-AGENT-REF: retrying HTTP session
from ai_trading.position_sizing import (
    resolve_max_position_size,
    _resolve_max_position_size,
)  # AI-AGENT-REF: dynamic max position sizing


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
    # AI-AGENT-REF: legacy feature currently disabled
    return None


def start_performance_monitoring():
    # AI-AGENT-REF: legacy no-op
    return None


# AI-AGENT-REF: Create global config AFTER .env loading and Settings import
from typing import Any

config: Any | None = None


# AI-AGENT-REF: structured logger already bound via get_logger(__name__); avoid rebinding

_SHUTDOWN = threading.Event()  # AI-AGENT-REF: signal-triggered shutdown flag


def _get_int_env(var: str, default: int | None = None) -> int | None:
    """Parse integer from environment. Return default on missing/invalid."""  # AI-AGENT-REF: env parsing helper
    val = os.getenv(var)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning(
            "Invalid integer for %s=%r; using default %r",
            var,
            val,
            default,
        )
        return default


def _install_signal_handlers() -> None:
    """Install SIGINT/SIGTERM handlers."""  # AI-AGENT-REF

    def _handler(signum, frame):
        logger.info(
            "SERVICE_SIGNAL",
            extra={"signal": signum, "ts": datetime.now(tz=UTC).isoformat()},
        )
        _SHUTDOWN.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _validate_runtime_config(cfg, tcfg) -> None:
    """Fail-fast runtime config checks."""  # AI-AGENT-REF

    errors = []
    mode = getattr(cfg, "trading_mode", "balanced")
    if mode not in {"aggressive", "balanced", "conservative"}:
        errors.append(f"TRADING_MODE invalid: {mode}")

    cap = float(getattr(tcfg, "capital_cap", 0.0))
    risk = float(getattr(tcfg, "dollar_risk_limit", 0.0))
    max_pos = float(getattr(tcfg, "max_position_size", 0.0))
    mp_mode = str(
        getattr(tcfg, "max_position_mode", getattr(cfg, "max_position_mode", "STATIC"))
    ).upper()  # AI-AGENT-REF: allow AUTO vs STATIC handling
    if not (0.0 < cap <= 1.0):
        errors.append(f"CAPITAL_CAP out of range: {cap}")
    if not (0.0 < risk <= 1.0):
        errors.append(f"DOLLAR_RISK_LIMIT out of range: {risk}")
    # AI-AGENT-REF: allow AUTO mode to defer resolution; STATIC mode auto-fixes nonpositive
    if not (max_pos > 0.0):
        if mp_mode == "AUTO":
            # AI-AGENT-REF: dynamic resolver will handle later
            pass
        else:
            eq = getattr(tcfg, "equity", getattr(cfg, "equity", None))
            fallback, _src = _resolve_max_position_size(max_pos, cap, eq)
            try:
                # AI-AGENT-REF: update only when field exists
                if hasattr(tcfg, "max_position_size"):
                    tcfg.max_position_size = float(fallback)
                else:
                    import os  # AI-AGENT-REF: runtime env override
                    os.environ["AI_TRADING_MAX_POSITION_SIZE"] = str(float(fallback))
                    logger.warning(
                        "CONFIG_AUTOFIX_FALLBACK_APPLIED_VIA_ENV",
                        extra={
                            "field": "max_position_size",
                            "fallback": float(fallback),
                        },
                    )
            except Exception as e:  # AI-AGENT-REF: log env fallback issues
                logger.warning(
                    "CONFIG_AUTOFIX_FALLBACK_APPLIED_VIA_ENV",
                    extra={
                        "field": "max_position_size",
                        "fallback": float(fallback),
                        "error": repr(e),
                    },
                )

    base_url = str(getattr(cfg, "alpaca_base_url", ""))
    paper = bool(getattr(cfg, "paper", True))
    if paper and "paper" not in base_url:
        errors.append(
            f"ALPACA_BASE_URL should be a paper endpoint when PAPER=True: {base_url}"
        )
    if not paper and "paper" in base_url:
        errors.append(
            f"ALPACA_BASE_URL should be a live endpoint when PAPER=False: {base_url}"
        )

    if errors:
        raise ValueError("; ".join(errors))


def _interruptible_sleep(total_seconds: float) -> None:
    """Sleep in slices while honoring shutdown."""  # AI-AGENT-REF

    remaining = float(total_seconds)
    step = 0.25
    while remaining > 0 and not _SHUTDOWN.is_set():
        time.sleep(min(step, remaining))
        remaining -= step


def validate_environment() -> None:
    """Ensure required environment variables are present and dependencies are available."""
    cfg = get_settings()
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
        rc = _get_run_cycle()
        return rc()

    except (ValueError, TypeError) as e:
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
    from ai_trading import app

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
    settings = get_settings()
    port = int(settings.api_port or 9001)  # AI-AGENT-REF: default API port fallback
    run_flask_app(port, ready_signal)


def start_api_with_signal(
    api_ready: threading.Event, api_error: threading.Event
) -> None:
    """Start API server and signal readiness/errors."""  # AI-AGENT-REF: explicit error handling
    try:
        start_api(api_ready)
    except (OSError, RuntimeError) as e:
        logger.error("Failed to start API: %s", str(e))
        api_error.set()


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
    global config
    config = get_settings()
    S = get_settings()
    logger.info(
        "DATA_CONFIG feed=%s adjustment=%s timeframe=1Day/1Min provider=alpaca",
        S.alpaca_data_feed,
        S.alpaca_adjustment,
    )
    try:
        _validate_runtime_config(config, S)
    except ValueError as e:  # AI-AGENT-REF: narrow runtime config errors
        logger.critical("RUNTIME_CONFIG_INVALID", extra={"error": str(e)})
        raise

    session = build_retrying_session(
        pool_maxsize=int(getattr(config, "http_pool_maxsize", 32)),
        total_retries=int(getattr(config, "http_total_retries", 3)),
        backoff_factor=float(getattr(config, "http_backoff_factor", 0.3)),
        connect_timeout=float(getattr(config, "http_connect_timeout", 5.0)),
        read_timeout=float(getattr(config, "http_read_timeout", 10.0)),
    )
    set_global_session(session)

    logger.info(
        "REQUESTS_POOL_STATS",
        extra={
            "transport": "requests",
            "pool_maxsize": getattr(config, "http_pool_maxsize", 32),
            "retries": getattr(config, "http_total_retries", 3),
            "backoff_factor": getattr(config, "http_backoff_factor", 0.3),
            "connect_timeout": getattr(config, "http_connect_timeout", 5.0),
            "read_timeout": getattr(config, "http_read_timeout", 10.0),
        },
    )

    # AI-AGENT-REF: resolve max position size before startup banner
    try:
        resolved_size, sizing_meta = resolve_max_position_size(config, S, force_refresh=True)
        try:
            setattr(S, "max_position_size", float(resolved_size))
        except (AttributeError, TypeError):
            pass
        if sizing_meta.get("source") == "fallback":
            logger.warning(
                "POSITION_SIZING_FALLBACK",
                extra={**sizing_meta, "resolved": resolved_size},
            )
        else:
            logger.info(
                "POSITION_SIZING_RESOLVED",
                extra={**sizing_meta, "resolved": resolved_size},
            )
    except (ValueError, TypeError) as e:  # pragma: no cover - defensive
        logger.warning("POSITION_SIZING_ERROR", extra={"error": str(e)})

    banner = {
        "mode": getattr(config, "trading_mode", "balanced"),
        "paper": getattr(config, "paper", True),
        "alpaca_base_url": getattr(config, "alpaca_base_url", ""),
        "capital_cap": float(getattr(S, "capital_cap", 0.0)),
        "dollar_risk_limit": float(getattr(S, "dollar_risk_limit", 0.0)),
        "max_position_mode": str(getattr(S, "max_position_mode", getattr(config, "max_position_mode", "STATIC"))).upper(),
        "max_position_size": float(getattr(S, "max_position_size", 0.0)),
    }
    logger.info("STARTUP_BANNER", extra=_redact(banner))

    _install_signal_handlers()

    rc = _get_run_cycle()
    rc()

    # Ensure API is ready before starting trading cycles
    api_ready = threading.Event()
    api_error = threading.Event()

    t = Thread(
        target=start_api_with_signal,
        args=(api_ready, api_error),
        daemon=True,
    )
    t.start()

    # Wait for API to be ready with proper error handling
    # AI-AGENT-REF: Improved timeout handling with more granular checks for test environments
    try:
        # Check for immediate startup errors first
        if api_error.wait(timeout=2):  # Quick check for startup errors
            raise RuntimeError("API failed to start")

        # Wait for API ready signal with reasonable timeout
        if not api_ready.wait(timeout=10):  # Reduced timeout for test environments
            # Check if thread is still alive - if not, there might be an unhandled exception
            if not t.is_alive():
                raise RuntimeError("API thread terminated unexpectedly during startup")
            # Thread is alive but not ready - log and continue in degraded mode
            logger.warning(
                "API startup taking longer than expected, proceeding with degraded functionality"
            )
    except (RuntimeError, TimeoutError, OSError) as e:
        logger.error("Failed to start API", exc_info=e)
        raise RuntimeError("API failed to start") from e

    import os  # AI-AGENT-REF: scheduler config

    S = get_settings()
    from ai_trading.utils.device import pick_torch_device  # AI-AGENT-REF: ML device log

    pick_torch_device()
    raw_tick = os.getenv("HEALTH_TICK_SECONDS") or getattr(
        S, "health_tick_seconds", 300
    )
    try:
        health_tick_seconds = int(raw_tick)
    except ValueError:
        health_tick_seconds = 300
    last_health = time.monotonic()

    # CLI takes precedence; then settings
    env_iter = _get_int_env("SCHEDULER_ITERATIONS")  # AI-AGENT-REF: env precedence for iterations
    raw_iter = (
        args.iterations
        if args.iterations is not None
        else (
            env_iter
            if env_iter is not None
            else (
                getattr(S, "iterations", None)
                or getattr(S, "scheduler_iterations", 0)
            )
        )
    )
    try:
        iterations = int(raw_iter)
    except ValueError:
        iterations = 0

    raw_interval = (
        args.interval if args.interval is not None else getattr(S, "interval", 60)
    )
    try:
        interval = int(raw_interval)
    except ValueError:
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
            except (ValueError, TypeError):  # pragma: no cover - log unexpected errors
                logger.exception("run_cycle failed")
            count += 1

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
            # AI-AGENT-REF: periodic AUTO sizing refresh
            try:
                mode_now = str(getattr(S, "max_position_mode", getattr(config, "max_position_mode", "STATIC"))).upper()
                if mode_now == "AUTO":
                    resolved_size, meta = resolve_max_position_size(config, S, force_refresh=False)
                    if float(getattr(S, "max_position_size", 0.0)) != resolved_size:
                        try:
                            setattr(S, "max_position_size", float(resolved_size))
                        except (AttributeError, TypeError):
                            pass
                        logger.info(
                            "POSITION_SIZING_REFRESHED",
                            extra={**meta, "resolved": resolved_size},
                        )
            except (ValueError, KeyError, TypeError) as e:  # pragma: no cover
                logger.warning("RUNTIME_SIZING_UPDATE_FAILED", exc_info=e)

            _interruptible_sleep(int(max(1, interval)))
            if _SHUTDOWN.is_set():
                logger.info("SERVICE_SHUTDOWN", extra={"reason": "signal"})
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down gracefully")
        return


if __name__ == "__main__":  # pragma: no cover
    main()
