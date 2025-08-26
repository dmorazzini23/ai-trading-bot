from __future__ import annotations
import argparse
import os
import threading
import time
from threading import Thread
import signal
from datetime import datetime, UTC
from ai_trading.env import ensure_dotenv_loaded
from ai_trading.logging import get_logger

logger = get_logger(__name__)
from ai_trading.settings import get_seed_int
from ai_trading.config import get_settings
from ai_trading.utils import get_free_port, get_pid_on_port
from ai_trading.utils.prof import StageTimer, SoftBudget
from ai_trading.logging.redact import redact as _redact
from ai_trading.net.http import build_retrying_session, set_global_session
from ai_trading.position_sizing import resolve_max_position_size, _resolve_max_position_size
from ai_trading.config.management import get_env


def _default_run_cycle():
    return None


run_cycle = _default_run_cycle


def _get_run_cycle():
    global run_cycle
    if run_cycle is _default_run_cycle:
        from ai_trading.runner import run_cycle as _runner_run_cycle

        run_cycle = _runner_run_cycle
    return run_cycle


def get_memory_optimizer():
    from ai_trading.config import get_settings

    S = get_settings()
    if not S.enable_memory_optimization:
        return None
    from ai_trading.utils import memory_optimizer

    return memory_optimizer


def optimize_memory():
    from ai_trading.config import get_settings

    S = get_settings()
    if not S.enable_memory_optimization:
        return {}
    from ai_trading.utils import memory_optimizer

    return memory_optimizer.report_memory_use()


from typing import Any

config: Any | None = None
_SHUTDOWN = threading.Event()


def _get_int_env(var: str, default: int | None = None) -> int | None:
    """Parse integer from environment. Return default on missing/invalid."""
    val = get_env(var)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        logger.warning("Invalid integer for %s=%r; using default %r", var, val, default)
        return default


def _as_float(val, default: float = 0.0) -> float:
    """Best-effort float conversion that tolerates None/str."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _install_signal_handlers() -> None:
    """Install SIGINT/SIGTERM handlers."""

    def _handler(signum, frame):
        logger.info("SERVICE_SIGNAL", extra={"signal": signum, "ts": datetime.now(tz=UTC).isoformat()})
        _SHUTDOWN.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _validate_runtime_config(cfg, tcfg) -> None:
    """Fail-fast runtime config checks."""
    errors = []
    mode = getattr(cfg, "trading_mode", "balanced")
    if mode not in {"aggressive", "balanced", "conservative"}:
        errors.append(f"TRADING_MODE invalid: {mode}")
    cap = _as_float(getattr(tcfg, "capital_cap", 0.0), 0.0)
    risk = _as_float(getattr(tcfg, "dollar_risk_limit", 0.0), 0.0)
    max_pos = _as_float(getattr(tcfg, "max_position_size", None), 0.0)
    env_pos = _as_float(get_env("AI_TRADING_MAX_POSITION_SIZE"), 0.0)
    user_pos = env_pos if env_pos > 0.0 else max_pos
    mp_mode = str(getattr(tcfg, "max_position_mode", getattr(cfg, "max_position_mode", "STATIC"))).upper()
    if not 0.0 < cap <= 1.0:
        errors.append(f"CAPITAL_CAP out of range: {cap}")
    if not 0.0 < risk <= 1.0:
        errors.append(f"DOLLAR_RISK_LIMIT out of range: {risk}")
    if not user_pos > 0.0:
        if mp_mode == "AUTO":
            pass
        else:
            eq = getattr(tcfg, "equity", getattr(cfg, "equity", None))
            fallback, _src = _resolve_max_position_size(0.0, cap, eq)
            try:
                if hasattr(tcfg, "max_position_size"):
                    tcfg.max_position_size = float(fallback)
                else:
                    os.environ["AI_TRADING_MAX_POSITION_SIZE"] = str(float(fallback))
                    logger.warning(
                        "CONFIG_AUTOFIX_FALLBACK_APPLIED_VIA_ENV",
                        extra={"field": "max_position_size", "fallback": float(fallback)},
                    )
            except (ValueError, OSError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
                logger.warning(
                    "CONFIG_AUTOFIX_FALLBACK_APPLIED_VIA_ENV",
                    extra={"field": "max_position_size", "fallback": float(fallback), "error": repr(e)},
                )
    else:
        if hasattr(tcfg, "max_position_size"):
            tcfg.max_position_size = float(user_pos)
        else:
            os.environ["AI_TRADING_MAX_POSITION_SIZE"] = str(float(user_pos))
    base_url = str(getattr(cfg, "alpaca_base_url", ""))
    paper = bool(getattr(cfg, "paper", True))
    if paper and "paper" not in base_url:
        errors.append(f"ALPACA_BASE_URL should be a paper endpoint when PAPER=True: {base_url}")
    if not paper and "paper" in base_url:
        errors.append(f"ALPACA_BASE_URL should be a live endpoint when PAPER=False: {base_url}")
    if errors:
        raise ValueError("; ".join(errors))


def _interruptible_sleep(total_seconds: float) -> None:
    """Sleep in slices while honoring shutdown."""
    remaining = float(total_seconds)
    step = 0.25
    while remaining > 0 and (not _SHUTDOWN.is_set()):
        time.sleep(min(step, remaining))
        remaining -= step


def validate_environment() -> None:
    """Ensure required environment variables are present and dependencies are available."""
    from ai_trading.config.management import validate_required_env

    validate_required_env()
    _ = get_settings()
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
    ensure_dotenv_loaded()
    global config
    from ai_trading.logging import setup_logging, validate_logging_setup

    logger = setup_logging(log_file="logs/bot.log", debug=False)
    validation_result = validate_logging_setup()
    if not validation_result["validation_passed"]:
        logger.error("Logging validation failed: %s", validation_result["issues"])
    logger.info("Application startup - logging configured once")
    try:
        config = get_settings()
        validate_environment()
        memory_optimizer = get_memory_optimizer()
        if memory_optimizer:
            memory_optimizer.enable_low_memory_mode()
            logger.info("Memory optimization enabled")
        logger.info("Bot startup complete - entering main loop")
        rc = _get_run_cycle()
        return rc()
    except (ValueError, TypeError) as e:
        logger.error("Bot startup failed: %s", e, exc_info=True)
        return 1


def run_flask_app(port: int = 5000, ready_signal: threading.Event = None) -> None:
    """Launch Flask API on an available port."""
    max_attempts = 10
    original_port = port
    for _attempt in range(max_attempts):
        if not get_pid_on_port(port):
            break
        port += 1
    else:
        free_port = get_free_port()
        if free_port is None:
            raise RuntimeError(f"Could not find available port starting from {original_port}")
        port = free_port
    from ai_trading import app

    application = app.create_app()
    if ready_signal is not None:
        logger.info(f"Flask app created successfully, signaling ready on port {port}")
        ready_signal.set()
    logger.info(f"Starting Flask app on 0.0.0.0:{port}")
    application.run(host="0.0.0.0", port=port, debug=False)


def start_api(ready_signal: threading.Event = None) -> None:
    """Spin up the Flask API server."""
    ensure_dotenv_loaded()
    settings = get_settings()
    port = int(settings.api_port or 9001)
    run_flask_app(port, ready_signal)


def start_api_with_signal(api_ready: threading.Event, api_error: threading.Event) -> None:
    """Start API server and signal readiness/errors."""
    try:
        start_api(api_ready)
    except (OSError, RuntimeError) as e:
        logger.error("Failed to start API: %s", str(e))
        api_error.set()


def parse_cli(argv: list[str] | None = None):
    """Parse CLI arguments, tolerating unknown flags."""
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument("--iterations")
    parser.add_argument("--interval")
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    """Start the API thread and repeatedly run trading cycles."""
    ensure_dotenv_loaded()
    args = parse_cli(argv)
    global config
    config = get_settings()
    S = get_settings()
    logger.info(
        "DATA_CONFIG feed=%s adjustment=%s timeframe=1Day/1Min provider=alpaca", S.alpaca_data_feed, S.alpaca_adjustment
    )
    try:
        _validate_runtime_config(config, S)
    except ValueError as e:
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
    try:
        resolved_size, sizing_meta = resolve_max_position_size(config, S, force_refresh=True)
        try:
            setattr(S, "max_position_size", float(resolved_size))
        except (AttributeError, TypeError):
            pass
        if sizing_meta.get("source") == "fallback":
            logger.warning("POSITION_SIZING_FALLBACK", extra={**sizing_meta, "resolved": resolved_size})
        else:
            logger.info("POSITION_SIZING_RESOLVED", extra={**sizing_meta, "resolved": resolved_size})
    except (ValueError, TypeError) as e:
        logger.warning("POSITION_SIZING_ERROR", extra={"error": str(e)})
    banner = {
        "mode": getattr(config, "trading_mode", "balanced"),
        "paper": getattr(config, "paper", True),
        "alpaca_base_url": getattr(config, "alpaca_base_url", ""),
        "capital_cap": _as_float(getattr(S, "capital_cap", 0.0), 0.0),
        "dollar_risk_limit": _as_float(getattr(S, "dollar_risk_limit", 0.0), 0.0),
        "max_position_mode": str(
            getattr(S, "max_position_mode", getattr(config, "max_position_mode", "STATIC"))
        ).upper(),
        "max_position_size": _as_float(getattr(S, "max_position_size", None), 0.0),
    }
    logger.info("STARTUP_BANNER", extra=_redact(banner))
    _install_signal_handlers()
    rc = _get_run_cycle()
    warmup_code = 0
    try:
        warmup_code = rc()
    except SystemExit as e:
        warmup_code = int(getattr(e, "code", 1) or 1)
        logger.error(
            "Warm-up run_cycle triggered SystemExit; continuing into main loop",
            exc_info=e,
        )
    except Exception as e:  # noqa: BLE001
        warmup_code = 1
        logger.exception(
            "Warm-up run_cycle failed; continuing into main loop",
            exc_info=e,
        )
    else:
        if warmup_code:
            logger.warning(
                "Warm-up returned non-zero code=%s; continuing into main loop",
                warmup_code,
            )
    api_ready = threading.Event()
    api_error = threading.Event()
    t = Thread(target=start_api_with_signal, args=(api_ready, api_error), daemon=True)
    t.start()
    try:
        if api_error.wait(timeout=2):
            raise RuntimeError("API failed to start")
        if not api_ready.wait(timeout=10):
            if not t.is_alive():
                raise RuntimeError("API thread terminated unexpectedly during startup")
            logger.warning("API startup taking longer than expected, proceeding with degraded functionality")
    except (RuntimeError, TimeoutError, OSError) as e:
        logger.error("Failed to start API", exc_info=e)
        raise RuntimeError("API failed to start") from e
    import os

    S = get_settings()
    from ai_trading.utils.device import get_device  # AI-AGENT-REF: guard torch import

    get_device()
    raw_tick = get_env("HEALTH_TICK_SECONDS") or getattr(S, "health_tick_seconds", 300)
    try:
        health_tick_seconds = int(raw_tick)
    except (ValueError, TypeError):
        health_tick_seconds = 300
    last_health = time.monotonic()
    env_iter = _get_int_env("SCHEDULER_ITERATIONS")
    raw_iter = (
        args.iterations
        if args.iterations is not None
        else (
            env_iter
            if env_iter is not None
            else getattr(S, "iterations", None) or getattr(S, "scheduler_iterations", 0)
        )
    )
    try:
        iterations = int(raw_iter)
    except ValueError:
        iterations = 0
    raw_interval = args.interval if args.interval is not None else getattr(S, "interval", 60)
    try:
        interval = int(raw_interval)
    except ValueError:
        interval = 60
    seed = get_seed_int()
    logger.info("Runtime defaults resolved", extra={"iterations": iterations, "interval": interval, "seed": seed})
    count = 0
    memory_check_interval = 10
    try:
        while iterations <= 0 or count < iterations:
            raw_fraction = get_env("CYCLE_BUDGET_FRACTION", 0.8)
            try:
                fraction = float(raw_fraction)
            except (TypeError, ValueError):
                fraction = 0.8
            budget = SoftBudget(interval_sec=float(interval), fraction=fraction)
            try:
                if count % memory_check_interval == 0:
                    gc_result = optimize_memory()
                    if gc_result.get("objects_collected", 0) > 100:
                        logger.info(f"Cycle {count}: Garbage collected {gc_result['objects_collected']} objects")
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
            except (ValueError, TypeError):
                logger.exception("run_cycle failed")
            count += 1
            logger.info("CYCLE_TIMING", extra={"elapsed_ms": budget.elapsed_ms(), "within_budget": not budget.over()})
            now_mono = time.monotonic()
            if now_mono - last_health >= max(30, health_tick_seconds):
                logger.info("HEALTH_TICK", extra={"iteration": count, "interval": interval})
                last_health = now_mono
            try:
                mode_now = str(getattr(S, "max_position_mode", getattr(config, "max_position_mode", "STATIC"))).upper()
                if mode_now == "AUTO":
                    resolved_size, meta = resolve_max_position_size(config, S, force_refresh=False)
                    if float(getattr(S, "max_position_size", 0.0)) != resolved_size:
                        try:
                            setattr(S, "max_position_size", float(resolved_size))
                        except (AttributeError, TypeError):
                            pass
                        logger.info("POSITION_SIZING_REFRESHED", extra={**meta, "resolved": resolved_size})
            except (ValueError, KeyError, TypeError) as e:
                logger.warning("RUNTIME_SIZING_UPDATE_FAILED", exc_info=e)
            _interruptible_sleep(int(max(1, interval)))
            if _SHUTDOWN.is_set():
                logger.info("SERVICE_SHUTDOWN", extra={"reason": "signal"})
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down gracefully")
        return


if __name__ == "__main__":
    main()
