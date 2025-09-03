from __future__ import annotations
import argparse
import os
import threading
import time
import logging
from threading import Thread
import signal
from datetime import datetime, UTC
from zoneinfo import ZoneInfo
from ai_trading.env import ensure_dotenv_loaded

# Ensure environment variables are loaded before any logging configuration
ensure_dotenv_loaded()

import ai_trading.logging as _logging

# Determine log file from environment
LOG_FILE = os.getenv("BOT_LOG_FILE", "logs/bot.log")

# Reset any prior logging configuration to apply new settings
if getattr(_logging, "_listener", None):
    try:
        _logging._listener.stop()
    except Exception:  # pragma: no cover - best effort cleanup
        pass
    _logging._listener = None
_logging._configured = False
_logging._LOGGING_CONFIGURED = False
logging.getLogger().handlers.clear()

# Configure logging with the desired file
# Logging must be initialized once here before importing heavy modules like
# ``ai_trading.core.bot_engine``.
_logging.setup_logging(log_file=LOG_FILE)

# Module logger
logger = _logging.get_logger(__name__)

from ai_trading.settings import get_seed_int
from ai_trading.config import get_settings
from ai_trading.utils import get_free_port, get_pid_on_port
from ai_trading.utils.prof import StageTimer, SoftBudget
from ai_trading.logging.redact import redact as _redact, redact_env
from ai_trading.net.http import build_retrying_session, set_global_session, mount_host_retry_profile
from ai_trading.utils.http import clamp_request_timeout
from ai_trading.utils.base import is_market_open as _is_market_open_base
from ai_trading.position_sizing import resolve_max_position_size, _get_equity_from_alpaca, _CACHE
from ai_trading.config.management import (
    get_env,
    validate_required_env,
    reload_env,
    _resolve_alpaca_env,
)
from ai_trading.metrics import get_histogram, get_counter
from time import monotonic as _mono


def preflight_import_health() -> None:
    """Best-effort import preflight to surface missing deps early."""
    import importlib
    import os

    if os.environ.get("IMPORT_PREFLIGHT_DISABLED", "").lower() in {"1", "true"}:
        return

    core_modules = [
        "ai_trading.core.bot_engine",
        "ai_trading.risk.engine",
        "ai_trading.rl_trading",
        "ai_trading.telemetry.metrics_logger",
        "alpaca.trading.client",
    ]
    for mod in core_modules:
        try:
            importlib.import_module(mod)
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - surface import issues
            logger.error(
                "IMPORT_PREFLIGHT_FAILED",
                extra={
                    "module_name": mod,
                    "error": repr(exc),
                    "exc_type": exc.__class__.__name__,
                },
            )
            raise SystemExit(1)
    logger.info("IMPORT_PREFLIGHT_OK")
    try:
        importlib.import_module("ai_trading.core.bot_engine").get_trade_logger()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("TRADE_LOG_INIT_FAILED", extra={"error": repr(exc)})


def run_cycle() -> None:
    """Execute a single trading cycle using the core bot engine."""
    from ai_trading.core.bot_engine import (
        BotState,
        run_all_trades_worker,
        get_ctx,
        get_trade_logger,
    )
    from ai_trading.core.runtime import (
        build_runtime,
        REQUIRED_PARAM_DEFAULTS,
        enhance_runtime_with_context,
    )
    from ai_trading.config.management import TradingConfig
    from ai_trading.config import get_settings

    # Ensure trade log file exists before any trade-log reads occur. The
    # ``get_trade_logger`` helper lazily creates the log and writes the header on
    # first use so downstream components can safely read from it during startup.
    get_trade_logger()

    state = BotState()
    cfg = TradingConfig.from_env()

    # Carry through a pre-resolved max position size if available on Settings.
    S = get_settings()
    mps = getattr(S, "max_position_size", None)
    if mps is not None:
        try:
            object.__setattr__(cfg, "max_position_size", float(mps))
        except Exception:
            try:
                setattr(cfg, "max_position_size", float(mps))
            except Exception:  # pragma: no cover - defensive
                pass

    runtime = build_runtime(cfg)

    lazy_ctx = get_ctx()
    if hasattr(state, "ctx") and state.ctx is None:
        state.ctx = lazy_ctx
    runtime = enhance_runtime_with_context(runtime, lazy_ctx)

    missing = [k for k in REQUIRED_PARAM_DEFAULTS if k not in runtime.params]
    if missing:
        logger.error(
            "PARAMS_VALIDATE: missing keys in runtime.params; defaults will be applied",
            extra={"missing": missing},
        )

    run_all_trades_worker(state, runtime)


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


def _fail_fast_env() -> None:
    """Reload and validate mandatory environment variables early."""
    required = (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_DATA_FEED",
        "WEBHOOK_SECRET",
        "CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
    )
    try:
        loaded = reload_env()
        validate_required_env(required)
        snapshot = {k: get_env(k, "") or "" for k in required}
        _, _, base_url = _resolve_alpaca_env()
        if not base_url:
            raise RuntimeError("Missing required environment variable: ALPACA_API_URL or ALPACA_BASE_URL")
        snapshot["ALPACA_API_URL"] = base_url
    except RuntimeError as e:
        logger.critical("ENV_VALIDATION_FAILED", extra={"error": str(e)})
        raise SystemExit(1) from e
    logger.info(
        "ENV_CONFIG_LOADED",
        extra={"dotenv_path": loaded, **redact_env(snapshot, drop=True)},
    )


def _validate_runtime_config(cfg, tcfg) -> None:
    """Fail-fast runtime config checks."""
    errors = []
    mode = getattr(cfg, "trading_mode", "balanced")
    if mode not in {"aggressive", "balanced", "conservative"}:
        errors.append(f"TRADING_MODE invalid: {mode}")
    cap = _as_float(getattr(tcfg, "capital_cap", 0.0), 0.0)
    risk = _as_float(getattr(tcfg, "dollar_risk_limit", 0.0), 0.0)
    if not 0.0 < cap <= 1.0:
        errors.append(f"CAPITAL_CAP out of range: {cap}")
    if not 0.0 < risk <= 1.0:
        errors.append(f"DOLLAR_RISK_LIMIT out of range: {risk}")
    prev_eq = _CACHE.equity
    eq = _get_equity_from_alpaca(cfg)
    targets = (cfg,) if cfg is tcfg else (cfg, tcfg)
    if eq > 0:
        for obj in targets:
            try:
                setattr(obj, "equity", eq)
            except Exception:
                try:
                    object.__setattr__(obj, "equity", eq)
                except Exception:  # pragma: no cover - defensive
                    pass
    else:
        logger.warning("ACCOUNT_EQUITY_MISSING", extra={"equity": eq})
        for obj in targets:
            try:
                setattr(obj, "equity", None)
            except Exception:
                try:
                    object.__setattr__(obj, "equity", None)
                except Exception:  # pragma: no cover - defensive
                    pass
    try:
        force = (_CACHE.value is None) or (eq != prev_eq)
        # Use full Settings for equity/credentials; mode is read from Settings
        resolved, _meta = resolve_max_position_size(cfg, tcfg, force_refresh=force)
        if hasattr(tcfg, "max_position_size"):
            tcfg.max_position_size = float(resolved)
        else:
            os.environ["AI_TRADING_MAX_POSITION_SIZE"] = str(float(resolved))
    except ValueError as e:
        errors.append(str(e))
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
    from ai_trading.logging import validate_logging_setup

    validation_result = validate_logging_setup()
    if not validation_result["validation_passed"]:
        logger.error("Logging validation failed: %s", validation_result["issues"])
    logger.info("Application startup - logging configured once")
    try:
        try:
            reload_env()
        except Exception as exc:  # noqa: BLE001
            logger.critical("ENV_LOAD_FAILED", extra={"error": str(exc)})
            return 1
        config = get_settings()
        if config is None:
            logger.critical("SETTINGS_UNAVAILABLE")
            return 1
        validate_environment()
        memory_optimizer = get_memory_optimizer()
        if memory_optimizer:
            memory_optimizer.enable_low_memory_mode()
            logger.info("Memory optimization enabled")
        logger.info("Bot startup complete - entering main loop")
        preflight_import_health()
        from ai_trading.core.bot_engine import get_trade_logger
        get_trade_logger()
        run_cycle()
        return 0
    except (ValueError, TypeError, RuntimeError) as e:
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
    except Exception:  # noqa: BLE001
        logger.error("Failed to start API", exc_info=True)
        api_error.set()


def _init_http_session(cfg, retries: int = 3, delay: float = 1.0) -> bool:
    """Initialize the global HTTP client session with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            connect_timeout = clamp_request_timeout(
                float(getattr(cfg, "http_connect_timeout", 5.0))
            )
            read_timeout = clamp_request_timeout(
                float(getattr(cfg, "http_read_timeout", 10.0))
            )
            session = build_retrying_session(
                pool_maxsize=int(getattr(cfg, "http_pool_maxsize", 32)),
                total_retries=int(getattr(cfg, "http_total_retries", 3)),
                backoff_factor=float(getattr(cfg, "http_backoff_factor", 0.3)),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
            # Apply host-specific retry profile for Alpaca if configured
            try:
                from urllib.parse import urlparse as _urlparse

                host = _urlparse(str(getattr(cfg, "alpaca_base_url", ""))).netloc
                if host:
                    # ENV override pattern: HTTP_RETRIES_<host> (dots → underscores), HTTP_BACKOFF_<host>
                    key = host.replace(".", "_")
                    import os as _os

                    _retries = int(_os.getenv(f"HTTP_RETRIES_{key}", "2"))
                    _bof = float(_os.getenv(f"HTTP_BACKOFF_{key}", "0.2"))
                    mount_host_retry_profile(session, host, total_retries=_retries, backoff_factor=_bof, pool_maxsize=int(getattr(cfg, "http_pool_maxsize", 32)))
            except Exception:
                pass
            set_global_session(session)
            logger.info(
                "REQUESTS_POOL_STATS",
                extra={
                    "transport": "requests",
                    "pool_maxsize": getattr(cfg, "http_pool_maxsize", 32),
                    "retries": getattr(cfg, "http_total_retries", 3),
                    "backoff_factor": getattr(cfg, "http_backoff_factor", 0.3),
                    "connect_timeout": connect_timeout,
                    "read_timeout": read_timeout,
                },
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "HTTP_SESSION_INIT_FAILED",
                extra={"attempt": attempt, "error": str(exc)},
            )
            if attempt < retries:
                _interruptible_sleep(delay)
    logger.critical("HTTP session initialization failed after %s attempts", retries)
    return False


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
    _fail_fast_env()
    args = parse_cli(argv)
    global config
    config = S = get_settings()
    try:
        if not _is_market_open_base():
            now = datetime.now(ZoneInfo("America/New_York"))
            logger.warning(
                "STARTUP_OUTSIDE_MARKET_HOURS",
                extra={"now": now.isoformat()},
            )
    except Exception:
        logger.debug("MARKET_OPEN_CHECK_FAILED", exc_info=True)
    # Align Settings.capital_cap with plain env when provided to avoid prefix alias gaps
    _cap_env = os.getenv("CAPITAL_CAP")
    if _cap_env:
        try:
            _cap_val = float(_cap_env)
            try:
                setattr(S, "capital_cap", _cap_val)
            except Exception:
                try:
                    object.__setattr__(S, "capital_cap", _cap_val)
                except Exception:
                    pass
            try:
                setattr(config, "capital_cap", _cap_val)
            except Exception:
                try:
                    object.__setattr__(config, "capital_cap", _cap_val)
                except Exception:
                    pass
        except Exception:
            pass
    if config is None:
        logger.critical(
            "SETTINGS_UNAVAILABLE",  # AI-AGENT-REF: clearer startup failure
            extra={
                "hint": "ensure configuration is accessible or run ai_trading.config.management.reload_env",
            },
        )
        raise SystemExit(1)
    logger.info(
        "DATA_CONFIG feed=%s adjustment=%s timeframe=1Day/1Min provider=alpaca", S.alpaca_data_feed, S.alpaca_adjustment
    )
    # Metrics for cycle timing and budget overruns (labels are no-op when metrics unavailable)
    # Labeled stage timings: fetch/compute/execute
    _cycle_stage_seconds = get_histogram("cycle_stage_seconds", "Cycle stage duration seconds", ["stage"])  # type: ignore[arg-type]
    _cycle_budget_over_total = get_counter("cycle_budget_over_total", "Budget-over events", ["stage"])  # type: ignore[arg-type]

    try:
        _validate_runtime_config(config, S)
    except ValueError as e:
        logger.critical("RUNTIME_CONFIG_INVALID", extra={"error": str(e)})
        raise
    if not _init_http_session(config):
        return
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
    from ai_trading.core.bot_engine import get_trade_logger
    get_trade_logger()
    try:
        run_cycle()
    except (TypeError, ValueError) as e:
        logger.critical(
            "Warm-up run_cycle failed during trading initialization; shutting down",
            exc_info=e,
        )
        logging.shutdown()
        raise SystemExit(1) from e
    except SystemExit as e:
        logger.error(
            "Warm-up run_cycle triggered SystemExit; shutting down",
            exc_info=e,
        )
        logging.shutdown()
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception(
            "Warm-up run_cycle failed unexpectedly; shutting down",
            exc_info=e,
        )
        logging.shutdown()
        raise SystemExit(1) from e
    logger.info("Warm-up run_cycle completed")
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
    try:
        closed_interval = int(getattr(S, "interval_when_closed", 300))
    except Exception:
        closed_interval = 300
    seed = get_seed_int()
    logger.info("Runtime defaults resolved", extra={"iterations": iterations, "interval": interval, "seed": seed})
    count = 0
    # Track HTTP profile to reduce retries when market is closed
    _http_closed_profile = None  # None=unknown, True=closed, False=open
    memory_check_interval = 10
    try:
        while iterations <= 0 or count < iterations:
            try:
                closed = not _is_market_open_base()
            except Exception:
                closed = False
            # Toggle HTTP session profile only when state changes
            if _http_closed_profile is None or closed != _http_closed_profile:
                try:
                    if closed:
                        connect_timeout = clamp_request_timeout(float(getattr(S, "http_connect_timeout_closed", getattr(S, "http_connect_timeout", 5.0)) or getattr(S, "http_connect_timeout", 5.0)))
                        read_timeout = clamp_request_timeout(float(getattr(S, "http_read_timeout_closed", getattr(S, "http_read_timeout", 10.0)) or getattr(S, "http_read_timeout", 10.0)))
                        # Optional hint for executors to downsize when closed
                        try:
                            _ewc = os.getenv("EXEC_WORKERS_WHEN_CLOSED")
                            if _ewc:
                                os.environ["AI_TRADING_EXEC_WORKERS"] = _ewc
                                os.environ["AI_TRADING_PRED_WORKERS"] = _ewc
                        except Exception:
                            pass
                        session = build_retrying_session(
                            pool_maxsize=int(getattr(S, "http_pool_maxsize", 32)),
                            total_retries=1,
                            backoff_factor=0.1,
                            connect_timeout=connect_timeout,
                            read_timeout=read_timeout,
                        )
                        try:
                            from urllib.parse import urlparse as _urlparse
                            host = _urlparse(str(getattr(S, "alpaca_base_url", ""))).netloc
                            if host:
                                mount_host_retry_profile(session, host, total_retries=1, backoff_factor=0.1, pool_maxsize=int(getattr(S, "http_pool_maxsize", 32)))
                        except Exception:
                            pass
                        set_global_session(session)
                        logger.info(
                            "HTTP_PROFILE_CLOSED",
                            extra={"retries": 1, "backoff_factor": 0.1, "connect_timeout": connect_timeout, "read_timeout": read_timeout},
                        )
                    else:
                        # Restore configured profile
                        connect_timeout = clamp_request_timeout(float(getattr(S, "http_connect_timeout", 5.0)))
                        read_timeout = clamp_request_timeout(float(getattr(S, "http_read_timeout", 10.0)))
                        try:
                            if "EXEC_WORKERS_WHEN_CLOSED" in os.environ:
                                # Unset closed hint; do not remove explicit AI_TRADING_* overrides if user set them
                                os.environ.pop("AI_TRADING_EXEC_WORKERS", None)
                                os.environ.pop("AI_TRADING_PRED_WORKERS", None)
                        except Exception:
                            pass
                        session = build_retrying_session(
                            pool_maxsize=int(getattr(S, "http_pool_maxsize", 32)),
                            total_retries=int(getattr(S, "http_total_retries", 3)),
                            backoff_factor=float(getattr(S, "http_backoff_factor", 0.3)),
                            connect_timeout=connect_timeout,
                            read_timeout=read_timeout,
                        )
                        try:
                            from urllib.parse import urlparse as _urlparse
                            host = _urlparse(str(getattr(S, "alpaca_base_url", ""))).netloc
                            if host:
                                _retries = int(os.getenv(host.replace('.', '_').join(["HTTP_RETRIES_", ""])) or getattr(S, "http_total_retries", 3))
                                _bof = float(os.getenv(host.replace('.', '_').join(["HTTP_BACKOFF_", ""])) or getattr(S, "http_backoff_factor", 0.3))
                                mount_host_retry_profile(session, host, total_retries=int(_retries), backoff_factor=float(_bof), pool_maxsize=int(getattr(S, "http_pool_maxsize", 32)))
                        except Exception:
                            pass
                        set_global_session(session)
                        logger.info(
                            "HTTP_PROFILE_OPEN",
                            extra={
                                "retries": getattr(S, "http_total_retries", 3),
                                "backoff_factor": getattr(S, "http_backoff_factor", 0.3),
                                "connect_timeout": connect_timeout,
                                "read_timeout": read_timeout,
                            },
                        )
                    _http_closed_profile = closed
                except Exception:
                    pass
            raw_fraction = get_env("CYCLE_BUDGET_FRACTION", 0.8)
            try:
                fraction = float(raw_fraction)
            except (TypeError, ValueError):
                fraction = 0.8
            # Dynamic interval: slow down when closed
            effective_interval = int(closed_interval if closed else interval)
            budget = SoftBudget(interval_sec=float(effective_interval), fraction=fraction)
            try:
                if count % memory_check_interval == 0:
                    gc_result = optimize_memory()
                    if gc_result.get("objects_collected", 0) > 100:
                        logger.info(f"Cycle {count}: Garbage collected {gc_result['objects_collected']} objects")
                _t0 = _mono()
                with StageTimer(logger, "CYCLE_FETCH"):
                    pass
                try:
                    _cycle_stage_seconds.labels(stage="fetch").observe(max(0.0, _mono() - _t0))  # type: ignore[call-arg]
                except Exception:
                    pass
                if budget.over():
                    logger.warning("BUDGET_OVER", extra={"stage": "CYCLE_FETCH"})
                    try:
                        _cycle_budget_over_total.labels(stage="fetch").inc()  # type: ignore[call-arg]
                    except Exception:
                        pass
                _t1 = _mono()
                with StageTimer(logger, "CYCLE_COMPUTE"):
                    run_cycle()
                try:
                    _cycle_stage_seconds.labels(stage="compute").observe(max(0.0, _mono() - _t1))  # type: ignore[call-arg]
                except Exception:
                    pass
                if budget.over():
                    logger.warning("BUDGET_OVER", extra={"stage": "CYCLE_COMPUTE"})
                    try:
                        _cycle_budget_over_total.labels(stage="compute").inc()  # type: ignore[call-arg]
                    except Exception:
                        pass
                _t2 = _mono()
                with StageTimer(logger, "CYCLE_EXECUTE"):
                    pass
                try:
                    _cycle_stage_seconds.labels(stage="execute").observe(max(0.0, _mono() - _t2))  # type: ignore[call-arg]
                except Exception:
                    pass
                if budget.over():
                    logger.warning("BUDGET_OVER", extra={"stage": "CYCLE_EXECUTE"})
                    try:
                        _cycle_budget_over_total.labels(stage="execute").inc()  # type: ignore[call-arg]
                    except Exception:
                        pass
            except (ValueError, TypeError):
                logger.exception("run_cycle failed")
            count += 1
            logger.info("CYCLE_TIMING", extra={"elapsed_ms": budget.elapsed_ms(), "within_budget": not budget.over()})
            now_mono = time.monotonic()
            if now_mono - last_health >= max(30, health_tick_seconds):
                logger.info("HEALTH_TICK", extra={"iteration": count, "interval": effective_interval, "closed": closed})
                last_health = now_mono
            try:
                # Resolve mode directly from env to honor MAX_POSITION_MODE without relying on Settings
                _mode_env = os.getenv("MAX_POSITION_MODE") or os.getenv("AI_TRADING_MAX_POSITION_MODE")
                mode_now = str(
                    _mode_env
                    if _mode_env is not None
                    else getattr(S, "max_position_mode", getattr(config, "max_position_mode", "STATIC"))
                ).upper()
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
            _interruptible_sleep(int(max(1, effective_interval)))
            if _SHUTDOWN.is_set():
                logger.info("SERVICE_SHUTDOWN", extra={"reason": "signal"})
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down gracefully")
        return
    # If a finite number of iterations was requested, exit promptly so tests
    # and batch runs do not hang. Production runs use infinite iterations.
    if iterations > 0:
        logger.info("SCHEDULER_COMPLETE", extra={"iterations": count})
        return


if __name__ == "__main__":
    main()
