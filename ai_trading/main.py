from __future__ import annotations
import argparse
import copy
import os
import threading
import time
import logging
from threading import Thread
import errno
import socket
import signal
import sys
from datetime import datetime, UTC
from zoneinfo import ZoneInfo
from pathlib import Path
import importlib.util
from typing import Any, Callable, Mapping, Tuple
from types import SimpleNamespace
from ai_trading.env import ensure_dotenv_loaded

# Ensure environment variables are loaded before any logging configuration
ensure_dotenv_loaded()

import ai_trading.logging as _logging
from ai_trading.paths import LOG_DIR, ensure_runtime_paths
from ai_trading.runtime.shutdown import register_signal_handlers, request_stop, should_stop
from ai_trading.data.fetch import DataFetchError, EmptyBarsError
from ai_trading.execution.live_trading import APIError, NonRetryableBrokerError
from ai_trading.execution import timing as execution_timing
from ai_trading.utils.datetime import ensure_datetime


def _resolve_log_file() -> str:
    """Resolve the bot log path from environment or defaults."""

    explicit = os.getenv("BOT_LOG_FILE")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            raise SystemExit("BOT_LOG_FILE must be an absolute path when set")
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    ensure_runtime_paths()
    log_path = (LOG_DIR / "bot.log").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return str(log_path)


LOG_FILE = _resolve_log_file()
# Configure logging with the desired file
# Logging must be initialized once here before importing heavy modules like
# ``ai_trading.core.bot_engine``.
_logging.configure_logging(log_file=LOG_FILE)

# Module logger
logger = _logging.get_logger(__name__)
_AUTH_PREFLIGHT_LOGGED = False
_TEST_ALPACA_CREDS_BACKFILLED = False

_STARTUP_PENDING_RECONCILED = False
_RUNTIME_CACHE_LOCK = threading.Lock()
_STATE_CACHE: Any | None = None
_RUNTIME_CACHE: Any | None = None
_RUNTIME_CFG_SNAPSHOT: dict[str, Any] | None = None


def _http_profile_logging_enabled() -> bool:
    truthy = {"1", "true", "yes", "on"}
    try:
        value = get_env("HTTP_PROFILE_LOG_ENABLED", "0", cast=bool)
    except Exception:
        raw = os.getenv("HTTP_PROFILE_LOG_ENABLED", "").strip().lower()
        return raw in truthy
    if isinstance(value, str):
        return value.strip().lower() in truthy
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except Exception:
        return bool(value)

# Detect Alpaca SDK availability without importing heavy modules
def _safe_find_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except ValueError:
        return None


ALPACA_AVAILABLE = (
    _safe_find_spec("alpaca") is not None
    and _safe_find_spec("alpaca.trading.client") is not None
)
if not ALPACA_AVAILABLE:
    if (
        sys.modules.get("alpaca") is not None
        and sys.modules.get("alpaca.trading.client") is not None
    ):
        ALPACA_AVAILABLE = True

from ai_trading.settings import get_seed_int
from ai_trading.config import get_settings
from ai_trading.utils import get_pid_on_port
from ai_trading.utils.prof import StageTimer, SoftBudget
from ai_trading.utils.time import monotonic_time
from ai_trading.logging.redact import redact as _redact
from ai_trading.env.config_redaction import redact_config_env
from ai_trading.net.http import build_retrying_session, set_global_session, mount_host_retry_profile
from ai_trading.utils.http import clamp_request_timeout
from ai_trading.utils.base import is_market_open as _is_market_open_base, next_market_open
from ai_trading.position_sizing import resolve_max_position_size, _get_equity_from_alpaca, _CACHE

try:  # prefer modern budget context
    from ai_trading.core.budget import set_cycle_budget_context
except Exception:  # pragma: no cover - fallback when budget module unavailable
    from contextlib import nullcontext

    def set_cycle_budget_context(*args, **kwargs):  # type: ignore[override]
        return nullcontext()

try:
    from ai_trading.core.bot_engine import (
        emit_cycle_budget_summary,
        clear_cycle_budget_context,
    )
except Exception:  # pragma: no cover - fallback when bot engine unavailable

    def emit_cycle_budget_summary(*args, **kwargs):
        return None

    def clear_cycle_budget_context(*args, **kwargs):
        return None
from ai_trading.config.management import (
    get_env,
    validate_required_env,
    reload_env,
    _resolve_alpaca_env,
    TradingConfig,
    enforce_alpaca_feed_policy,
    get_trading_config,
)
from ai_trading.metrics import get_histogram, get_counter
from ai_trading.telemetry import runtime_state
from ai_trading.utils.env import alpaca_credential_status


def _config_snapshot(cfg: Any) -> dict[str, Any]:
    """Return a deep-copied snapshot of config values for cache comparison."""

    try:
        return copy.deepcopy(cfg.to_dict())
    except Exception:  # pragma: no cover - defensive fallback
        return {"__repr__": repr(cfg)}


def _resolve_cached_context(
    cfg: Any,
    state_factory: Callable[[], Any],
    runtime_builder: Callable[[Any], Any],
) -> Tuple[Any, Any, bool]:
    """Return cached bot state/runtime or rebuild them when config changes."""

    global _STATE_CACHE, _RUNTIME_CACHE, _RUNTIME_CFG_SNAPSHOT

    snapshot = _config_snapshot(cfg)

    with _RUNTIME_CACHE_LOCK:
        cached_state = _STATE_CACHE
        cached_runtime = _RUNTIME_CACHE
        if (
            cached_state is None
            or cached_runtime is None
            or _RUNTIME_CFG_SNAPSHOT != snapshot
        ):
            state = state_factory()
            runtime = runtime_builder(cfg)
            try:
                runtime.cfg = cfg
            except Exception:  # pragma: no cover - runtime without cfg attribute
                pass
            _STATE_CACHE = state
            _RUNTIME_CACHE = runtime
            _RUNTIME_CFG_SNAPSHOT = snapshot
            reused = False
        else:
            state = cached_state
            runtime = cached_runtime
            try:
                runtime.cfg = cfg
            except Exception:  # pragma: no cover - runtime without cfg attribute
                pass
            reused = True

    return state, runtime, reused


def _reset_warmup_cooldown_timestamp() -> None:
    """Clear the cached state's cooldown after a warm-up cycle."""

    with _RUNTIME_CACHE_LOCK:
        state = _STATE_CACHE
        if state is None:
            return
        try:
            if hasattr(state, "last_run_at"):
                setattr(state, "last_run_at", None)
        except Exception:
            return


def _emit_data_config_log(settings: Any, cfg_obj: Any) -> None:
    """Log the resolved feed/provider configuration after any fallbacks."""

    provider_for_log = getattr(cfg_obj, "data_provider", None) or os.environ.get("DATA_PROVIDER") or "unknown"
    feed_for_log = getattr(settings, "alpaca_data_feed", "")
    adjustment_for_log = getattr(settings, "alpaca_adjustment", "")
    try:
        trading_cfg = get_trading_config()
    except Exception:
        trading_cfg = None
    if trading_cfg is not None:
        provider_candidate = getattr(trading_cfg, "data_provider", None)
        if provider_candidate:
            provider_for_log = provider_candidate
        feed_candidate = getattr(trading_cfg, "alpaca_data_feed", None)
        if feed_candidate:
            feed_for_log = feed_candidate
        adj_candidate = getattr(trading_cfg, "alpaca_adjustment", None)
        if adj_candidate:
            adjustment_for_log = adj_candidate
    provider_normalized = str(provider_for_log or "unknown").strip().lower()
    feed_normalized = str(feed_for_log or "").strip().lower()
    mismatch_detected = False
    if trading_cfg is not None:
        mismatch_detected = bool(getattr(trading_cfg, "alpaca_feed_ignored", False))
    if not mismatch_detected and feed_normalized in {"iex", "sip"}:
        mismatch_detected = provider_normalized not in {"alpaca", "alpaca_iex", "alpaca_sip"}
    note_msg = None
    if mismatch_detected:
        note_msg = "feed only used with alpaca* providers"
    log_message = "DATA_CONFIG feed=%s adjustment=%s timeframe=1Day/1Min provider=%s" % (
        str(feed_for_log or ""),
        str(adjustment_for_log or ""),
        str(provider_for_log or "unknown"),
    )
    if note_msg:
        log_message = f"{log_message} note={note_msg}"
    log_extra = {
        "feed": str(feed_for_log or ""),
        "adjustment": str(adjustment_for_log or ""),
        "provider": str(provider_for_log or "unknown"),
    }
    if note_msg:
        log_extra["note"] = note_msg
    logger.info(log_message, extra=log_extra)


def _is_truthy_env(name: str) -> bool:
    """Return ``True`` when environment variable ``name`` is truthy."""

    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_test_mode() -> bool:
    """Detect whether the process is running under automated tests."""

    def _flag_enabled(flag: str) -> bool:
        raw = os.getenv(flag)
        if raw is None:
            return False
        candidate = raw.strip()
        if not candidate:
            return False
        return candidate.lower() not in {"0", "false", "no", "off"}

    return any(_flag_enabled(flag) for flag in ("PYTEST_RUNNING", "TESTING"))


def preflight_import_health() -> bool:
    """Run best-effort import checks returning ``True`` when all succeed."""

    import importlib

    if _is_truthy_env("IMPORT_PREFLIGHT_DISABLED"):
        logger.info("IMPORT_PREFLIGHT_SKIPPED", extra={"reason": "env_override"})
        return True

    feed_validation_failed = False
    core_modules = [
        "ai_trading.core.bot_engine",
        "ai_trading.risk.engine",
        "ai_trading.rl_trading",
        "ai_trading.telemetry.metrics_logger",
        "alpaca.trading.client",
    ]
    failures: list[dict[str, str]] = []
    for mod in core_modules:
        try:
            importlib.import_module(mod)
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - surface import issues
            info = {
                "module_name": mod,
                "error": repr(exc),
                "exc_type": exc.__class__.__name__,
            }
            failures.append(info)
            logger.error("IMPORT_PREFLIGHT_FAILED", extra=info)

    if failures:
        logger.warning(
            "IMPORT_PREFLIGHT_DEGRADED",
            extra={"failed_modules": [f["module_name"] for f in failures]},
        )
    else:
        logger.info("IMPORT_PREFLIGHT_OK")
        feed_snapshot = enforce_alpaca_feed_policy()
        if feed_snapshot:
            status = feed_snapshot.get("status")
            log_fn = logger.info
            if status == "fallback":
                log_fn = logger.warning
            log_extra = dict(feed_snapshot)
            if log_fn is logger.warning or status == "fallback":
                context_extra: dict[str, Any] = {}
                candidate_cfg = None
                try:
                    candidate_cfg = get_trading_config()
                except Exception:
                    candidate_cfg = None
                settings_obj = None
                try:
                    settings_obj = get_settings()
                except Exception:
                    settings_obj = None
                for attr in ("paper", "alpaca_base_url", "alpaca_has_sip", "alpaca_allow_sip"):
                    value = None
                    if candidate_cfg is not None and hasattr(candidate_cfg, attr):
                        value = getattr(candidate_cfg, attr)
                    if value is None and settings_obj is not None and hasattr(settings_obj, attr):
                        value = getattr(settings_obj, attr)
                    if value is not None:
                        context_extra[attr] = value
                if context_extra:
                    log_extra.update(context_extra)
            log_fn("ALPACA_PROVIDER_PREFLIGHT", extra=log_extra)

    ensure_trade_log_path()
    return not failures and not feed_validation_failed


def should_enforce_strict_import_preflight() -> bool:
    """Return ``True`` when import preflight failures should abort startup."""

    if _is_truthy_env("IMPORT_PREFLIGHT_DISABLED"):
        return False

    if any(
        _is_truthy_env(flag)
        for flag in ("AI_TRADING_SYSTEMD_COMPAT", "SYSTEMD_COMPAT", "SYSTEMD_COMPAT_MODE")
    ):
        logger.info("IMPORT_PREFLIGHT_COMPAT_MODE", extra={"mode": "systemd"})
        return False

    if os.environ.get("PYTEST_CURRENT_TEST"):
        has_key, has_secret = alpaca_credential_status()
        if not (has_key and has_secret):
            logger.info(
                "IMPORT_PREFLIGHT_RELAXED",
                extra={"reason": "pytest_missing_credentials"},
            )
            return False

    return True


def _check_alpaca_sdk() -> None:
    """Ensure the Alpaca SDK is installed before continuing."""

    if ALPACA_AVAILABLE:
        return

    if not should_enforce_strict_import_preflight():
        logger.warning(
            "ALPACA_PY_SKIPPED_UNDER_TEST",
            extra={"reason": "import_preflight_relaxed"},
        )
        return

    logger.error("ALPACA_PY_REQUIRED: pip install alpaca-py (alpaca-trade-api==3.2.0) is required")
    raise SystemExit(1)


def run_cycle() -> None:
    """Execute a single trading cycle using the core bot engine."""

    if should_stop():
        logger.info("CYCLE_STOP_REQUESTED", extra={"stage": "start"})
        return

    from ai_trading.alpaca_api import (
        AlpacaAuthenticationError,
        alpaca_get,
        is_alpaca_service_available,
        _set_alpaca_service_available,
    )

    execution_mode = str(get_env("EXECUTION_MODE", "sim", cast=str) or "sim").lower()
    warmup_mode = os.getenv("AI_TRADING_WARMUP_MODE", "").strip().lower() in {"1", "true", "yes"}
    if execution_mode == "disabled":
        _set_alpaca_service_available(False)
        _log_auth_preflight_failure(
            detail="Execution mode is disabled",
            action="Set AI_TRADING_EXECUTION_MODE to paper or live",
        )
        return

    if execution_mode in {"paper", "live"}:
        has_key, has_secret = alpaca_credential_status()
        if not (has_key and has_secret):
            _set_alpaca_service_available(False)
            _log_auth_preflight_failure(
                detail="Missing Alpaca API credentials",
                action="Verify ALPACA_API_KEY/ALPACA_SECRET_KEY",
            )
            return

    allow_after_hours = bool(get_env("ALLOW_AFTER_HOURS", "0", cast=bool))
    if not allow_after_hours:
        try:
            if not _is_market_open_base():
                logger.info("MARKET_CLOSED_SKIP_CYCLE")
                return
        except Exception:
            logger.debug("MARKET_OPEN_CHECK_FAILED", exc_info=True)

    if not is_alpaca_service_available():
        _log_auth_preflight_failure(
            detail="Alpaca authentication previously marked unavailable",
            action="Verify ALPACA_API_KEY/ALPACA_SECRET_KEY",
        )
        return

    try:
        alpaca_get("/v2/account/configurations", timeout=5)
    except Exception as exc:  # pragma: no cover - defensive
        # Handle stale-module exception identity mismatches during reload-heavy
        # test flows by matching the canonical auth exception name as fallback.
        is_auth_failure = isinstance(exc, AlpacaAuthenticationError) or exc.__class__.__name__ == "AlpacaAuthenticationError"
        if is_auth_failure:
            _set_alpaca_service_available(False)
            _log_auth_preflight_failure(
                detail=str(exc),
                action="Verify ALPACA_API_KEY/ALPACA_SECRET_KEY",
            )
            return
        logger.warning(
            "ALPACA_PREFLIGHT_UNEXPECTED",
            extra={"detail": str(exc), "exc_type": exc.__class__.__name__},
        )

    from ai_trading.core.bot_engine import (
        BotState,
        run_all_trades_worker,
        get_ctx,
        get_trade_logger,
        ensure_alpaca_attached,
        list_open_orders,
        cancel_all_open_orders,
        get_confirmed_pending_orders,
        set_cycle_budget_context,
        emit_cycle_budget_summary,
        clear_cycle_budget_context,
        _safe_mode_blocks_trading,
        _failsoft_mode_active,
    )
    from ai_trading.core.runtime import (
        build_runtime,
        REQUIRED_PARAM_DEFAULTS,
        enhance_runtime_with_context,
    )
    from ai_trading.config.management import TradingConfig
    from ai_trading.config import get_settings
    from ai_trading.data import fetch as data_fetcher_module
    from ai_trading.data import provider_monitor
    from ai_trading.data.provider_monitor import safe_mode_reason

    # Ensure trade log file exists before any trade-log reads occur. The
    # ``get_trade_logger`` helper lazily creates the log and writes the header on
    # first use so downstream components can safely read from it during startup.
    get_trade_logger()

    cfg = TradingConfig.from_env()

    skip_compute_when_disabled = bool(getattr(cfg, "skip_compute_when_provider_disabled", False))
    provider_disabled = False
    provider_reason = "provider_disabled"
    provider_check = getattr(data_fetcher_module, "is_primary_provider_enabled", None)
    if callable(provider_check):
        try:
            provider_disabled = not bool(provider_check())
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "PRIMARY_PROVIDER_STATUS_ERROR",
                extra={"detail": str(exc), "exc_type": exc.__class__.__name__},
            )
            provider_disabled = False
    failsoft_guard = _failsoft_mode_active()
    if warmup_mode:
        warmup_skip_reason: str | None = None
        if provider_monitor.is_safe_mode_active():
            warmup_skip_reason = safe_mode_reason() or "provider_safe_mode"
        elif provider_disabled:
            warmup_skip_reason = provider_reason
        else:
            try:
                provider_state_snapshot = runtime_state.observe_data_provider_state()
            except Exception:
                provider_state_snapshot = {}
            if isinstance(provider_state_snapshot, Mapping):
                status_token = str(provider_state_snapshot.get("status") or "").lower()
                if status_token in {"degraded", "disabled", "down", "offline"}:
                    warmup_skip_reason = status_token or "provider_degraded"
        if warmup_skip_reason:
            logger.warning(
                "WARMUP_DEGRADED_SKIP",
                extra={
                    "reason": warmup_skip_reason,
                    "safe_mode": provider_monitor.is_safe_mode_active(),
                },
            )
            _interruptible_sleep(5.0)
            return
    if provider_disabled and skip_compute_when_disabled and not failsoft_guard and _safe_mode_blocks_trading():
        log_extra = {
            "reason": provider_reason,
            "skip_compute_when_provider_disabled": skip_compute_when_disabled,
            "degraded_feed_mode": str(getattr(cfg, "degraded_feed_mode", "block")),
        }
        logger.info("PRIMARY_PROVIDER_DISABLED_CYCLE_SKIP", extra=log_extra)
        cycle_getter = getattr(data_fetcher_module, "_get_cycle_id", None)
        logged_cycles = getattr(data_fetcher_module, "_SAFE_MODE_LOGGED", None)
        if callable(cycle_getter) and isinstance(logged_cycles, set):
            try:
                logged_cycles.add(str(cycle_getter()))
            except Exception:  # pragma: no cover - best effort dedupe
                pass
        _interruptible_sleep(5.0)
        return

    try:
        logger.info(
            "EXEC_CONFIG_RESOLVED",
            extra={
                "execution_require_realtime_nbbo": bool(getattr(cfg, "execution_require_realtime_nbbo", True)),
                "degraded_feed_mode": str(getattr(cfg, "degraded_feed_mode", "widen")),
                "execution_market_on_degraded": bool(getattr(cfg, "execution_market_on_degraded", False)),
                "min_quote_freshness_ms": int(getattr(cfg, "min_quote_freshness_ms", 1500)),
            },
        )
    except Exception:
        pass

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

    state, runtime, _ = _resolve_cached_context(cfg, BotState, build_runtime)

    lazy_ctx = get_ctx()
    if hasattr(state, "ctx") and getattr(state, "ctx", None) is None:
        state.ctx = lazy_ctx
    runtime = enhance_runtime_with_context(runtime, lazy_ctx)

    if not isinstance(getattr(runtime, "state", None), dict):
        try:
            runtime.state = {}
        except Exception:
            pass

    global _STARTUP_PENDING_RECONCILED
    if not _STARTUP_PENDING_RECONCILED:
        try:
            ensure_alpaca_attached(runtime)
            api = getattr(runtime, "api", None)
            if api is None:
                raise RuntimeError("runtime.api missing")
            open_orders = list_open_orders(api)
            pending_orders = get_confirmed_pending_orders(
                api,
                open_orders,
                require_confirmation=True,
            )
        except Exception:
            logger.debug("STARTUP_PENDING_RECONCILE_SKIPPED", exc_info=True)
        else:
            now_dt = datetime.now(UTC)
            pending_ids: list[str] = []
            pending_ages: list[int | None] = []
            for order in pending_orders:
                status_raw = getattr(order, "status", "")
                status_val = getattr(status_raw, "value", status_raw)
                try:
                    status = str(status_val).lower()
                except Exception:
                    status = ""
                if status not in {"new", "pending_new"}:
                    continue
                pending_ids.append(str(getattr(order, "id", "?")))
                age_s: int | None = None
                for attr in ("updated_at", "submitted_at", "created_at"):
                    raw_ts = getattr(order, attr, None)
                    if not raw_ts:
                        continue
                    try:
                        submitted = ensure_datetime(raw_ts)
                    except Exception:
                        continue
                    age_s = int(max(0.0, (now_dt - submitted).total_seconds()))
                    break
                pending_ages.append(age_s)

            try:
                cleanup_after = float(getattr(cfg, "order_stale_cleanup_interval", 120))
            except (TypeError, ValueError):
                cleanup_after = 120.0
            cleanup_after = max(10.0, min(cleanup_after, 3600.0))
            cleanup_after_int = int(cleanup_after)

            if pending_ids:
                numeric_ages = [age for age in pending_ages if age is not None]
                should_cancel = False
                if numeric_ages:
                    max_age = max(numeric_ages)
                    should_cancel = max_age >= cleanup_after
                else:
                    should_cancel = True
                if should_cancel:
                    try:
                        cancel_all_open_orders(runtime)
                    except Exception as exc:
                        extra = {
                            "canceled_ids": pending_ids[:20],
                            "cleanup_after_s": cleanup_after_int,
                            "detail": str(exc),
                        }
                        if numeric_ages:
                            extra["max_age_s"] = max_age
                        logger.warning(
                            "PENDING_ORDERS_STARTUP_CLEANUP_FAILED",
                            extra=extra,
                            exc_info=True,
                        )
                    else:
                        extra = {
                            "canceled_ids": pending_ids[:20],
                            "cleanup_after_s": cleanup_after_int,
                        }
                        if numeric_ages:
                            extra["max_age_s"] = max_age
                        logger.info(
                            "PENDING_ORDERS_STARTUP_CLEANUP",
                            extra=extra,
                        )
            _STARTUP_PENDING_RECONCILED = True

    missing = [k for k in REQUIRED_PARAM_DEFAULTS if k not in runtime.params]
    if missing:
        logger.error(
            "PARAMS_VALIDATE: missing keys in runtime.params; defaults will be applied",
            extra={"missing": missing},
        )

    try:
        run_all_trades_worker(state, runtime)
    except (EmptyBarsError, DataFetchError, NonRetryableBrokerError, APIError) as exc:
        logger.warning(
            "WARMUP_SYMBOL_ERRORS_TOLERATED",
            extra={"error": str(exc), "exc_type": exc.__class__.__name__},
        )
        return


class _NullOptimizer:
    """No-op memory optimizer placeholder."""

    def __call__(self) -> None:
        return None

    def enable_low_memory_mode(self) -> None:  # pragma: no cover - simple no-op
        return None


_NULL_MEMORY_OPTIMIZER = _NullOptimizer()


def get_memory_optimizer():
    from ai_trading.config import safe_settings

    settings = safe_settings()
    if not bool(getattr(settings, "enable_memory_optimization", False)):
        return _NULL_MEMORY_OPTIMIZER
    try:
        from ai_trading.utils import memory_optimizer
    except Exception:
        return _NULL_MEMORY_OPTIMIZER

    return memory_optimizer


def optimize_memory():
    from ai_trading.config import safe_settings

    settings = safe_settings()
    if not bool(getattr(settings, "enable_memory_optimization", False)):
        return {}
    from ai_trading.utils import memory_optimizer

    return memory_optimizer.report_memory_use()


class PortInUseError(RuntimeError):
    """Raised when the API server cannot bind to the requested port."""

    def __init__(self, port: int, pid: int | None = None, *, message: str | None = None):
        self.port = port
        self.pid = pid
        if message is not None:
            resolved = message
        elif pid is not None:
            resolved = f"port {port} in use by pid {pid}"
        else:
            resolved = f"port {port} in use"
        super().__init__(resolved)


class ExistingApiDetected(PortInUseError):
    """Raised when another healthy ai-trading API instance owns the port."""

    def __init__(self, port: int):
        super().__init__(port, pid=None, message=f"healthy API already on {port}")


def _get_wait_window(settings: Any) -> float:
    """Resolve the API port wait window from multiple settings aliases."""

    for attr in (
        "wait_window",
        "api_port_wait_window",
        "api_port_retry_window",
        "port_retry_window_secs",
        "startup_wait_window",
        "api_port_wait_seconds",
    ):
        value = getattr(settings, attr, None)
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return 0.0

config: Any | None = None
register_signal_handlers()


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
    """Install signal handlers that log and request cooperative shutdown."""

    def _handler(signum, frame):  # pragma: no cover - exercised via unit tests
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        logger.info("SERVICE_SIGNAL", extra={"signal": signame})
        request_stop(f"signal:{signame}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError) as exc:
            logger.warning(
                "SIGNAL_HANDLER_INSTALL_FAILED",
                extra={"signal": sig, "error": str(exc)},
            )


def _fail_fast_env() -> None:
    """Reload and validate mandatory environment variables early."""

    global _TEST_ALPACA_CREDS_BACKFILLED
    _TEST_ALPACA_CREDS_BACKFILLED = False
    test_mode = _is_test_mode()
    risk_default: str | None = None
    backfilled_alpaca: list[str] = []
    alpaca_backfilled_during_failfast = False
    if test_mode:
        defaults = {
            "ALPACA_API_KEY": "test-key",
            "ALPACA_SECRET_KEY": "test-secret",
            "ALPACA_DATA_FEED": "iex",
            "WEBHOOK_SECRET": "test-webhook",
            "CAPITAL_CAP": "0.25",
            "DOLLAR_RISK_LIMIT": "0.05",
            "ALPACA_API_URL": "https://paper-api.alpaca.markets",
            "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        }
        risk_default = defaults.pop("DOLLAR_RISK_LIMIT")
        for key, value in defaults.items():
            if not os.getenv(key):
                os.environ[key] = value
                if key in {"ALPACA_API_KEY", "ALPACA_SECRET_KEY"}:
                    backfilled_alpaca.append(key)
                    alpaca_backfilled_during_failfast = True

    alias_backfilled = False
    alias_risk_limit = os.getenv("DAILY_LOSS_LIMIT")
    canonical_risk_limit = os.getenv("DOLLAR_RISK_LIMIT")
    if (canonical_risk_limit is None or canonical_risk_limit.strip() == "") and (
        alias_risk_limit is not None and alias_risk_limit.strip() != ""
    ):
        os.environ["DOLLAR_RISK_LIMIT"] = alias_risk_limit
        alias_backfilled = True
    elif test_mode and (canonical_risk_limit is None or canonical_risk_limit.strip() == ""):
        if risk_default is not None:
            os.environ["DOLLAR_RISK_LIMIT"] = risk_default

    required = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_DATA_FEED",
        "WEBHOOK_SECRET",
        "CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
    ]
    loaded = reload_env(override=False)
    allow_missing_drawdown = test_mode or _is_truthy_env("RUN_HEALTHCHECK")
    if allow_missing_drawdown and "DOLLAR_RISK_LIMIT" in required:
        required.remove("DOLLAR_RISK_LIMIT")
    required_tuple = tuple(required)
    try:
        trading_cfg = TradingConfig.from_env(
            allow_missing_drawdown=allow_missing_drawdown
        )
    except (RuntimeError, ValueError) as e:
        logger.critical("ENV_VALIDATION_FAILED", extra={"error": str(e)})
        raise SystemExit(1) from e

    credential_warning_logged = False
    try:
        validate_required_env(required_tuple)
    except RuntimeError as exc:
        message = str(exc)
        _, _, tail = message.partition(":")
        missing = tuple(sorted(part.strip() for part in tail.split(",") if part.strip()))
        if not missing:
            logger.critical("ENV_VALIDATION_FAILED", extra={"error": message})
            raise SystemExit(1) from exc
        alpaca_fields = {"ALPACA_API_KEY", "ALPACA_SECRET_KEY"}
        non_alpaca_missing = tuple(name for name in missing if name not in alpaca_fields)
        if non_alpaca_missing:
            logger.critical("ENV_VALIDATION_FAILED", extra={"error": message})
            raise SystemExit(1) from exc
        missing_alpaca = tuple(name for name in missing if name in alpaca_fields)
        if missing_alpaca:
            logger.warning(
                "ALPACA_CREDENTIALS_MISSING",
                extra={"missing": missing_alpaca},
            )
            backfilled_alpaca.clear()
            credential_warning_logged = True

    if backfilled_alpaca and not credential_warning_logged:
        logger.warning(
            "ALPACA_CREDENTIALS_MISSING",
            extra={"missing": tuple(sorted(backfilled_alpaca))},
        )
    if test_mode:
        _TEST_ALPACA_CREDS_BACKFILLED = alpaca_backfilled_during_failfast

    snapshot = {k: get_env(k, "") or "" for k in required_tuple}
    _, _, base_url = _resolve_alpaca_env()
    if not base_url:
        error = "Missing required environment variable: ALPACA_API_URL or ALPACA_BASE_URL"
        logger.critical("ENV_VALIDATION_FAILED", extra={"error": error})
        raise SystemExit(1)
    snapshot["ALPACA_API_URL"] = base_url
    logger.info(
        "ENV_CONFIG_LOADED",
        extra={"dotenv_path": loaded, **redact_config_env(snapshot)},
    )

    if os.getenv("RUN_HEALTHCHECK", "").strip() == "1":
        from ai_trading.settings import get_settings

        settings = get_settings()
        health_port = int(getattr(settings, "healthcheck_port", 0) or 0)
        api_port = int(getattr(settings, "api_port", 0) or 0)
        if health_port == api_port:
            message = (
                "RUN_HEALTHCHECK=1 requires HEALTHCHECK_PORT to differ from API_PORT; "
                f"both are set to {api_port}."
            )
            logger.critical(
                "HEALTHCHECK_PORT_CONFLICT",
                extra={"api_port": api_port, "healthcheck_port": health_port},
            )
            raise SystemExit(message)

    raw_risk_limit = get_env("DOLLAR_RISK_LIMIT")
    cfg_risk_limit = getattr(trading_cfg, "dollar_risk_limit", None)
    alias_raw = get_env("DAILY_LOSS_LIMIT")
    canonical_env_value = os.getenv("DOLLAR_RISK_LIMIT")
    if alias_backfilled:
        logger.warning(
            "DOLLAR_RISK_LIMIT_ALIAS_OVERRIDE",
            extra={
                "env_value": alias_raw,
                "canonical_env_value": canonical_env_value,
                "trading_config_value": cfg_risk_limit,
            },
        )
    elif raw_risk_limit not in (None, "") and cfg_risk_limit is not None:
        mismatch = False
        try:
            raw_risk_as_float = float(raw_risk_limit)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raw_risk_as_float = None
        try:
            cfg_risk_as_float = float(cfg_risk_limit)
        except (TypeError, ValueError):
            cfg_risk_as_float = None

        if raw_risk_as_float is not None and cfg_risk_as_float is not None:
            mismatch = raw_risk_as_float != cfg_risk_as_float
        else:
            mismatch = str(cfg_risk_limit) != str(raw_risk_limit)

        if mismatch:
            logger.warning(
                "DOLLAR_RISK_LIMIT_ALIAS_OVERRIDE",
                extra={
                    "env_value": raw_risk_limit,
                    "trading_config_value": cfg_risk_limit,
                },
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
    raw_eq = _get_equity_from_alpaca(cfg)
    eq: float | None
    if raw_eq is None:
        eq = None
    else:
        try:
            eq = float(raw_eq)
        except (TypeError, ValueError):
            logger.warning("ACCOUNT_EQUITY_INVALID", extra={"equity": raw_eq})
            eq = None
    targets = (cfg,) if cfg is tcfg else (cfg, tcfg)
    resolved_eq = eq if eq is not None and eq > 0 else None
    if resolved_eq is not None:
        for obj in targets:
            try:
                setattr(obj, "equity", resolved_eq)
            except Exception:
                try:
                    object.__setattr__(obj, "equity", resolved_eq)
                except Exception:  # pragma: no cover - defensive
                    pass
    else:
        logger.warning(
            "ACCOUNT_EQUITY_MISSING",
            extra={"equity": eq if eq is not None else raw_eq},
        )
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
    """Sleep in slices while honoring shutdown requests."""

    remaining = max(float(total_seconds), 0.0)
    if remaining <= 0:
        return
    if _is_test_mode():
        time.sleep(remaining)
        return
    step = 0.25
    while remaining > 0 and (not should_stop()):
        slice_seconds = min(step, remaining)
        time.sleep(max(0.0, slice_seconds))
        remaining -= slice_seconds


def validate_environment() -> None:
    """Ensure required environment variables are present and dependencies are available."""
    from ai_trading.config.management import get_env, _resolve_alpaca_env

    global config

    missing: list[str] = []
    key, secret, base_url = _resolve_alpaca_env()
    if not key:
        missing.append("ALPACA_API_KEY")
    if not secret:
        missing.append("ALPACA_SECRET_KEY")
    if not base_url:
        missing.append("ALPACA_API_URL")

    if hasattr(config, "WEBHOOK_SECRET"):
        webhook_secret = getattr(config, "WEBHOOK_SECRET", "")
    else:
        webhook_secret = get_env("WEBHOOK_SECRET", "")
    if not webhook_secret:
        missing.append("WEBHOOK_SECRET")

    for var in ("CAPITAL_CAP", "DOLLAR_RISK_LIMIT"):
        if not get_env(var):
            missing.append(var)

    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    _ = get_settings()
    from ai_trading.paths import CACHE_DIR, DATA_DIR, LOG_DIR, MODELS_DIR, OUTPUT_DIR, ensure_runtime_paths

    ensure_runtime_paths()
    logger.info(
        "RUNTIME_PATHS_READY",
        extra={
            "data": str(DATA_DIR),
            "log": str(LOG_DIR),
            "cache": str(CACHE_DIR),
            "models": str(MODELS_DIR),
            "output": str(OUTPUT_DIR),
        },
    )


_TRADE_LOG_INITIALIZED = False


def ensure_trade_log_path() -> None:
    """Initialize trade log and verify the path is writable."""
    from ai_trading.core.bot_engine import get_trade_logger

    global _TRADE_LOG_INITIALIZED

    if _TRADE_LOG_INITIALIZED:
        return

    tl = get_trade_logger()
    path = Path(tl.path)
    parent = path.parent
    if parent.exists() and not os.access(parent, os.W_OK | os.X_OK):
        logger.warning(
            "TRADE_LOGGER_FALLBACK_ACTIVE",
            extra={"reason": "log_dir_not_writable", "path": str(parent)},
        )
        raise SystemExit(1)
    try:
        parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a"):
            pass
    except OSError as exc:  # pragma: no cover - fail fast on unwritable path
        logger.critical(
            "TRADE_LOG_PATH_UNWRITABLE",
            extra={"path": str(path), "error": str(exc)},
        )
        raise SystemExit(1) from exc
    else:
        logger.info(
            "TRADE_LOG_PATH_READY",
            extra={"path": str(path)},
        )
        _TRADE_LOG_INITIALIZED = True


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
        if callable(memory_optimizer):
            try:
                memory_optimizer()
            except Exception:
                logger.debug("MEMORY_OPTIMIZER_DISABLED", exc_info=True)
        elif memory_optimizer:
            memory_optimizer.enable_low_memory_mode()
            logger.info("Memory optimization enabled")
        logger.info("Bot startup complete - entering main loop")
        preflight_ok = preflight_import_health()
        if not preflight_ok:
            if should_enforce_strict_import_preflight():
                logger.critical("IMPORT_PREFLIGHT_ABORT", extra={"strict": True})
                raise SystemExit(1)
            logger.warning("IMPORT_PREFLIGHT_SOFT_FAIL", extra={"strict": False})
        if not _TRADE_LOG_INITIALIZED:
            ensure_trade_log_path()
        run_cycle()
        _reset_warmup_cooldown_timestamp()
        return 0
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("Bot startup failed: %s", e, exc_info=True)
        return 1


def run_flask_app(
    port: int = 5000,
    ready_signal: threading.Event | None = None,
    **run_kwargs,
) -> None:
    """Launch Flask API, retrying on sequential ports when necessary."""

    from ai_trading import app

    application = app.create_app()

    def _mask_identifier(value: str | None, keep: int = 4) -> str:
        if not value:
            return ""
        prefix = str(value)[: max(keep, 0)]
        if not prefix:
            return "***"
        return f"{prefix}***"

    def _broker_snapshot() -> dict[str, Any]:
        snapshot: dict[str, Any] = {"available": False}
        try:
            from ai_trading.core import bot_engine as _bot_engine
        except Exception:
            logger.debug("BROKER_DIAG_IMPORT_FAILED", exc_info=True)
            return snapshot

        client = getattr(_bot_engine, "trading_client", None)
        if client is None:
            return snapshot

        snapshot["available"] = True

        try:
            orders = _bot_engine.list_open_orders(client)
            if orders is None:
                snapshot["open_orders"] = 0
            else:
                orders_list = list(orders)
                snapshot["open_orders"] = len(orders_list)
        except Exception:
            logger.debug("BROKER_DIAG_OPEN_ORDERS_FAILED", exc_info=True)

        try:
            list_positions = getattr(client, "list_positions", None)
            if callable(list_positions):
                positions = list_positions()
                if positions is None:
                    snapshot["positions"] = 0
                else:
                    positions_list = list(positions)
                    snapshot["positions"] = len(positions_list)
        except Exception:
            logger.debug("BROKER_DIAG_POSITIONS_FAILED", exc_info=True)

        try:
            get_account = getattr(client, "get_account", None)
            if callable(get_account):
                account = get_account()
                if account is not None:
                    acct_payload: dict[str, Any] = {
                        "status": getattr(account, "status", None),
                        "trading_blocked": getattr(account, "trading_blocked", None),
                    }
                    identifier = getattr(account, "id", None) or getattr(
                        account, "account_number", None
                    )
                    if identifier:
                        acct_payload["id_masked"] = _mask_identifier(str(identifier))
                    snapshot["account"] = acct_payload
        except Exception:
            logger.debug("BROKER_DIAG_ACCOUNT_FAILED", exc_info=True)

        return snapshot

    config_obj = getattr(application, "config", None)
    if isinstance(config_obj, dict):
        config_obj.setdefault("broker_snapshot_fn", _broker_snapshot)
    elif hasattr(application, "config"):
        try:
            application.config = {"broker_snapshot_fn": _broker_snapshot}
        except Exception:
            pass
    debug = run_kwargs.pop("debug", False)

    def _port_available(candidate: int) -> bool:
        families = [(socket.AF_INET, ("0.0.0.0", candidate))]
        if socket.has_ipv6:
            families.append((socket.AF_INET6, ("::", candidate)))

        for family, addr in families:
            try:
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    if family == socket.AF_INET6 and hasattr(socket, "IPPROTO_IPV6"):
                        try:
                            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                        except (OSError, AttributeError):
                            pass
                    sock.bind(addr)
            except OSError as bind_exc:
                if bind_exc.errno == errno.EADDRINUSE:
                    return False
        return True

    max_attempts = run_kwargs.pop("max_attempts", 5)
    attempt_port = port
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        pid = get_pid_on_port(attempt_port)
        available = _port_available(attempt_port)

        if not available:
            logger.warning(
                "API_PORT_UNAVAILABLE", extra={"port": attempt_port, "reason": "socket_bound"}
            )
            attempt_port += 1
            continue

        if pid:
            logger.error("API_PORT_OCCUPIED", extra={"port": attempt_port, "pid": pid})
            raise PortInUseError(attempt_port, pid)

        try:
            if ready_signal is not None:
                logger.info(
                    "Flask app created successfully, signaling ready on port %s",
                    attempt_port,
                )
                ready_signal.set()
            logger.info("Starting Flask app on 0.0.0.0:%s", attempt_port)
            application.run(host="0.0.0.0", port=attempt_port, debug=debug, **run_kwargs)
            return
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                logger.warning(
                    "API_PORT_BOUND_DURING_START", extra={"port": attempt_port}
                )
                attempt_port += 1
                continue
            raise

    raise PortInUseError(attempt_port)


def _probe_local_api_health(port: int) -> bool:
    """Return ``True`` when the ai-trading API responds on ``port``."""

    try:
        import http.client as _http
        import json as _json
    except Exception:  # pragma: no cover - stdlib import failures are unexpected
        return False

    conn = None
    resp = None
    try:
        conn = _http.HTTPConnection("127.0.0.1", port, timeout=1.5)
        conn.request("GET", "/healthz")
        resp = conn.getresponse()
        payload = resp.read()  # must consume response before closing
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if resp is None or resp.status != 200:
        return False
    try:
        data = _json.loads(payload.decode("utf-8"))
    except Exception:
        return False
    return bool(data) and data.get("service") == "ai-trading"


def start_api(ready_signal: threading.Event | None = None) -> None:
    """Spin up the Flask API server or raise if the port is unavailable."""

    ensure_dotenv_loaded()
    settings = get_settings()
    port = int(getattr(settings, "api_port", 9001) or 9001)
    wait_window = _get_wait_window(settings)
    if wait_window <= 0:
        wait_window = 1.0
    start_time = time.monotonic()
    truthy_env = {"1", "true", "yes", "on"}
    is_pytest = (
        os.getenv("PYTEST_RUNNING", "").strip().lower() in truthy_env
        or "pytest" in sys.modules
    )

    # ── Aux health server on HEALTHCHECK_PORT (non-blocking, separate Flask app)
    try:
        from ai_trading.health import HealthCheck
        health_port = int(getattr(settings, "healthcheck_port", 8081) or 8081)
        if int(port) != int(health_port):
            ctx = SimpleNamespace(host="0.0.0.0", port=health_port, service="ai-trading")
            hc = HealthCheck(ctx=ctx)
            th = threading.Thread(target=hc.run, name="health-server", daemon=True)
            th.start()
            logger.info("HEALTH_SERVER_STARTED", extra={"port": health_port})
        else:
            logger.info("HEALTH_SERVER_PORT_SHARED", extra={"port": port})
    except Exception as _exc:  # pragma: no cover - defensive
        logger.warning("HEALTH_SERVER_START_FAILED", extra={"error": str(_exc)})

    retry_budget = max(wait_window, 0.0)
    retry_deadline = start_time + retry_budget
    retry_attempts = 0

    while True:
        probe: socket.socket | None = None
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("0.0.0.0", port))
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                if probe is not None:
                    probe.close()
                raise
            now_monotonic = time.monotonic()
            elapsed = now_monotonic - start_time
            pid = get_pid_on_port(port)
            conflict_extra = {"port": port}
            if pid:
                conflict_extra["pid"] = pid
                logger.warning("HEALTHCHECK_PORT_CONFLICT", extra=conflict_extra)
                logger.info(
                    "API_STARTUP_ABORTED",
                    extra={"port": port, "reason": "port_in_use", "pid": pid},
                )
                raise PortInUseError(port, pid) from exc
            elif _probe_local_api_health(port):
                logger.warning("HEALTHCHECK_PORT_CONFLICT", extra=conflict_extra)
                logger.info(
                    "API_STARTUP_ABORTED",
                    extra={"port": port, "reason": "existing_api"},
                )
                raise ExistingApiDetected(port) from exc

            if now_monotonic >= retry_deadline:
                logger.warning("HEALTHCHECK_PORT_CONFLICT", extra=conflict_extra)
                logger.info(
                    "API_STARTUP_ABORTED",
                    extra={"port": port, "reason": "port_timeout"},
                )
                raise PortInUseError(
                    port,
                    message=f"port {port} busy (transient) after {wait_window}s",
                ) from exc

            retry_attempts += 1
            sleep_window = min(0.5, 0.05 * retry_attempts)
            remaining = retry_deadline - now_monotonic
            if remaining <= 0:
                remaining = 0
            time.sleep(min(sleep_window, remaining) if remaining > 0 else 0.05)
            if not is_pytest and should_stop():
                logger.info("API_STARTUP_ABORTED", extra={"reason": "shutdown"})
                if ready_signal is not None:
                    ready_signal.set()
                return
            continue
        else:
            break
        finally:
            if probe is not None:
                try:
                    probe.close()
                except OSError:
                    pass

    if not is_pytest and should_stop():
        logger.info("API_STARTUP_ABORTED", extra={"reason": "shutdown"})
        if ready_signal is not None:
            ready_signal.set()
        return

    run_flask_app(port, ready_signal)
    return


def _assert_singleton_api(settings) -> None:
    """Ensure we are the only ai-trading API instance before trading warm-up."""

    env_label = str(getattr(settings, "env", "")).strip().lower()
    if (
        os.getenv("PYTEST_RUNNING", "").strip().lower() in {"1", "true", "yes"}
        or env_label == "test"
        or "pytest" in sys.modules
    ):
        return
    port = int(getattr(settings, "api_port", 9001) or 9001)
    pid = get_pid_on_port(port)
    if pid:
        logger.critical(
            "API_PORT_OCCUPIED_BEFORE_START",
            extra={"port": port, "pid": pid},
        )
        raise PortInUseError(port, pid)
    if _probe_local_api_health(port):
        logger.critical(
            "API_PORT_HEALTHY_ELSEWHERE",
            extra={"port": port},
        )
        raise ExistingApiDetected(port)


def start_api_with_signal(api_ready: threading.Event, api_error: threading.Event) -> None:
    """Start API server and signal readiness/errors."""
    if should_stop():
        api_ready.set()
        return
    try:
        start_api(api_ready)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to start API", exc_info=True)
        try:
            setattr(api_error, "exception", exc)
        except Exception:
            pass
        api_error.set()


def _init_http_session(cfg, retries: int = 3, delay: float = 1.0) -> bool:
    """Initialize the global HTTP client session with retry logic."""
    for attempt in range(1, retries + 1):
        if should_stop():
            logger.info("HTTP_SESSION_INIT_ABORT", extra={"attempt": attempt})
            return False
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
    _check_alpaca_sdk()
    _fail_fast_env()
    args = parse_cli(argv)
    global config
    config = S = get_settings()
    try:
        runtime_state.update_service_status(status="warming_up", reason="startup")
    except Exception:
        logger.debug("SERVICE_STATUS_WARMUP_INIT_FAILED", exc_info=True)
    # Initialize TradingConfig-backed overrides (import-safe)
    try:
        from ai_trading.core import bot_engine as _be

        _be.initialize_runtime_config()
    except Exception:  # pragma: no cover - defensive
        logger.debug("RUNTIME_CONFIG_INIT_SKIPPED", exc_info=True)
    try:
        _assert_singleton_api(S)
    except PortInUseError as exc:
        logger.critical(
            "API_PORT_CONFLICT_FATAL",
            extra={"port": exc.port, "pid": exc.pid},
        )
        if os.getenv("PYTEST_RUNNING", "").strip():
            logger.warning("API_PORT_CONFLICT_IGNORED_UNDER_TEST")
        else:
            raise SystemExit(errno.EADDRINUSE) from exc
    api_ready = threading.Event()
    api_error = threading.Event()
    api_thread = Thread(target=start_api_with_signal, args=(api_ready, api_error), daemon=True)
    api_thread.start()
    try:
        if api_error.wait(timeout=2):
            raise getattr(api_error, "exception", RuntimeError("API failed to start"))
        if not api_ready.wait(timeout=5):
            if not api_thread.is_alive():
                raise RuntimeError("API thread terminated unexpectedly during startup")
            logger.warning(
                "API_STARTUP_LAGGING",
                extra={"note": "health endpoints may remain warming_up during warm-up"},
            )
    except PortInUseError as exc:
        logger.critical(
            "API_PORT_CONFLICT_FATAL",
            extra={"port": exc.port, "pid": exc.pid},
        )
        if _is_test_mode():
            logger.warning("API_PORT_CONFLICT_IGNORED_UNDER_TEST")
        else:
            raise SystemExit(errno.EADDRINUSE) from exc
    except (RuntimeError, TimeoutError, OSError) as exc:
        logger.error("API_STARTUP_DEGRADED", exc_info=exc)
        api_error.set()
        try:
            runtime_state.update_service_status(status="degraded", reason="api_start_failed")
        except Exception:
            logger.debug("SERVICE_STATUS_API_DEGRADED_UPDATE_FAILED", exc_info=True)
    allow_after_hours = bool(get_env("ALLOW_AFTER_HOURS", "0", cast=bool))
    try:
        if not _is_market_open_base():
            now = datetime.now(ZoneInfo("America/New_York"))
            if allow_after_hours:
                logger.warning(
                    "STARTUP_OUTSIDE_MARKET_HOURS",
                    extra={"now": now.isoformat(), "override": True},
                )
            else:
                try:
                    nxt = next_market_open(now)
                except Exception:
                    logger.error("NEXT_MARKET_OPEN_FAILED", exc_info=True)
                    raise SystemExit(1)
                wait = max((nxt - now).total_seconds(), 0.0)
                cap_raw = getattr(S, "interval_when_closed", 300)
                try:
                    cap = float(cap_raw)
                except (TypeError, ValueError):
                    cap = 300.0
                if cap != cap:  # NaN guard without importing math
                    cap = 300.0
                cap = max(cap, 0.0)
                sleep_for = min(wait, cap) if cap > 0 else 0.0
                logger.warning(
                    "MARKET_CLOSED_SLEEP",
                    extra={
                        "now": now.isoformat(),
                        "next_open": nxt.isoformat(),
                        "sleep_s": int(sleep_for),
                        "sleep_original_s": int(wait),
                        "sleep_cap_s": int(cap),
                    },
                )
                if sleep_for > 0:
                    _interruptible_sleep(sleep_for)
    except Exception:
        logger.debug("MARKET_OPEN_CHECK_FAILED", exc_info=True)
    # Align Settings.capital_cap with plain env when provided to avoid prefix alias gaps
    if not _is_test_mode():
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
    _emit_data_config_log(S, config)
    # Metrics for cycle timing and budget overruns (labels are no-op when metrics unavailable)
    # Labeled stage timings: fetch/compute/execute
    _cycle_stage_seconds = get_histogram("cycle_stage_seconds", "Cycle stage duration seconds", ["stage"])  # type: ignore[arg-type]
    _cycle_budget_over_total = get_counter("cycle_budget_over_total", "Budget-over events", ["stage"])  # type: ignore[arg-type]

    try:
        _validate_runtime_config(config, S)
    except ValueError as e:
        logger.critical("RUNTIME_CONFIG_INVALID", extra={"error": str(e)})
        raise
    if should_stop():
        logger.info("SERVICE_SHUTDOWN", extra={"reason": "preflight-stop"})
        return
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
    ensure_trade_log_path()
    warmup_ok = True
    test_mode = _is_test_mode()
    run_warmup = not test_mode
    if test_mode and not run_warmup:
        run_warmup = args.iterations is not None or _TEST_ALPACA_CREDS_BACKFILLED
    if test_mode and not run_warmup:
        try:
            key, secret, _ = _resolve_alpaca_env()
        except Exception:
            key = os.getenv("ALPACA_API_KEY")
            secret = os.getenv("ALPACA_SECRET_KEY")
        run_warmup = not (key and secret)
    if run_warmup:
        os.environ["AI_TRADING_WARMUP_MODE"] = "1"
        try:
            run_cycle()
        except (TypeError, ValueError) as e:
            logger.critical(
                "Warm-up run_cycle failed during trading initialization; shutting down",
                exc_info=e,
            )
            raise SystemExit(1) from e
        except SystemExit as e:
            logger.error(
                "Warm-up run_cycle triggered SystemExit; shutting down",
                exc_info=e,
            )
            raise
        except (NonRetryableBrokerError, DataFetchError, EmptyBarsError, APIError, ConnectionError, TimeoutError) as e:
            warmup_ok = False
            logger.warning(
                "WARMUP_RECOVERED",
                extra={"error": str(e), "exc_type": e.__class__.__name__},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "Warm-up run_cycle failed unexpectedly; shutting down",
                exc_info=e,
            )
            raise SystemExit(1) from e
        else:
            logger.info("Warm-up run_cycle completed")
        finally:
            os.environ.pop("AI_TRADING_WARMUP_MODE", None)
        if not warmup_ok:
            logger.info("Warm-up run_cycle completed with recovery")
    else:
        os.environ.pop("AI_TRADING_WARMUP_MODE", None)
    _reset_warmup_cooldown_timestamp()
    try:
        if api_error.is_set():
            reason = "api_start_failed"
            if not warmup_ok:
                reason = "warmup_recovered_api_failed"
            runtime_state.update_service_status(status="degraded", reason=reason)
        elif warmup_ok:
            runtime_state.update_service_status(status="ready")
        else:
            runtime_state.update_service_status(status="ready", reason="warmup_recovered")
    except Exception:
        logger.debug("SERVICE_STATUS_READY_UPDATE_FAILED", exc_info=True)
    S = get_settings()
    from ai_trading.utils.device import get_device  # AI-AGENT-REF: guard torch import

    get_device()
    raw_tick = get_env("HEALTH_TICK_SECONDS") or getattr(S, "health_tick_seconds", 300)
    recommended_health_tick = 30
    try:
        health_tick_seconds = int(raw_tick)
    except (ValueError, TypeError):
        health_tick_seconds = int(getattr(S, "health_tick_seconds", 300))
    if health_tick_seconds < recommended_health_tick:
        logger.warning(
            "HEALTH_TICK_INTERVAL_MINIMUM_ENFORCED",
            extra={
                "configured": health_tick_seconds,
                "normalized": recommended_health_tick,
            },
        )
        health_tick_seconds = recommended_health_tick
    health_tick_runtime = health_tick_seconds
    last_health = monotonic_time()
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
                        if _http_profile_logging_enabled():
                            logger.info(
                                "HTTP_PROFILE_CLOSED",
                                extra={
                                    "retries": 1,
                                    "backoff_factor": 0.1,
                                    "connect_timeout": connect_timeout,
                                    "read_timeout": read_timeout,
                                },
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
                        if _http_profile_logging_enabled():
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
            raw_fraction = get_env(
                "CYCLE_COMPUTE_BUDGET",
                get_env("CYCLE_BUDGET_FRACTION", 0.9),
            )
            try:
                fraction = float(raw_fraction)
            except (TypeError, ValueError):
                fraction = 0.9
            # Dynamic interval: slow down when closed
            effective_interval = int(closed_interval if closed else interval)
            budget = None
            try:
                interval_ms = max(0.0, float(effective_interval)) * 1000.0
                fraction_clamped = max(0.0, min(1.0, float(fraction)))
                budget = SoftBudget(int(interval_ms * fraction_clamped))
                execution_timing.reset_cycle()
                try:
                    _t0 = monotonic_time()
                    with StageTimer(logger, "CYCLE_FETCH"):
                        if count % memory_check_interval == 0:
                            gc_result = optimize_memory()
                            if gc_result.get("objects_collected", 0) > 100:
                                logger.info(
                                    "CYCLE_FETCH_GC",
                                    extra={
                                        "cycle": count,
                                        "objects_collected": gc_result["objects_collected"],
                                    },
                                )
                    try:
                        _cycle_stage_seconds.labels(stage="fetch").observe(max(0.0, monotonic_time() - _t0))  # type: ignore[call-arg]
                    except Exception:
                        pass
                    if budget.over_budget():
                        logger.warning("BUDGET_OVER", extra={"stage": "CYCLE_FETCH"})
                        try:
                            _cycle_budget_over_total.labels(stage="fetch").inc()  # type: ignore[call-arg]
                        except Exception:
                            pass
                    _t1 = monotonic_time()
                    if budget is not None:
                        set_cycle_budget_context(
                            budget,
                            interval_s=float(effective_interval),
                            fraction=fraction_clamped,
                        )
                    try:
                        with StageTimer(logger, "CYCLE_COMPUTE"):
                            run_cycle()
                    finally:
                        if budget is not None:
                            emit_cycle_budget_summary(logger)
                            clear_cycle_budget_context()
                    try:
                        _cycle_stage_seconds.labels(stage="compute").observe(max(0.0, monotonic_time() - _t1))  # type: ignore[call-arg]
                    except Exception:
                        pass
                    if budget.over_budget():
                        logger.warning("BUDGET_OVER", extra={"stage": "CYCLE_COMPUTE"})
                        try:
                            _cycle_budget_over_total.labels(stage="compute").inc()  # type: ignore[call-arg]
                        except Exception:
                            pass
                    execute_seconds = execution_timing.cycle_seconds()
                    with StageTimer(
                        logger,
                        "CYCLE_EXECUTE",
                        override_ms=execute_seconds * 1000.0,
                    ):
                        pass
                    try:
                        _cycle_stage_seconds.labels(stage="execute").observe(max(0.0, execute_seconds))  # type: ignore[call-arg]
                    except Exception:
                        pass
                    if budget.over_budget():
                        logger.warning("BUDGET_OVER", extra={"stage": "CYCLE_EXECUTE"})
                        try:
                            _cycle_budget_over_total.labels(stage="execute").inc()  # type: ignore[call-arg]
                        except Exception:
                            pass
                    if budget is not None:
                        try:
                            elapsed_ms = int(max(0.0, budget.elapsed_ms()))
                            budget_ms = int(max(0.0, budget.ms))
                        except Exception:
                            elapsed_ms = None
                            budget_ms = None
                        if elapsed_ms is not None and budget_ms is not None:
                            logger.info(
                                "CYCLE_COMPUTE_BUDGET",
                                extra={
                                    "elapsed_ms": elapsed_ms,
                                    "budget_ms": budget_ms,
                                    "status": "OVER" if budget.over_budget() else "OK",
                                },
                            )
                except (ValueError, TypeError):
                    logger.exception("run_cycle failed")
            except Exception:
                logger.error(
                    "SCHEDULER_RUN_CYCLE_EXCEPTION",
                    extra={"iteration": count},
                    exc_info=True,
                )
                count += 1
                try:
                    backoff_seconds = int(effective_interval)
                except Exception:
                    backoff_seconds = interval
                backoff_seconds = max(1, min(30, backoff_seconds))
                _interruptible_sleep(backoff_seconds)
                _logging.flush_log_throttle_summaries()
                if should_stop():
                    logger.info("SERVICE_SHUTDOWN", extra={"reason": "signal"})
                    break
                continue
            if budget is None:
                continue
            count += 1
            logger.info(
                "CYCLE_TIMING",
                extra={"elapsed_ms": budget.elapsed_ms(), "within_budget": not budget.over_budget()},
            )
            _logging.flush_log_throttle_summaries()
            now_mono = monotonic_time()
            if now_mono - last_health >= health_tick_runtime:
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
            if should_stop():
                logger.info("SERVICE_SHUTDOWN", extra={"reason": "signal"})
                break
    except KeyboardInterrupt:
        request_stop("keyboard-interrupt")
        logger.info("KeyboardInterrupt received — shutting down gracefully")
        return
    # If a finite number of iterations was requested, exit promptly so tests
    # and batch runs do not hang. Production runs use infinite iterations.
    if iterations > 0:
        logger.info("SCHEDULER_COMPLETE", extra={"iterations": count})
        return


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            logger.error("SERVICE_EXIT", extra={"code": code})
        raise
    except BaseException:  # noqa: BLE001
        logger.exception("SERVICE_CRASH")
        sys.exit(1)
    else:
        sys.exit(0)
def _log_auth_preflight_failure(detail: str, action: str) -> None:
    """Emit a single critical log for Alpaca auth preflight failures."""

    global _AUTH_PREFLIGHT_LOGGED
    if _AUTH_PREFLIGHT_LOGGED:
        return
    _AUTH_PREFLIGHT_LOGGED = True
    logger.critical(
        "ALPACA_AUTH_PREFLIGHT_FAILED",
        extra={"detail": detail, "action": action},
    )
    _emit_capture_handler_record(detail, action)


def _emit_capture_handler_record(detail: str, action: str) -> None:
    """Forward the auth failure to pytest log capture handlers when present."""

    handler_refs = getattr(logging, "_handlerList", None)
    if not handler_refs:
        return
    try:
        base_logger = logger.logger  # type: ignore[attr-defined]
    except AttributeError:
        base_logger = logger
    for handler_ref in list(handler_refs):
        handler = handler_ref() if callable(handler_ref) else handler_ref
        if handler is None:
            continue
        if handler.__class__.__name__ != "LogCaptureHandler":
            continue
        record = base_logger.makeRecord(
            base_logger.name,
            logging.CRITICAL,
            __file__,
            _log_auth_preflight_failure.__code__.co_firstlineno,
            "ALPACA_AUTH_PREFLIGHT_FAILED",
            args=(),
            exc_info=None,
        )
        record.detail = detail
        record.action = action
        try:
            handler.emit(record)
        except Exception:  # pragma: no cover - defensive guard for custom handlers
            continue
