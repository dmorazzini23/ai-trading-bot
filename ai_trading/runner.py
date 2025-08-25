from __future__ import annotations

# ruff: noqa
import os
import time as _time
from threading import RLock
import warnings
import argparse  # AI-AGENT-REF: CLI parser
from typing import Optional

from ai_trading.logging import get_logger

log = get_logger(__name__)
if os.getenv("BOT_SHOW_DEPRECATIONS", "").lower() in {"1", "true", "yes"}:
    warnings.filterwarnings("default", category=DeprecationWarning)
    warnings.warn(
        "runner.py is deprecated",
        DeprecationWarning,
    )  # AI-AGENT-REF: deprecation notice

_run_lock = RLock()  # Use RLock for re-entrant capability
_last_run_time = 0.0
_min_interval = 5.0  # Minimum seconds between runs

# Lazy cache used by tests to verify determinism
_LAZY_CACHE = {}

# AI-AGENT-REF: Lazy import variables to defer heavy imports until runtime
_bot_engine = None
_bot_state_class = None


# AI-AGENT-REF: startup import preflight
def _preflight_import_health() -> None:
    import importlib
    import os

    if os.environ.get("IMPORT_PREFLIGHT_DISABLED", "").lower() in {"1", "true"}:
        return

    core_modules = [
        "ai_trading.core.bot_engine",
        "ai_trading.risk.engine",
        "ai_trading.rl_trading",
        "ai_trading.telemetry.metrics_logger",
    ]
    for mod in core_modules:
        try:
            importlib.import_module(mod)
        # noqa: BLE001 TODO: narrow exception
        # noqa: BLE001 TODO: narrow exception
        except Exception as exc:  # pragma: no cover - surface import issues
            log.error(
                "IMPORT_PREFLIGHT_FAILED",
                extra={"module_name": mod, "error": repr(exc)},  # AI-AGENT-REF: avoid reserved key
            )
            if os.environ.get("FAIL_FAST_IMPORTS", "").lower() in {"1", "true"}:
                raise SystemExit(1)
    log.info("IMPORT_PREFLIGHT_OK")


def _load_engine():
    """
    Lazy loader for bot engine components to avoid import-time side effects.

    This function defers importing heavy modules until they are actually needed,
    preventing import-time crashes due to missing environment variables or
    circular dependencies.
    """
    global _bot_engine, _bot_state_class

    if _bot_engine is None or _bot_state_class is None:
        try:
            # Import only when needed to avoid import-time validation issues
            from ai_trading.core.bot_engine import BotState, run_all_trades_worker

            _bot_engine = run_all_trades_worker
            _bot_state_class = BotState
            if _bot_engine and _bot_state_class:
                # Use emit-once for startup banners
                from ai_trading.core.bot_engine import _emit_once
                import logging
                _emit_once(log, "engine_components_loaded", logging.INFO, "Bot engine components loaded successfully")
        # noqa: BLE001 TODO: narrow exception
        # noqa: BLE001 TODO: narrow exception
        except Exception as e:
            log.error("Failed to load bot engine components: %s", e)
            raise RuntimeError(f"Cannot load bot engine: {e}")

    return _bot_engine, _bot_state_class


def lazy_load_workers():
    """
    Lazy-imported accessor with deterministic caching.
    First call imports and caches; subsequent calls return the exact same object.
    On import failure, raise RuntimeError (test expectation).
    """
    if "workers" in _LAZY_CACHE:
        return _LAZY_CACHE["workers"]
    try:
        from ai_trading.workers import run_all_trades_worker  # patched in tests
    # noqa: BLE001 TODO: narrow exception
    # noqa: BLE001 TODO: narrow exception
    except Exception as e:  # ImportError or anything raised during import
        raise RuntimeError(f"Failed to lazy import workers: {e}") from e
    _LAZY_CACHE["workers"] = run_all_trades_worker
    return run_all_trades_worker


def run_cycle() -> None:
    """Execute a single trading cycle with basic re-entrancy protection."""
    global _last_run_time

    current_time = _time.time()
    if current_time - _last_run_time < _min_interval:
        log.debug("RUN_CYCLE_SKIPPED_TOO_FREQUENT")
        return

    if not _run_lock.acquire(blocking=False):
        log.info("RUN_ALL_TRADES_SKIPPED_OVERLAP")
        return
    try:
        _last_run_time = current_time

        # AI-AGENT-REF: Lazy load bot engine components only when needed
        try:
            run_all_trades_worker, BotState = _load_engine()
        except RuntimeError as e:
            log.error("Trading cycle failed: %s", e)
            # Backoff a touch to avoid immediate duplicate startup spam
            _time.sleep(1.0)
            raise

        state = BotState()

        # Before invoking engine, give it an opportunity to warm cache exactly once.
        try:
            # Import lazily to avoid import-time penalties if engine not needed
            from ai_trading.core.bot_engine import _maybe_warm_cache

            if hasattr(state, "ctx"):
                _maybe_warm_cache(
                    state.ctx
                )  # best-effort; ignores if disabled or already warmed
        # noqa: BLE001 TODO: narrow exception
        # noqa: BLE001 TODO: narrow exception
        except Exception as e:
            # Cache warming failed - log warning but continue execution
            log.warning("Failed to warm cache during state setup: %s", e)  # AI-AGENT-REF: use module logger

        # Get runtime context and ensure it has proper parameter hydration
        from ai_trading.core.bot_engine import get_ctx
        from ai_trading.core.runtime import build_runtime, REQUIRED_PARAM_DEFAULTS
        from ai_trading.config.management import TradingConfig
        from ai_trading.data_fetcher import DataFetchError
        
        lazy_ctx = get_ctx()
        
        # Build a proper runtime with guaranteed parameter hydration
        cfg = TradingConfig.from_env()  # Load config from environment
        runtime = build_runtime(cfg)
        
        # One-time validation & log as specified in problem statement
        missing = [k for k in REQUIRED_PARAM_DEFAULTS if k not in runtime.params]
        if missing:
            log.error(
                "PARAMS_VALIDATE: missing keys in runtime.params; defaults will be applied",
                extra={"missing": missing}
            )

        log.info(
            "PARAMS_EFFECTIVE",
            extra={
                "CAPITAL_CAP": runtime.params["CAPITAL_CAP"],
                "DOLLAR_RISK_LIMIT": runtime.params["DOLLAR_RISK_LIMIT"],
                "MAX_POSITION_SIZE": runtime.params["MAX_POSITION_SIZE"],
            },
        )
        
        # Enhance runtime with lazy context attributes after initialization
        from ai_trading.core.runtime import enhance_runtime_with_context
        try:
            runtime = enhance_runtime_with_context(runtime, lazy_ctx)
        except DataFetchError as e:
            log.warning("DATA_FETCHER_INIT_FAILED", extra={"detail": str(e)})
            return
        
        # Execute the trading cycle
        run_all_trades_worker(state, runtime)
    # noqa: BLE001 TODO: narrow exception
    # noqa: BLE001 TODO: narrow exception
    except Exception as e:
        log.exception("Trading cycle failed: %s", e)
    finally:
        _run_lock.release()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai_trading.runner",
        description="Lightweight CLI wrapper around run_cycle() for manual pokes and smoke-tests.",
        add_help=True,
    )
    parser.add_argument("-n", "--iterations", type=int, default=1, help="How many trade cycles to run (default: 1)")
    parser.add_argument("-i", "--interval", type=float, default=0.0, help="Seconds to sleep between iterations (default: 0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    return parser


def main() -> int:  # AI-AGENT-REF: argparse-based entrypoint
    """CLI entrypoint used by ``python -m ai_trading.runner``."""
    parser = _build_parser()
    args = parser.parse_args()
    for i in range(args.iterations):
        run_cycle()
        if args.interval and i < args.iterations - 1:
            _time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    _preflight_import_health()
    raise SystemExit(main())
