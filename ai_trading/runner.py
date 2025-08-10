from __future__ import annotations

from threading import RLock

import time
from ai_trading.logging import get_logger

log = get_logger(__name__)

_run_lock = RLock()  # Use RLock for re-entrant capability
_last_run_time = 0.0
_min_interval = 5.0  # Minimum seconds between runs

# Lazy cache used by tests to verify determinism
_LAZY_CACHE = {}

# AI-AGENT-REF: Lazy import variables to defer heavy imports until runtime
_bot_engine = None
_bot_state_class = None


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
            from ai_trading.core.bot_engine import run_all_trades_worker, BotState
            _bot_engine = run_all_trades_worker
            _bot_state_class = BotState
            if _bot_engine and _bot_state_class:
                # Avoid duplicate info spam on repeated lazy loads
                if not getattr(run_cycle, "_engine_loaded_logged", False):
                    log.info("Bot engine components loaded successfully")
                    setattr(run_cycle, "_engine_loaded_logged", True)
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
    except Exception as e:  # ImportError or anything raised during import
        raise RuntimeError(f"Failed to lazy import workers: {e}") from e
    _LAZY_CACHE["workers"] = run_all_trades_worker
    return run_all_trades_worker


def run_cycle() -> None:
    """Execute a single trading cycle with basic re-entrancy protection."""
    global _last_run_time

    current_time = time.time()
    if current_time - _last_run_time < _min_interval:
        log.debug("RUN_CYCLE_SKIPPED_TOO_FREQUENT")
        return

    if not _run_lock.acquire(blocking=False):
        log.info("RUN_ALL_TRADES_SKIPPED_OVERLAP")
        return
    try:
        _last_run_time = current_time
        
        # AI-AGENT-REF: Lazy load bot engine components only when needed
        run_all_trades_worker, BotState = _load_engine()
        
        state = BotState()
        
        # Before invoking engine, give it an opportunity to warm cache exactly once.
        try:
            # Import lazily to avoid import-time penalties if engine not needed
            from ai_trading.core.bot_engine import _maybe_warm_cache
            if hasattr(state, "ctx"):
                _maybe_warm_cache(state.ctx)  # best-effort; ignores if disabled or already warmed
        except Exception as e:
            # Cache warming failed - log warning but continue execution
            logger.warning("Failed to warm cache during state setup: %s", e)
        
        # Execute the trading cycle
        run_all_trades_worker(state, None)
    except Exception as e:
        log.exception("Trading cycle failed: %s", e)
    finally:
        _run_lock.release()
