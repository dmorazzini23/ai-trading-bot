from __future__ import annotations

from threading import Lock, RLock

import time
from ai_trading.logging import get_logger

log = get_logger(__name__)

_run_lock = RLock()  # Use RLock for re-entrant capability
_last_run_time = 0.0
_min_interval = 5.0  # Minimum seconds between runs

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
            log.info("Bot engine components loaded successfully")
        except Exception as e:
            log.error("Failed to load bot engine components: %s", e)
            raise RuntimeError(f"Cannot load bot engine: {e}")
    
    return _bot_engine, _bot_state_class


def run_cycle() -> None:
    """Execute a single trading cycle if not already running."""
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
        
        # Execute the trading cycle
        run_all_trades_worker(BotState(), None)
    except Exception as e:
        log.exception("Trading cycle failed: %s", e)
    finally:
        _run_lock.release()
