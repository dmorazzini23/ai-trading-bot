from __future__ import annotations

from threading import Lock, RLock

import time
from bot_engine import run_all_trades_worker, BotState
from logger import get_logger

log = get_logger(__name__)

_run_lock = RLock()  # Use RLock for re-entrant capability
_last_run_time = 0.0
_min_interval = 5.0  # Minimum seconds between runs



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
        run_all_trades_worker(BotState(), None)
    except Exception as e:
        log.exception("Trading cycle failed: %s", e)
    finally:
        _run_lock.release()
