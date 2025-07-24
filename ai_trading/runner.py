from __future__ import annotations

from threading import Lock

from bot_engine import run_all_trades_worker, BotState
from logger import get_logger

log = get_logger(__name__)

_run_lock = Lock()


def run_cycle() -> None:
    """Execute a single trading cycle if not already running."""
    # AI-AGENT-REF: run cycle in-process to avoid extra interpreter forks
    if not _run_lock.acquire(blocking=False):
        log.info("RUN_ALL_TRADES_SKIPPED_OVERLAP")
        return
    try:
        run_all_trades_worker(BotState(), None)
    finally:
        _run_lock.release()
