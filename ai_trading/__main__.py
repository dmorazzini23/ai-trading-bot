import logging
import time
from threading import Lock
from filelock import FileLock, Timeout

from bot_engine import run_all_trades_worker, BotState
from data_fetcher import DataFetchError
import config

logger = logging.getLogger(__name__)
_run_lock = Lock()


def run_all_trades() -> None:
    """Run trading loop if not already in progress."""
    if not _run_lock.acquire(blocking=False):
        logger.info("RUN_ALL_TRADES_SKIPPED_OVERLAP")
        return
    try:
        run_all_trades_worker(BotState(), None)
    finally:
        _run_lock.release()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    while True:
        try:
            run_all_trades()
        except DataFetchError as exc:
            logger.warning("DATA_SOURCE_EMPTY | %s", exc)
        time.sleep(config.SCHEDULER_SLEEP_SECONDS)


if __name__ == "__main__":
    lock = FileLock("/tmp/ai_trading_scheduler.lock", timeout=0)
    try:
        with lock:
            main()
    except Timeout:
        logger.info("RUN_ALL_TRADES_SKIPPED_OVERLAP")
