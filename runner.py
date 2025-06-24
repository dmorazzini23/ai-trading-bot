"""Entry point for running the trading bot with graceful shutdown."""

"""Simple runner that restarts the trading bot on failures."""

from __future__ import annotations

import logging
import signal
import time
from typing import NoReturn

import requests

from bot_engine import main

logger = logging.getLogger(__name__)

_shutdown = False


def _handle_signal(signum: int, _unused_frame) -> None:
    """Handle termination signals by setting the shutdown flag."""
    global _shutdown
    logger.info("Received signal %s, shutting down", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def _run_forever() -> NoReturn:
    """Run ``bot.main`` in a loop until a shutdown signal is received."""

    while not _shutdown:
        try:
            main()
        except SystemExit as exc:  # graceful exit
            if exc.code == 0:
                break
            logger.error("Bot exited with code %s", exc.code)
        except requests.exceptions.RequestException as exc:
            logger.exception("API request failed", exc_info=exc)
            raise
        except (RuntimeError, ValueError) as exc:
            logger.exception("Unexpected error", exc_info=exc)
            raise

        if not _shutdown:
            time.sleep(5)


if __name__ == "__main__":
    _run_forever()
