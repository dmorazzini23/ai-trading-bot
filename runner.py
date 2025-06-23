"""Entry point for running the trading bot with graceful shutdown."""

from __future__ import annotations

import signal
import time

import logging
import requests

from bot import main

logger = logging.getLogger(__name__)

_shutdown = False


def _handle_signal(signum: int, _unused_frame) -> None:
    """Handle termination signals and set shutdown flag."""
    global _shutdown
    logger.info("Received signal %s, shutting down", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


if __name__ == "__main__":
    while not _shutdown:
        try:
            main()
        except SystemExit as exc:  # graceful exit
            if exc.code == 0:
                break
            logger.error("Bot exited with code %s", exc.code)
        except requests.exceptions.RequestException as e:
            logger.exception("API request failed", exc_info=e)
            raise
        except Exception as exc:  # TODO: narrow exception type
            logger.exception("Unexpected error", exc_info=exc)
            raise
        if not _shutdown:
            time.sleep(5)
