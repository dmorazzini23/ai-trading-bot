"""Entry point for running the trading bot with graceful shutdown."""

from __future__ import annotations

import signal
import time

from bot import main
import logging
logger = logging.getLogger(__name__)
import requests

_shutdown = False


def _handle_signal(signum: int, _unused_frame) -> None:
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
        except Exception as exc:  # pragma: no cover - safety
            logger.exception("Unexpected error", exc_info=exc)
            raise
        if not _shutdown:
            time.sleep(5)
