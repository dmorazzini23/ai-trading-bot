"""Entry point for running the trading bot with graceful shutdown."""

from __future__ import annotations

import logging
import signal
import time

from bot import main
from logger import logger

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
        except Exception as exc:  # pragma: no cover - safety
            logger.exception("Unhandled exception in bot: %s", exc)
        if not _shutdown:
            time.sleep(5)
