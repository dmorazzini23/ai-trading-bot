from __future__ import annotations

import logging
import os
import time
from threading import Thread

from dotenv import load_dotenv

import ai_trading.app as app
from ai_trading.runner import run_cycle
import utils

logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """Ensure required environment variables are present."""
    import config

    if not config.WEBHOOK_SECRET:
        raise RuntimeError("Missing required environment variables: WEBHOOK_SECRET")
    config.validate_environment()


def run_bot(*_a, **_k) -> int:
    """Compatibility wrapper to execute one trading cycle."""
    # AI-AGENT-REF: run cycle directly instead of spawning subprocesses
    run_cycle()
    return 0


def run_flask_app(port: int) -> None:
    """Launch Flask API on an available port."""
    application = app.create_app()
    if utils.get_pid_on_port(port):
        port = utils.get_free_port(port + 1) or (port + 1)
    application.run(host="0.0.0.0", port=port)


def start_api() -> None:
    """Spin up the Flask API server."""
    run_flask_app(int(os.getenv("API_PORT", 9001)))


def main() -> None:
    """Start the API thread and repeatedly run trading cycles."""
    load_dotenv()
    validate_environment()

    t = Thread(target=start_api, daemon=True)
    t.start()

    interval = int(os.getenv("SCHEDULER_SLEEP_SECONDS", 30))
    iterations = int(os.getenv("SCHEDULER_ITERATIONS", 0))  # AI-AGENT-REF: test hook
    count = 0
    while iterations <= 0 or count < iterations:
        try:
            run_cycle()
        except Exception:  # pragma: no cover - log unexpected errors
            logger.exception("run_cycle failed")
        count += 1
        time.sleep(interval)


if __name__ == "__main__":
    main()
