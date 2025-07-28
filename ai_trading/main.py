from __future__ import annotations

import logging
import os
import time
from threading import Thread
import threading

from dotenv import load_dotenv
from validate_env import Settings

import ai_trading.app as app
from ai_trading.runner import run_cycle
import utils

config = Settings()

logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """Ensure required environment variables are present."""
    if not config.WEBHOOK_SECRET:
        raise RuntimeError("WEBHOOK_SECRET is required")
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")


def run_bot(*_a, **_k) -> int:
    """Compatibility wrapper to execute one trading cycle."""
    # AI-AGENT-REF: run cycle directly instead of spawning subprocesses
    run_cycle()
    return 0


def run_flask_app(port: int = 5000) -> None:
    """Launch Flask API on an available port."""
    # AI-AGENT-REF: simplified port fallback logic with get_free_port fallback
    max_attempts = 10
    original_port = port

    for attempt in range(max_attempts):
        if not utils.get_pid_on_port(port):
            break
        port += 1
    else:
        # If consecutive ports are all occupied, use get_free_port as fallback
        free_port = utils.get_free_port()
        if free_port is None:
            raise RuntimeError(f"Could not find available port starting from {original_port}")
        port = free_port

    application = app.create_app()
    application.run(host="0.0.0.0", port=port)


def start_api() -> None:
    """Spin up the Flask API server."""
    run_flask_app(int(os.getenv("API_PORT", 9001)))


def main() -> None:
    """Start the API thread and repeatedly run trading cycles."""
    load_dotenv()
    validate_environment()

    # Ensure API is ready before starting trading cycles
    api_ready = threading.Event()

    def start_api_with_signal():
        start_api()
        api_ready.set()

    t = Thread(target=start_api_with_signal, daemon=True)
    t.start()

    # Wait for API to be ready
    api_ready.wait(timeout=10)

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
