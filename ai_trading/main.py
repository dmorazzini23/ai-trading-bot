"""Primary entry module for the ai_trading package."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import subprocess
import threading  # AI-AGENT-REF: needed for API serving thread
from threading import Lock
from pathlib import Path
import signal

from bot_engine import run_all_trades_worker, BotState
import config  # AI-AGENT-REF: allow tests to patch config attributes
from config import Settings as Config


def validate_environment() -> None:
    """Expose config validation with testable hook."""
    if not config.WEBHOOK_SECRET:
        raise RuntimeError("Missing required environment variables: WEBHOOK_SECRET")
    config.validate_environment()
from logger import setup_logging
from dotenv import load_dotenv
import utils
from flask import Flask

logger = logging.getLogger(__name__)
_run_lock = Lock()


def run_bot(python: str, run_py: str, extra_args: list[str] | None = None) -> int:
    """Spawn a trading child process using ``python`` to execute ``run_py``."""
    # AI-AGENT-REF: ensure child process only runs trading loop
    cmd = [python, "-m", "ai_trading.main", "--bot-only"]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd)


def create_flask_app() -> Flask:
    """Factory for the Flask application (used by tests)."""
    app = Flask(__name__)

    @app.route("/health")
    @app.route("/healthz")
    def _health():  # pragma: no cover - simple route
        return {"status": "ok"}

    return app


def run_flask_app(port: int | None = None):
    """Run the Flask server on ``port`` falling back when in use."""
    port = port or int(os.getenv("PORT", 8000))
    if utils.get_pid_on_port(port):
        port = utils.get_free_port()
    app = create_flask_app()
    app.run(host="0.0.0.0", port=port)


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
    """Entry-point used by ``python -m ai_trading``."""
    load_dotenv()
    validate_environment()
    setup_logging()
    project_root = Path(__file__).resolve().parents[1]

    if "--bot-only" in sys.argv:
        exit_code = run_bot(sys.prefix, str(project_root / "runner.py"))
        sys.exit(exit_code)
        return

    if "--serve-api" in sys.argv:
        port = int(os.getenv("FLASK_PORT", 8000))
        stop_event = threading.Event()

        def _handler(_s, _f) -> None:
            stop_event.set()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handler)
        thread = threading.Thread(target=run_flask_app, args=(port,), daemon=True)
        thread.start()
        # spawn trading child in --bot-only mode
        run_bot(sys.prefix, str(project_root / "run.py"))
        # now block parent until a shutdown signal arrives
        stop_event.wait()

    run_flask_app()


__all__ = [
    "run_bot",
    "create_flask_app",
    "run_flask_app",
    "run_all_trades",
    "main",
]


if __name__ == "__main__":
    main()
