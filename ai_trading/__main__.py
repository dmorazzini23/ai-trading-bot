import argparse
import logging
import os
import sys
import subprocess
from threading import Lock

from bot_engine import run_all_trades_worker, BotState
from config import Settings as Config
from logger import setup_logging
import utils

from flask import Flask

logger = logging.getLogger(__name__)
_run_lock = Lock()


def run_bot(argv=None, service: bool = False) -> int:
    """Launch the trading bot as a subprocess."""
    argv = argv or sys.argv[1:]
    python = sys.executable
    if not os.path.isfile(python):
        raise RuntimeError(f"Python executable not found: {python}")
    cmd = [python, "-m", "ai_trading.main"] + argv
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


def main() -> int:
    """Entry-point used by ``python -m ai_trading``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot-only", action="store_true")
    args = parser.parse_args()
    if args.bot_only:
        return run_bot()
    return run_flask_app()


if __name__ == "__main__":
    main()
