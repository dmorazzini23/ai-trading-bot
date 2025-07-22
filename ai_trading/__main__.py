import logging
import os
import sys
import time
import subprocess
from threading import Lock

from bot_engine import run_all_trades_worker, BotState
from config import Config
from logger import setup_logging
import utils

from flask import Flask

logger = logging.getLogger(__name__)
_run_lock = Lock()


def run_bot(venv_dir: str, script: str) -> int:
    """Launch the bot script under the given virtualenv."""
    python_exec = os.path.join(venv_dir, "bin", f"python{sys.version_info.major}.{sys.version_info.minor}")
    if not os.path.isfile(python_exec):
        raise RuntimeError(f"Python executable not found at {python_exec}")
    proc = subprocess.Popen([python_exec, script])
    return proc.wait()


def create_flask_app() -> Flask:
    """Factory for the Flask application (used by tests)."""
    app = Flask(__name__)

    @app.route("/health")
    @app.route("/healthz")
    def _health():  # pragma: no cover - simple route
        return {"status": "ok"}

    return app


def run_flask_app(port: int = 5000):
    """Run the Flask app, falling back if the port is in use."""
    app = create_flask_app()
    host = "0.0.0.0"
    if utils.get_pid_on_port(port):
        port = utils.get_free_port()
    app.run(host=host, port=port)


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
    if "--bot-only" in sys.argv:
        project_root = os.path.dirname(os.path.dirname(__file__))
        venv_dir = sys.prefix
        script = os.path.join(project_root, "run.py")
        exit_code = run_bot(venv_dir, script)
        sys.exit(exit_code)
    run_flask_app()


if __name__ == "__main__":
    main()
