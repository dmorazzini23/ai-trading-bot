#!/usr/bin/env python3.12
import argparse
import os
import signal
import subprocess
import sys
import threading
from typing import Any

from alpaca_trade_api.rest import APIError  # noqa: F401

from dotenv import load_dotenv
from flask import Flask, jsonify
from logger import setup_logging

import config
from alerting import send_slack_alert


def create_flask_app() -> Flask:
    """Return a minimal Flask application with health endpoints."""
    app = Flask(__name__)

    @app.route("/health")
    def health():
        app.logger.info("Health check called")
        return jsonify(status="ok")

    @app.route("/healthz")
    def healthz():
        from flask import Response
        return Response("OK", status=200)

    return app


def run_flask_app(port: int) -> None:
    """Start the Flask application on ``port``."""
    app = create_flask_app()
    app.run(host="0.0.0.0", port=port)


def run_bot(venv_path: str, bot_script: str) -> int:
    """Execute ``bot_script`` using the Python from ``venv_path``."""
    python_executable = os.path.join(venv_path, "bin", "python3.12")
    if not os.path.isfile(python_executable):
        raise RuntimeError(f"Python executable not found at {python_executable}")

    process = subprocess.Popen(
        [python_executable, "-u", bot_script],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ.copy(),
    )
    return process.wait()


def validate_environment() -> None:
    """Ensure mandatory environment variables are present."""
    if not config.WEBHOOK_SECRET:
        raise RuntimeError("WEBHOOK_SECRET must be set")


def main() -> None:
    """Entry point for running the unified bot or API server."""
    parser = argparse.ArgumentParser(description="Unified AI Trading Bot runner")
    parser.add_argument(
        "--serve-api", action="store_true", help="Run the Flask API server"
    )
    parser.add_argument(
        "--bot-only", action="store_true", help="Run only the trading bot"
    )
    parser.add_argument(
        "--flask-port", type=int, default=9000, help="Port for Flask server"
    )
    parser.add_argument(
        "--venv-path",
        default=os.path.join(os.getcwd(), "venv"),
        help="Path to the Python virtual environment",
    )
    parser.add_argument(
        "--bot-script",
        default=os.path.join(os.getcwd(), "bot_engine.py"),
        help="Path to the bot main script",
    )
    parser.add_argument(
        "--log-file",
        default=os.path.join(os.getcwd(), "logs", "trading_bot.log"),
        help="Path to the log file",
    )
    args = parser.parse_args()

    # ✅ Fix: get configured logger object from setup_logging
    logger = setup_logging(log_file=args.log_file)
    logger.info("Starting AI Trading Bot unified runner")

    load_dotenv(dotenv_path=".env", override=True)
    validate_environment()

    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.serve_api and not args.bot_only:
        from threading import Thread

        flask_thread = Thread(target=run_flask_app, args=(args.flask_port,), daemon=True)
        flask_thread.start()
        logger.info(f"Flask server started on port {args.flask_port}")

        bot_exit_code = run_bot(args.venv_path, args.bot_script)
        logger.info(f"Bot process exited with code {bot_exit_code}")
    else:
        bot_exit_code = run_bot(args.venv_path, args.bot_script)
        logger.info(f"Bot process exited with code {bot_exit_code}")

    logger.info("AI Trading Bot runner shutting down")
    sys.exit(bot_exit_code)


if __name__ == "__main__":
    main()
