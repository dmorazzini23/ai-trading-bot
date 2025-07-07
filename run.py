#!/usr/bin/env python3.12
import argparse
import errno
import os
import signal
import subprocess
import sys
import threading
from typing import Any
import warnings

# AI-AGENT-REF: suppress noisy external library warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

from alpaca_trade_api.rest import APIError  # noqa: F401
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import socket

import utils

import bot_engine
import runner

import config
from alerting import send_slack_alert
from logger import setup_logging


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


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

    @app.route('/api/override_cooldown', methods=['POST'])
    def override_cooldown():
        payload = request.get_json() or {}
        symbols = payload.get("symbols", [])
        reset = payload.get("reset", True)
        cleared = []
        for sym in symbols:
            if reset:
                bot_engine.state.trade_cooldowns.pop(sym, None)
                cleared.append(sym)
        return jsonify({"status": "success", "details": f"Cooldowns cleared for {cleared}"})

    @app.route('/api/force_trade', methods=['POST'])
    def force_trade():
        payload = request.get_json() or {}
        mode = payload.get("mode", "balanced")
        symbols = payload.get("symbols")
        try:
            runner.run_all_trades(bot_engine.ctx, mode_override=mode, symbols_override=symbols)
            summary = "triggered"
        except Exception as exc:  # pragma: no cover - safety
            summary = str(exc)
        return jsonify({"status": "success", "summary": summary})

    return app


def run_flask_app(port: int) -> None:
    """Start the Flask application on an available ``port``."""
    app = create_flask_app()

    candidate = port
    if port == 9000 and is_port_in_use(9000):
        app.logger.critical("Port 9000 already in use. Exiting to prevent collision.")
        sys.exit(1)
    if utils.get_pid_on_port(port):
        alt = utils.get_free_port(start=port + 1, end=port + 10)
        if alt is None:
            app.logger.error("No free port available starting at %s", port)
            return
        app.logger.warning("Port %s busy, switching to %s", port, alt)
        candidate = alt
    else:
        app.logger.info("Starting Flask on port %s", candidate)

    try:
        app.run(host="0.0.0.0", port=candidate)
    except OSError as exc:  # AI-AGENT-REF: handle runtime reuse
        if exc.errno == errno.EADDRINUSE:
            pid = utils.get_pid_on_port(candidate)
            hint = f" by PID {pid}" if pid else ""
            app.logger.error("Port already in use on %s%s", candidate, hint)
        else:
            raise


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
    if config.FORCE_TRADES:
        logger.warning(
            "\ud83d\ude80 FORCE_TRADES is ENABLED. This run will ignore normal health halts!"
        )

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

        bot_exit_code = run_bot(args.venv_path, args.bot_script)
        logger.info(f"Bot process exited with code {bot_exit_code}")
    else:
        bot_exit_code = run_bot(args.venv_path, args.bot_script)
        logger.info(f"Bot process exited with code {bot_exit_code}")

    logger.info("AI Trading Bot runner shutting down")
    sys.exit(bot_exit_code)


if __name__ == "__main__":
    main()
