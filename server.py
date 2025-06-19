from logger import setup_logging
import hmac
import os
import socket
import subprocess
from typing import Any

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request

# Load .env early so config.WEBHOOK_SECRET is set
load_dotenv(dotenv_path=".env", override=True)

import config
import logging
logger = logging.getLogger(__name__)

setup_logging()

import sys
import traceback
import requests


def send_slack_alert(message: str):
    webhook_url = os.getenv("SLACK_WEBHOOK")
    if webhook_url:
        payload = {"text": message}
        try:
            requests.post(webhook_url, json=payload)
        except Exception as e:
            logging.error(f"Failed to send Slack alert: {e}")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    send_slack_alert(f"🚨 AI Trading Bot Exception:\n```{error_message}```")
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

if not config.WEBHOOK_SECRET:
    logger.error("WEBHOOK_SECRET must be set")
    raise RuntimeError("WEBHOOK_SECRET must be set")

def check_port_available(port: int) -> bool:
    """Return ``True`` if ``port`` can be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def verify_sig(payload: bytes, signature_header: str, secret: bytes) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    sig = signature_header.split("=", 1)[1]
    expected = hmac.new(secret, payload, "sha256").hexdigest()
    return hmac.compare_digest(expected, sig)


def create_app(cfg: Any = config) -> Flask:
    app = Flask(__name__)
    secret = cfg.WEBHOOK_SECRET.encode()

    @app.route("/github-webhook", methods=["POST"])
    def hook():
        # Refresh environment variables on each webhook event
        load_dotenv(dotenv_path=".env", override=True)

        payload = request.get_json(force=True)
        if not payload or "symbol" not in payload or "action" not in payload:
            return jsonify({"error": "Missing fields"}), 400

        sig = request.headers.get("X-Hub-Signature-256", "")
        if not verify_sig(request.data, sig, secret):
            abort(403)

        if request.headers.get("X-GitHub-Event") == "push":
            subprocess.Popen([
                os.path.join(os.path.dirname(__file__), "deploy.sh")
            ])
        return jsonify({"status": "ok"})

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify(status="ok")

    return app


app = create_app()


if __name__ == "__main__":
    # Disable Flask's reloader so no extra watcher process is spawned.
    flask_port = int(os.getenv("FLASK_PORT", "9000"))
    # If the port was not set, ensure the env var is populated for consistency
    os.environ.setdefault("FLASK_PORT", str(flask_port))
    if not check_port_available(flask_port):
        raise RuntimeError(f"Port {flask_port} is already in use")
    os.environ["WEBHOOK_PORT"] = str(flask_port)
    app.run(host="0.0.0.0", port=flask_port, debug=False, use_reloader=False)
