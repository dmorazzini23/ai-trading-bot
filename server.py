import hmac
import logging
import os
import re
import signal
import sys
import threading
import traceback
from typing import Any

from pydantic import BaseModel, ValidationError

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request

from alerting import send_slack_alert
from validate_env import settings

# Load .env early for configuration
load_dotenv(dotenv_path=".env", override=True)

import config

# Configure Flask and root logger to integrate with Gunicorn's error logger
def configure_logging(flask_app: Flask) -> None:
    """Configure log handlers using Gunicorn's logger when available."""
    gunicorn_logger = logging.getLogger("gunicorn.error")
    if hasattr(flask_app, "logger"):
        flask_app.logger.handlers = gunicorn_logger.handlers
        flask_app.logger.setLevel(gunicorn_logger.level)
        logging.root.handlers = gunicorn_logger.handlers
        logging.root.setLevel(gunicorn_logger.level)
    else:  # pragma: no cover - test stubs
        logging.basicConfig(level=gunicorn_logger.level)


logger = logging.getLogger(__name__)

_shutdown = threading.Event()


class WebhookPayload(BaseModel):
    """Schema for incoming webhook requests."""

    symbol: str
    action: str

def _handle_shutdown(signum: int, _unused_frame) -> None:
    logger.info("Received signal %s, shutting down", signum)
    _shutdown.set()

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    send_slack_alert(f"🚨 AI Trading Bot Exception:\n```{error_message}```")
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

if not config.WEBHOOK_SECRET:
    logger.error("WEBHOOK_SECRET must be set")
    raise RuntimeError("WEBHOOK_SECRET must be set")

def verify_sig(payload: bytes, signature_header: str, secret: bytes) -> bool:
    """Validate the GitHub webhook signature."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    sig = signature_header.split("=", 1)[1]
    expected = hmac.new(secret, payload, "sha256").hexdigest()
    return hmac.compare_digest(expected, sig)

def create_app(cfg: Any = config) -> Flask:
    """Return a Flask application configured for webhook handling."""
    cfg.validate_env_vars()
    cfg.log_config(cfg.REQUIRED_ENV_VARS)
    flask_app = Flask(__name__)
    secret = cfg.WEBHOOK_SECRET.encode()
    configure_logging(flask_app)

    @flask_app.route("/github-webhook", methods=["POST"])
    def hook():
        load_dotenv(dotenv_path=".env", override=True)
        try:
            data = request.get_json(force=True)
            if not isinstance(data, dict):
                raise TypeError("Payload must be JSON object")
            payload = WebhookPayload.model_validate(data)
        except (TypeError, ValidationError) as exc:
            logger.warning("Invalid webhook payload: %s", exc)
            return jsonify({"error": "Invalid payload"}), 400

        symbol = str(payload.symbol).upper()
        action = str(payload.action).lower()
        if not re.fullmatch(r"[A-Z]{1,5}", symbol):
            return jsonify({"error": "Invalid symbol"}), 400
        if action not in {"buy", "sell"}:
            return jsonify({"error": "Invalid action"}), 400
        sig = request.headers.get("X-Hub-Signature-256", "")
        if not verify_sig(request.data, sig, secret):
            abort(403)
        if request.headers.get("X-GitHub-Event") == "push":
            # Use subprocess with stdout/stderr inherited for logging visibility
            subprocess.Popen(
                [os.path.join(os.path.dirname(__file__), "deploy.sh")],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        logger.info("Received valid webhook push event")
        return jsonify({"status": "ok"})

    @flask_app.route("/health", methods=["GET"])
    def health() -> Any:
        logger.info("Health check called")
        return jsonify(status="ok")

    @flask_app.route("/healthz", methods=["GET"])
    def healthz() -> Any:
        from flask import Response
        return Response("OK", status=200)

    return flask_app

app = create_app()

if __name__ == "__main__":
    flask_port = settings.FLASK_PORT
    os.environ.setdefault("FLASK_PORT", str(flask_port))
    os.environ["WEBHOOK_PORT"] = str(flask_port)
    from gunicorn.app.wsgiapp import run

    sys.argv = [
        "gunicorn",
        "-w",
        "4",
        "-b",
        f"0.0.0.0:{flask_port}",
        "--log-level",
        "info",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "--capture-output",
        "--enable-stdio-inheritance",
        "server:app",
    ]
    try:
        run()
    except Exception as exc:  # TODO: narrow exception type
        logger.exception("Failed to start gunicorn: %s", exc)
        raise
