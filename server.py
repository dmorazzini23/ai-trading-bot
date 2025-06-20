import logging
import sys

# Basic config: all INFO+ logs to stderr, consistent formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)

import hmac
import os
import signal
import subprocess
import threading
import traceback
from typing import Any

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request

from alerting import send_slack_alert

load_dotenv(dotenv_path=".env", override=True)

app = Flask(__name__)
logger = logging.getLogger(__name__)
logger.info("Server starting up")

_shutdown = threading.Event()

def _handle_shutdown(signum: int, _unused_frame) -> None:
    logger.info(f"Received signal {signum}, shutting down")
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

import config

if not config.WEBHOOK_SECRET:
    logger.error("WEBHOOK_SECRET must be set")
    raise RuntimeError("WEBHOOK_SECRET must be set")

def verify_sig(payload: bytes, signature_header: str, secret: bytes) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    sig = signature_header.split("=", 1)[1]
    expected = hmac.new(secret, payload, "sha256").hexdigest()
    return hmac.compare_digest(expected, sig)

def create_app(cfg: Any = config) -> Flask:
    secret = cfg.WEBHOOK_SECRET.encode()

    @app.route("/github-webhook", methods=["POST"])
    def hook():
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
        logger.info("Received valid webhook push event")
        return jsonify({"status": "ok"})

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        logging.getLogger(__name__).info("Health endpoint called")
        return jsonify(status="ok")

    @app.route("/healthz", methods=["GET"])
    def healthz() -> Any:
        from flask import Response
        return Response("OK", status=200)

    return app

app = create_app()

if __name__ == "__main__":
    flask_port = int(os.getenv("FLASK_PORT", "9000"))
    os.environ.setdefault("FLASK_PORT", str(flask_port))
    os.environ["WEBHOOK_PORT"] = str(flask_port)
    from gunicorn.app.wsgiapp import run
    sys.argv = [
        "gunicorn",
        "-w", "4",
        "-b", f"0.0.0.0:{flask_port}",
        "--log-level", "info",
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--capture-output",
        "--enable-stdio-inheritance",
        "server:app",
    ]
    run()
