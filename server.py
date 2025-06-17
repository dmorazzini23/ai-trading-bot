import hmac
import logging
import os
import subprocess
from typing import Any

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request

load_dotenv(dotenv_path=".env", override=True)
import config

if not config.WEBHOOK_SECRET:
    logging.getLogger(__name__).error("WEBHOOK_SECRET must be set")
    raise RuntimeError("WEBHOOK_SECRET must be set")



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
            subprocess.Popen([os.path.join(os.path.dirname(__file__), "deploy.sh")])
        return jsonify({"status": "ok"})

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify(status="ok")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", config.WEBHOOK_PORT)), debug=False)
