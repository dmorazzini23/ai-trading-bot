import hmac
import hashlib
import logging
import os
import subprocess
from flask import Flask, abort, jsonify, request
from config import WEBHOOK_SECRET, WEBHOOK_PORT

logger = logging.getLogger(__name__)

if not WEBHOOK_SECRET:
    raise RuntimeError("WEBHOOK_SECRET must be set for webhook authentication")
if not isinstance(WEBHOOK_PORT, int) or WEBHOOK_PORT <= 0:
    raise RuntimeError("WEBHOOK_PORT must be a positive integer")

app = Flask(__name__)
SECRET = WEBHOOK_SECRET.encode()


def verify_sig(data: bytes, signature: str) -> bool:
    try:
        sha_name, sig = signature.split("=", 1)
        if sha_name != "sha256":
            return False
        mac = hmac.new(SECRET, msg=data, digestmod=hashlib.sha256)
        return hmac.compare_digest(mac.hexdigest(), sig)
    except Exception as exc:
        logger.warning("Signature verification failed: %s", exc)
        return False


@app.route("/github-webhook", methods=["POST"])
def hook():
    payload = request.get_json(force=True)
    if not payload or "symbol" not in payload or "action" not in payload:
        return jsonify({"error": "Missing fields"}), 400
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not verify_sig(request.data, sig):
        abort(403)
    if request.headers.get("X-GitHub-Event") == "push":
        subprocess.Popen([os.path.join(os.path.dirname(__file__), "deploy.sh")])
    return jsonify({"status": "ok"})


def start():
    app.run(host="0.0.0.0", port=WEBHOOK_PORT)


if __name__ == "__main__":
    start()
