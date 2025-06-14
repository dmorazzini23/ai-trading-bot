import os
import hmac
import hashlib
import subprocess
import logging
from flask import Flask
from flask import abort
from flask import jsonify
from flask import request
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
from config import WEBHOOK_SECRET, WEBHOOK_PORT

if not WEBHOOK_SECRET:
    logging.getLogger(__name__).error("WEBHOOK_SECRET must be set")
    raise RuntimeError("WEBHOOK_SECRET must be set")

app = Flask(__name__)
SECRET = WEBHOOK_SECRET.encode()


def verify_sig(data: bytes, signature: str) -> bool:
    try:
        sha_name, sig = signature.split("=", 1)
        if sha_name != "sha256":
            return False
        mac = hmac.new(SECRET, msg=data, digestmod=hashlib.sha256)
        return hmac.compare_digest(mac.hexdigest(), sig)
    except Exception as e:
        logging.getLogger(__name__).error("verify_sig error: %s", e)
        return False


@app.route("/github-webhook", methods=["POST"])
def hook():
    # Refresh environment variables on each webhook event
    load_dotenv(dotenv_path=".env", override=True)
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
    # Reload env vars when starting the Flask server
    load_dotenv(dotenv_path=".env", override=True)
    app.run(host="0.0.0.0", port=WEBHOOK_PORT)


if __name__ == "__main__":
    start()
