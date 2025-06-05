#!/usr/bin/env python3
import os
import hmac
import hashlib
import subprocess
from flask import Flask, request, abort

from config import WEBHOOK_SECRET, WEBHOOK_PORT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
SECRET = WEBHOOK_SECRET.encode()


def verify_sig(data: bytes, signature: str) -> bool:
    try:
        sha_name, sig = signature.split("=", 1)
        if sha_name != "sha256":
            return False
        mac = hmac.new(SECRET, msg=data, digestmod=hashlib.sha256)
        return hmac.compare_digest(mac.hexdigest(), sig)
    except Exception:
        return False


@app.route("/github-webhook", methods=["POST"])
def hook():
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not verify_sig(request.data, sig):
        abort(403)
    if request.headers.get("X-GitHub-Event") == "push":
        subprocess.Popen(["/root/ai-trading-bot/deploy.sh"])
    return "OK", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=WEBHOOK_PORT)
