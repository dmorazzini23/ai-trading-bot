from __future__ import annotations

import logging

from flask import Flask, jsonify


def create_app():
    app = Flask(__name__)
    # AI-AGENT-REF: silence Flask development server noise
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/health")
    def health():
        return jsonify(status="ok")

    return app
