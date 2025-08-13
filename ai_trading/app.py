from __future__ import annotations

import logging

from flask import Flask, jsonify
from ai_trading.config.settings import get_settings


def create_app():
    app = Flask(__name__)
    # AI-AGENT-REF: silence Flask development server noise
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/health")
    def health():
        return jsonify(status="ok")

    return app


if __name__ == "__main__":
    s = get_settings()
    port = int(s.api_port or 9001)  # AI-AGENT-REF: default Flask port fallback
    app = create_app()
    app.logger.info("Starting Flask", extra={"port": port})
    app.run(host="0.0.0.0", port=port)
