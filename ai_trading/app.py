from __future__ import annotations

import logging

from flask import Flask, jsonify


# AI-AGENT-REF: expose a sensible default port when running the API directly
DEFAULT_PORT = 9001


def create_app():
    app = Flask(__name__)
    # AI-AGENT-REF: silence Flask development server noise
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/health")
    def health():
        return jsonify(status="ok")

    return app


if __name__ == "__main__":
    from os import getenv
    from dotenv import load_dotenv
    from ai_trading.config import get_settings

    load_dotenv()
    settings = get_settings()
    port = getattr(settings, "api_port", None) or int(getenv("PORT", DEFAULT_PORT))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)
