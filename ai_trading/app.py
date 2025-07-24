from __future__ import annotations

from flask import Flask


def create_app() -> Flask:
    """Flask application factory used by the scheduler API."""
    app = Flask(__name__)

    @app.route("/health")
    @app.route("/healthz")
    def _health() -> dict[str, str]:  # pragma: no cover - trivial route
        return {"status": "ok"}

    return app
