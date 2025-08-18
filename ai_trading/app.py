from __future__ import annotations

import logging
import os  # AI-AGENT-REF: environment diagnostics

from flask import Flask, jsonify


def create_app():
    app = Flask(__name__)
    # AI-AGENT-REF: silence Flask development server noise
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/health")
    def health():
        """Lightweight liveness probe with Alpaca diagnostics."""  # AI-AGENT-REF
        # Lazy imports to avoid heavy side effects at module import time
        try:
            from ai_trading.alpaca_api import ALPACA_AVAILABLE as sdk_ok  # type: ignore
        except Exception:
            sdk_ok = False
        try:
            from ai_trading.core.bot_engine import (
                _resolve_alpaca_env,
                trading_client,
            )  # type: ignore

            key, secret, base_url = _resolve_alpaca_env()
            paper = bool(base_url and ("paper" in base_url))
        except Exception:
            trading_client, base_url, paper = None, "", False
        return jsonify(
            status="ok",
            alpaca_sdk_available=bool(sdk_ok),
            alpaca_client_initialized=bool(trading_client is not None),
            base_url=base_url,
            paper=paper,
            shadow_mode=bool(
                getattr(__import__("ai_trading", fromlist=["config"]).config, "SHADOW_MODE", False)
                or os.getenv("SHADOW_MODE", "").lower() in ("true", "1", "yes")
            ),
        )

    return app


if __name__ == "__main__":
    from ai_trading.config.settings import get_settings

    s = get_settings()
    port = int(s.api_port or 9001)  # AI-AGENT-REF: default Flask port fallback
    app = create_app()
    app.logger.info("Starting Flask", extra={"port": port})
    app.run(host="0.0.0.0", port=port)
