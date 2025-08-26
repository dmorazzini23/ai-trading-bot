from __future__ import annotations
import logging
import os
from flask import Flask, jsonify

def create_app():
    app = Flask(__name__)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    @app.route('/health')
    def health():
        """Lightweight liveness probe with Alpaca diagnostics."""
        try:
            try:
                from ai_trading.alpaca_api import ALPACA_AVAILABLE as sdk_ok
            except (KeyError, ValueError, TypeError):
                sdk_ok = False

            try:
                from ai_trading.core.bot_engine import _resolve_alpaca_env, trading_client
                key, secret, base_url = _resolve_alpaca_env()
                paper = bool(base_url and 'paper' in base_url)
            except (KeyError, ValueError, TypeError):
                trading_client, key, secret, base_url, paper = (None, None, None, '', False)

            from ai_trading.config.management import is_shadow_mode

            shadow = is_shadow_mode()

            return jsonify(
                alpaca=dict(
                    sdk_ok=bool(sdk_ok),
                    initialized=bool(trading_client),
                    client_attached=bool(trading_client),
                    has_key=bool(key),
                    has_secret=bool(secret),
                    base_url=base_url,
                    paper=paper,
                    shadow_mode=shadow,
                )
            )
        except Exception as e:  # noqa: BLE001
            return jsonify(ok=False, error=str(e))

    @app.route('/healthz')
    def healthz():
        """Minimal liveness probe."""
        from datetime import UTC, datetime

        return jsonify(
            ok=True,
            ts=datetime.now(UTC).isoformat(),
            service="ai-trading",
        )

    @app.route('/metrics')
    def metrics():
        """Expose Prometheus metrics if available."""
        try:
            from ai_trading.metrics import PROMETHEUS_AVAILABLE, REGISTRY
        except Exception:
            PROMETHEUS_AVAILABLE, REGISTRY = False, None
        if not PROMETHEUS_AVAILABLE:
            return ('metrics unavailable', 501)
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return generate_latest(REGISTRY), 200, {'Content-Type': CONTENT_TYPE_LATEST}

    return app


if __name__ == '__main__':
    if os.getenv('RUN_HEALTHCHECK') == '1':
        from ai_trading.config.management import validate_required_env
        from ai_trading.config.settings import get_settings

        validate_required_env()
        s = get_settings()
        port = int(s.healthcheck_port or 9001)
        app = create_app()
        app.logger.info('Starting Flask', extra={'port': port})
        app.run(host='0.0.0.0', port=port)
