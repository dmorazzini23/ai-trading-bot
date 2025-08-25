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
            from ai_trading.alpaca_api import ALPACA_AVAILABLE as sdk_ok
        except (KeyError, ValueError, TypeError):
            sdk_ok = False
        try:
            from ai_trading.core.bot_engine import _resolve_alpaca_env, trading_client
            key, secret, base_url = _resolve_alpaca_env()
            paper = bool(base_url and 'paper' in base_url)
        except (KeyError, ValueError, TypeError):
            trading_client, key, secret, base_url, paper = (None, None, None, '', False)
        shadow = bool(getattr(__import__('ai_trading', fromlist=['config']).config, 'SHADOW_MODE', False) or os.getenv('SHADOW_MODE', '').lower() in ('true', '1', 'yes'))
        return jsonify(alpaca=dict(sdk_ok=bool(sdk_ok), initialized=bool(trading_client), client_attached=bool(trading_client), has_key=bool(key), has_secret=bool(secret), base_url=base_url, paper=paper, shadow_mode=shadow))

    @app.route('/healthz')
    def healthz():
        """Minimal liveness probe."""
        return jsonify(ok=True)

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
        from ai_trading.config.settings import get_settings
        s = get_settings()
        port = int(s.api_port or 9001)
        app = create_app()
        app.logger.info('Starting Flask', extra={'port': port})
        app.run(host='0.0.0.0', port=port)
