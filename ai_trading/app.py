from __future__ import annotations
import logging
import os
from importlib import import_module
from typing import TYPE_CHECKING

from ai_trading.logging import get_logger
from ai_trading.utils.optional_dep import missing
from flask import jsonify

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from flask import Flask

_log = get_logger(__name__)

if missing("ai_trading.metrics", "metrics"):
    _PROM_OK, _PROM_REG = False, None
else:
    from ai_trading.metrics import PROMETHEUS_AVAILABLE as _PROM_OK, REGISTRY as _PROM_REG


def create_app():
    """Create and configure the Flask application."""
    # Bypass any mocked Flask import by resolving the class at call time
    FlaskClass = import_module("flask.app").Flask
    app: "Flask" = FlaskClass(__name__)

    # Some tests may monkeypatch Flask and return objects without a real config
    if not isinstance(getattr(app, "config", None), dict):
        app.config = dict(getattr(app, "config", {}))

    get_logger('werkzeug').setLevel(logging.ERROR)

    # Cache required env validation once during app startup.
    try:
        from ai_trading.config.management import validate_required_env
        validate_required_env()
        app.config["_ENV_VALID"] = True
        app.config["_ENV_ERR"] = None
    except (ImportError, RuntimeError) as e:
        _log.exception("ENV_VALIDATION_FAILED")
        app.config["_ENV_VALID"] = False
        app.config["_ENV_ERR"] = str(e)

    @app.route('/health')
    def health():
        """Lightweight liveness probe with Alpaca diagnostics."""
        ok = True
        error: str | None = None
        try:
            try:
                from ai_trading.alpaca_api import ALPACA_AVAILABLE as sdk_ok
            except ImportError as exc:
                sdk_ok = False
                ok = False
                error = str(exc)
            except (KeyError, ValueError, TypeError) as exc:
                sdk_ok = False
                error = str(exc)

            try:
                from ai_trading.core.bot_engine import _resolve_alpaca_env, trading_client
                key, secret, base_url = _resolve_alpaca_env()
                paper = bool(base_url and 'paper' in base_url)
            except (KeyError, ValueError, TypeError) as exc:
                trading_client, key, secret, base_url, paper = (None, None, None, '', False)
                if error is None:
                    error = str(exc)

            from ai_trading.config.management import is_shadow_mode

            shadow = is_shadow_mode()

            if error is not None:
                ok = False

            payload = dict(
                ok=ok,
                alpaca=dict(
                    sdk_ok=bool(sdk_ok),
                    initialized=bool(trading_client),
                    client_attached=bool(trading_client),
                    has_key=bool(key),
                    has_secret=bool(secret),
                    base_url=base_url,
                    paper=paper,
                    shadow_mode=shadow,
                ),
            )
            if error is not None:
                payload["error"] = error
            return jsonify(payload)
        except Exception as e:  # /health must not raise
            _log.exception("HEALTH_CHECK_FAILED")
            return jsonify(ok=False, error=str(e))

    @app.route('/healthz')
    def healthz():
        """Minimal liveness probe."""
        from datetime import UTC, datetime

        ok = bool(app.config.get("_ENV_VALID"))
        payload = {
            "ok": ok,
            "ts": datetime.now(UTC).isoformat(),
            "service": "ai-trading",
        }
        err = app.config.get("_ENV_ERR")
        if not ok and err:
            payload["error"] = err
        return jsonify(payload)

    @app.route('/metrics')
    def metrics():
        """Expose Prometheus metrics if available."""
        if not _PROM_OK:
            return ('metrics unavailable', 501)
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return generate_latest(_PROM_REG), 200, {'Content-Type': CONTENT_TYPE_LATEST}

    return app


def get_test_client():
    """Return a Flask test client or ``None`` if unavailable.

    Importing ``flask.testing`` can fail in environments that provide a
    lightweight Flask stub. This helper guards the import and falls back
    gracefully when the testing utilities are missing.
    """

    if missing("flask.testing", "flask.testing"):
        return None
    flask_testing = import_module("flask.testing")

    app = create_app()
    return flask_testing.FlaskClient(app)


if __name__ == '__main__':
    if os.getenv('RUN_HEALTHCHECK') == '1':
        from ai_trading.config.settings import get_settings

        app = create_app()
        s = get_settings()
        port = int(s.healthcheck_port or 9101)
        app.logger.info('Starting Flask', extra={'port': port})
        app.run(host='0.0.0.0', port=port)
