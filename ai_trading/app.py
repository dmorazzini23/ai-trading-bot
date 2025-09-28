from __future__ import annotations
import logging
import os
import json
from importlib import import_module
from typing import TYPE_CHECKING, Any

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

    def _json_response(data: dict, *, status: int = 200, fallback: dict | None = None) -> Any:
        """Return a JSON ``Response`` with a resilient fallback.

        When ``jsonify`` is unavailable or raises, fall back to ``json.dumps`` with
        a stable payload to keep ``/health`` reliable under tests that monkeypatch
        Flask internals.
        """

        func = globals().get("jsonify")
        failure_exc: Exception | None = None
        if callable(func):
            try:
                response = func(data)
            except Exception as exc:  # pragma: no cover - defensive fallback
                _log.exception("HEALTH_JSONIFY_FALLBACK", exc_info=exc)
                failure_exc = exc
            else:
                try:
                    response.status_code = status
                except Exception:  # pragma: no cover - defensive
                    pass
                return response

        payload = fallback if fallback is not None else data
        if failure_exc is not None:
            message = str(failure_exc) or failure_exc.__class__.__name__
            if not isinstance(payload, dict):
                payload = {"ok": False, "error": message}
            else:
                payload = dict(payload)
                payload.setdefault("alpaca", {})
                if not payload.get("error"):
                    payload["error"] = message
                payload["ok"] = False
        try:
            body = json.dumps(payload, default=str)
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("HEALTH_JSON_ENCODE_FAILED", exc_info=exc)
            safe_payload = {
                "ok": False,
                "alpaca": payload.get("alpaca", {}),
                "error": str(exc) or "serialization_error",
            }
            payload = safe_payload
            body = json.dumps(payload, default=str)

        response_factory = getattr(app, "response_class", None)
        if callable(response_factory):
            return response_factory(body, status=status, mimetype="application/json")
        return payload, status

    @app.route('/health')
    def health():
        """Lightweight liveness probe with Alpaca diagnostics."""
        ok = True
        errors: list[str] = []
        minimal_alpaca_payload = dict(
            sdk_ok=False,
            initialized=False,
            client_attached=False,
            has_key=False,
            has_secret=False,
            base_url="",
            paper=False,
            shadow_mode=False,
        )
        alpaca_payload = dict(minimal_alpaca_payload)
        sdk_ok = False
        trading_client = None
        key = secret = None
        base_url = ""
        paper = False
        shadow = False
        last_error: str | None = None
        alpaca_import_ok = True

        def record_error(exc: Exception) -> str:
            nonlocal last_error, ok
            ok = False
            message = str(exc) or exc.__class__.__name__
            if message and message not in errors:
                errors.append(message)
            if message:
                last_error = message
            return message

        try:
            from ai_trading.alpaca_api import ALPACA_AVAILABLE as sdk_ok
        except ImportError as exc:
            ok = False
            record_error(exc)
            alpaca_import_ok = False
            sdk_ok = False
        except (KeyError, ValueError, TypeError) as exc:
            record_error(exc)
            alpaca_import_ok = False

        if alpaca_import_ok:
            try:
                from ai_trading.core.bot_engine import _resolve_alpaca_env, trading_client as _trading_client
                trading_client = _trading_client
                key, secret, base_url = _resolve_alpaca_env()
                base_url = base_url or ""
                paper = bool(base_url and 'paper' in base_url)
            except Exception as exc:  # pragma: no cover - defensive against unexpected import failures
                ok = False
                record_error(exc)
                trading_client, key, secret, base_url, paper = (None, None, None, '', False)
        else:
            trading_client, key, secret, base_url, paper = (None, None, None, '', False)

        try:
            from ai_trading.config.management import is_shadow_mode
            shadow = is_shadow_mode()
        except Exception as exc:  # pragma: no cover - defensive against unexpected import failures
            ok = False
            record_error(exc)
            shadow = False
        else:
            shadow = bool(shadow)

        if errors:
            ok = False

        alpaca_payload.update(
            sdk_ok=bool(sdk_ok),
            initialized=bool(trading_client),
            client_attached=bool(trading_client),
            has_key=bool(key),
            has_secret=bool(secret),
            base_url=base_url,
            paper=paper,
            shadow_mode=shadow,
        )

        payload = dict(
            ok=ok,
            alpaca=alpaca_payload,
        )
        if errors:
            payload["error"] = "; ".join(errors)
            payload["ok"] = False
        fallback_payload = dict(payload)
        fallback_payload["alpaca"] = dict(minimal_alpaca_payload)
        if errors:
            err_msg = payload.get("error") or last_error or "; ".join(errors)
            if err_msg:
                fallback_payload["error"] = err_msg
        else:
            fallback_payload.setdefault("error", last_error or "jsonify unavailable")
        fallback_payload["ok"] = False

        return _json_response(payload, fallback=fallback_payload)

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

    module_name = "flask.testing"
    feature_name = "flask.testing"

    if missing(module_name, feature_name):
        return None

    try:
        flask_testing = import_module(module_name)
    except ImportError:
        # The dependency may have been removed after the cache was populated.
        # Clear and repopulate the cache so future calls see the updated state.
        try:
            missing.cache_clear()
        except AttributeError:
            pass

        if missing(module_name, feature_name):
            return None

        try:
            flask_testing = import_module(module_name)
        except ImportError:
            return None

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
