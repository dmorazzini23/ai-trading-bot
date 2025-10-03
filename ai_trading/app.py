from __future__ import annotations
import logging
import os
import json
from importlib import import_module
from typing import TYPE_CHECKING, Any

from ai_trading.logging import get_logger
from ai_trading.utils.optional_dep import missing
try:
    from flask import jsonify as _jsonify
except ImportError as _jsonify_import_error:  # pragma: no cover - exercised via tests
    jsonify = None  # type: ignore[assignment]
else:  # pragma: no cover - import path only evaluated once
    jsonify = _jsonify  # noqa: F401
    _jsonify_import_error = None

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from flask import Flask

_log = get_logger(__name__)

if missing("ai_trading.metrics", "metrics"):
    _PROM_OK, _PROM_REG = False, None
else:
    from ai_trading.metrics import PROMETHEUS_AVAILABLE as _PROM_OK, REGISTRY as _PROM_REG

_ALPACA_SECTION_DEFAULTS: dict[str, Any] = {
    "sdk_ok": False,
    "initialized": False,
    "client_attached": False,
    "has_key": False,
    "has_secret": False,
    "base_url": "",
    "paper": False,
    "shadow_mode": False,
}

_ALPACA_BOOL_KEYS = {
    "sdk_ok",
    "initialized",
    "client_attached",
    "has_key",
    "has_secret",
    "paper",
    "shadow_mode",
}


def _normalise_alpaca_section(raw: Any) -> dict[str, Any]:
    """Return a fresh Alpaca payload seeded with required keys."""

    normalised = dict(_ALPACA_SECTION_DEFAULTS)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in _ALPACA_BOOL_KEYS:
                normalised[key] = bool(value)
            elif key == "base_url":
                normalised[key] = str(value) if value is not None else ""
            elif key in normalised:
                normalised[key] = value
    return normalised


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
        """Return a JSON ``Response`` with a resilient fallback."""

        def _normalise_payload(raw: dict | None) -> dict:
            """Return a shallow copy with defensive defaults."""

            base = dict(raw or {})
            base.setdefault("ok", False)
            base["ok"] = bool(base.get("ok", False))
            base["alpaca"] = _normalise_alpaca_section(base.get("alpaca"))
            return base

        def _ensure_core_fields(payload: dict) -> dict:
            """Ensure ``ok`` and ``alpaca`` keys exist with safe defaults."""

            normalised = dict(payload or {})
            normalised.setdefault("ok", False)
            normalised["ok"] = bool(normalised.get("ok", False))
            normalised["alpaca"] = _normalise_alpaca_section(normalised.get("alpaca"))
            return normalised

        def _merge_payloads(primary: dict, secondary: dict) -> dict:
            """Merge fallback data into the canonical payload."""

            merged = _ensure_core_fields(dict(secondary)) if secondary else _ensure_core_fields({})
            primary = _ensure_core_fields(dict(primary))

            merged_alpaca = _normalise_alpaca_section(merged.get("alpaca"))
            primary_alpaca = _normalise_alpaca_section(primary.get("alpaca"))
            merged_alpaca.update(primary_alpaca)
            merged["alpaca"] = _normalise_alpaca_section(merged_alpaca)

            for key, value in primary.items():
                if key == "alpaca":
                    continue
                if key == "ok":
                    merged["ok"] = bool(value)
                else:
                    merged[key] = value

            return _ensure_core_fields(merged)

        canonical_payload = _ensure_core_fields(_normalise_payload(data))
        fallback_payload = (
            _ensure_core_fields(_normalise_payload(fallback))
            if fallback is not None
            else {}
        )

        response_payload = _merge_payloads(canonical_payload, fallback_payload)
        sanitized_payload = _ensure_core_fields(dict(response_payload))

        func = globals().get("jsonify")
        fallback_used = False
        fallback_reasons: list[str] = []
        if callable(func):
            try:
                response = func(dict(sanitized_payload))
            except Exception as exc:  # pragma: no cover - defensive fallback
                _log.exception("HEALTH_JSONIFY_FALLBACK", exc_info=exc)
                fallback_used = True
                fallback_reasons.extend(
                    reason
                    for reason in {str(exc) or exc.__class__.__name__, exc.__class__.__name__}
                    if reason
                )
            else:
                try:
                    response.status_code = status
                except Exception:  # pragma: no cover - defensive
                    pass
                return response
        else:
            fallback_used = True
            fallback_reasons.append("jsonify unavailable")
            if '_jsonify_import_error' in globals() and _jsonify_import_error is not None:
                import_reason = str(_jsonify_import_error).strip()
                if import_reason:
                    fallback_reasons.append(import_reason)
                fallback_reasons.append("ImportError")

        final_payload = dict(sanitized_payload)

        if fallback_used:
            final_payload["ok"] = False

        # Ensure the exposed payload always carries the canonical structure
        # regardless of how we arrive here (missing ``jsonify`` or runtime
        # failures). Re-running the normaliser guarantees ``ok`` and
        # ``alpaca`` are present and that ``alpaca`` is seeded from
        # ``_ALPACA_SECTION_DEFAULTS``.
        final_payload = _ensure_core_fields(final_payload)

        message_candidates: list[str] = []
        existing_error = final_payload.get("error")
        if existing_error is None:
            existing_error = fallback_payload.get("error") or canonical_payload.get("error")
        existing_error_str: str | None = None
        if isinstance(existing_error, str):
            existing_error_str = existing_error.strip() or None
            if existing_error_str:
                message_candidates.append(existing_error_str)

        for reason in fallback_reasons:
            if reason and reason not in message_candidates:
                message_candidates.append(reason)

        if fallback_used and not message_candidates:
            message_candidates.append("jsonify unavailable")

        if message_candidates:
            merged = "; ".join(dict.fromkeys(message_candidates))
            if existing_error_str is not None or not final_payload.get("error"):
                final_payload["error"] = merged
            else:
                # Preserve non-string canonical error payloads while surfacing
                # fallback context in a dedicated string field.
                final_payload.setdefault("error_details", {})
                if isinstance(final_payload["error_details"], dict):
                    final_payload["error_details"].setdefault("messages", merged)
                else:
                    final_payload["error_details"] = {"messages": merged}

        try:
            final_payload = _ensure_core_fields(final_payload)
            body = json.dumps(final_payload, default=str)
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("HEALTH_JSON_ENCODE_FAILED", exc_info=exc)
            fallback_reasons = []
            fallback_used = True
            alpaca_section = _normalise_alpaca_section(final_payload.get("alpaca"))
            safe_payload = {
                "ok": False,
                "alpaca": alpaca_section,
                "error": str(exc) or exc.__class__.__name__ or "serialization_error",
            }
            final_payload = _ensure_core_fields(safe_payload)
            body = json.dumps(final_payload, default=str)

        response_factory = getattr(app, "response_class", None)
        if callable(response_factory):
            final_payload = _ensure_core_fields(final_payload)
            return response_factory(body, status=status, mimetype="application/json")

        # When ``response_class`` is unavailable (for example when stub clients swap
        # Flask out for light-weight shims) surface the fully populated payload
        # directly so callers don't need to understand Flask's ``(body, status)``
        # tuple convention. Callers running under a real Flask stack will already
        # receive a wrapped ``Response`` above, preserving status semantics.
        final_payload = _ensure_core_fields(final_payload)
        return final_payload

    @app.route('/health')
    def health():
        """Lightweight liveness probe with Alpaca diagnostics."""
        ok = True
        errors: list[str] = []
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

        if (not alpaca_import_ok) or (not key) or (not secret):
            shadow = False

        if errors:
            ok = False

        alpaca_payload = _normalise_alpaca_section(
            {
                "sdk_ok": sdk_ok,
                "initialized": bool(trading_client),
                "client_attached": bool(trading_client),
                "has_key": bool(key),
                "has_secret": bool(secret),
                "base_url": base_url,
                "paper": paper,
                "shadow_mode": shadow,
            }
        )

        payload = {
            "ok": bool(ok),
            "alpaca": dict(alpaca_payload),
        }
        if errors:
            payload["error"] = "; ".join(dict.fromkeys(errors))
            payload["ok"] = False

        if errors:
            err_msg = payload.get("error") or last_error or "; ".join(errors)
        else:
            err_msg = last_error

        fallback_payload = {
            "ok": payload["ok"],
            "alpaca": dict(alpaca_payload),
        }
        if err_msg:
            fallback_payload["error"] = err_msg

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
