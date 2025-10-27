from __future__ import annotations
import logging
import os
import json
from collections.abc import Mapping
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
    try:
        from ai_trading.diagnostics.http_diag import diag_bp  # type: ignore
    except ImportError:
        diag_bp = None  # type: ignore[assignment]
    if diag_bp is not None:
        try:
            app.register_blueprint(diag_bp)
        except Exception:
            _log.debug("DIAG_BLUEPRINT_REGISTER_FAILED", exc_info=True)

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

        def _stamp_fallback_meta(
            payload: dict,
            *,
            used: bool,
            reasons: list[str] | tuple[str, ...] | None = None,
        ) -> dict:
            """Ensure payload carries structured fallback metadata."""

            meta = payload.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            fallback_meta = meta.get("fallback")
            if not isinstance(fallback_meta, dict):
                fallback_meta = {}
            fallback_meta["used"] = bool(used)
            clean_reasons: list[str] = []
            for reason in reasons or ():
                text = str(reason).strip()
                if text and text not in clean_reasons:
                    clean_reasons.append(text)
            if clean_reasons or used:
                fallback_meta["reasons"] = clean_reasons
            else:
                fallback_meta.pop("reasons", None)
            meta["fallback"] = fallback_meta
            payload["meta"] = meta
            return payload

        canonical_payload = _ensure_core_fields(_normalise_payload(data))
        fallback_payload = (
            _ensure_core_fields(_normalise_payload(fallback))
            if fallback is not None
            else {}
        )

        response_payload = _merge_payloads(canonical_payload, fallback_payload)
        sanitized_payload = _stamp_fallback_meta(
            _ensure_core_fields(dict(response_payload)), used=False, reasons=[]
        )

        func = globals().get("jsonify")
        fallback_used = False
        fallback_reasons: list[str] = []
        if callable(func):
            try:
                response = func(dict(sanitized_payload))
            except Exception as exc:  # pragma: no cover - defensive fallback
                _log.exception("HEALTH_JSONIFY_FALLBACK", exc_info=exc)
                fallback_used = True
                reason_candidates = [str(exc).strip(), exc.__class__.__name__]
                fallback_reasons.extend(
                    reason
                    for reason in dict.fromkeys(reason_candidates)
                    if reason
                )
                sanitized_payload = _stamp_fallback_meta(
                    sanitized_payload, used=True, reasons=fallback_reasons
                )
            else:
                has_get_data = callable(getattr(response, "get_data", None))
                has_status = hasattr(response, "status_code")
                if has_get_data and has_status:
                    try:
                        response.status_code = status
                    except Exception:  # pragma: no cover - defensive
                        pass
                    return response
                response = None
        else:
            fallback_used = True
            fallback_reasons.append("jsonify unavailable")
            if '_jsonify_import_error' in globals() and _jsonify_import_error is not None:
                import_reason = str(_jsonify_import_error).strip()
                if import_reason:
                    fallback_reasons.append(import_reason)
                fallback_reasons.append("ImportError")
            sanitized_payload = _stamp_fallback_meta(
                sanitized_payload, used=True, reasons=fallback_reasons
            )

        final_payload = _ensure_core_fields(dict(sanitized_payload))

        if fallback_used:
            final_payload["ok"] = False

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

        sanitized_payload = _ensure_core_fields(dict(final_payload))
        sanitized_payload = _stamp_fallback_meta(
            sanitized_payload, used=fallback_used, reasons=fallback_reasons
        )

        try:
            body = json.dumps(sanitized_payload, default=str)
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("HEALTH_JSON_ENCODE_FAILED", exc_info=exc)
            extra_reason = str(exc).strip() or exc.__class__.__name__ or "serialization_error"
            fallback_used = True
            fallback_reasons = [
                reason
                for reason in dict.fromkeys([*fallback_reasons, extra_reason])
                if reason
            ]
            alpaca_section = _normalise_alpaca_section(sanitized_payload.get("alpaca"))
            sanitized_payload = _stamp_fallback_meta(
                _ensure_core_fields(
                    {
                        "ok": False,
                        "alpaca": alpaca_section,
                        "error": extra_reason,
                    }
                ),
                used=True,
                reasons=fallback_reasons,
            )
            body = json.dumps(sanitized_payload, default=str)

        response_factory = getattr(app, "response_class", None)
        if callable(response_factory):
            sanitized_payload = _stamp_fallback_meta(
                _ensure_core_fields(dict(sanitized_payload)),
                used=fallback_used,
                reasons=fallback_reasons,
            )
            return response_factory(body, status=status, mimetype="application/json")

        # When ``response_class`` is unavailable (for example when stub clients swap
        # Flask out for light-weight shims) surface the fully populated payload
        # directly so callers don't need to understand Flask's ``(body, status)``
        # tuple convention. Callers running under a real Flask stack will already
        # receive a wrapped ``Response`` above, preserving status semantics.
        sanitized_payload = _stamp_fallback_meta(
            _ensure_core_fields(dict(sanitized_payload)),
            used=fallback_used,
            reasons=fallback_reasons,
        )
        return sanitized_payload

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

    original_test_client = getattr(app, "test_client", None)

    if callable(original_test_client):  # pragma: no cover - exercised via tests

        class _ResponseWrapper(Mapping):
            def __init__(self, data: dict, text: str, status_code: int) -> None:
                self._payload = dict(data)
                self._text = text
                self.status_code = status_code

            def __getitem__(self, key):
                return self._payload[key]

            def __iter__(self):
                return iter(self._payload)

            def __len__(self):
                return len(self._payload)

            def get_json(self):
                return dict(self._payload)

            def get_data(self, as_text: bool = False):
                if as_text:
                    return self._text
                return self._text.encode("utf-8")

        def _wrap_response(resp: Any) -> Any:
            if callable(getattr(resp, "get_data", None)) and callable(getattr(resp, "get_json", None)):
                return resp
            status_code = getattr(resp, "status_code", 200)
            payload = resp
            if callable(getattr(resp, "get_json", None)):
                try:
                    payload = resp.get_json()
                except Exception:
                    payload = resp
            if isinstance(payload, Mapping):
                payload_dict = dict(payload)
            elif isinstance(payload, list):
                payload_dict = {"data": payload}
            else:
                payload_dict = {"data": payload}
            try:
                body = json.dumps(payload_dict, default=str)
            except Exception:
                payload_dict = {
                    "ok": bool(payload_dict.get("ok", False)),
                    "alpaca": _normalise_alpaca_section(payload_dict.get("alpaca", {})),
                    "error": str(payload_dict.get("error", "serialization_error")),
                }
                body = json.dumps(payload_dict, default=str)
            return _ResponseWrapper(payload_dict, body, status_code)

        def _patched_test_client(*args: Any, **kwargs: Any):
            client = original_test_client(*args, **kwargs)
            getter = getattr(client, "get", None)
            if callable(getter):
                def _patched_get(path: str, *g_args: Any, **g_kwargs: Any):
                    raw = getter(path, *g_args, **g_kwargs)
                    return _wrap_response(raw)

                client.get = _patched_get
            return client

        app.test_client = _patched_test_client

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
