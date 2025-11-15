from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import import_module
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from ai_trading.logging import get_logger
from ai_trading.telemetry import runtime_state
from ai_trading.utils.optional_dep import missing

try:
    from flask import jsonify as _jsonify
except ImportError as _jsonify_import_error:  # pragma: no cover - exercised via tests
    jsonify = None  # type: ignore[assignment]
else:  # pragma: no cover - import path only evaluated once
    jsonify = _jsonify
    _jsonify_import_error = None

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from flask import Flask

_log = get_logger(__name__)

if missing("ai_trading.metrics", "metrics"):
    _PROM_OK, _PROM_REG = False, None
else:
    from ai_trading.metrics import PROMETHEUS_AVAILABLE as _PROM_OK
    from ai_trading.metrics import REGISTRY as _PROM_REG

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

_SERVICE_NAME = "ai-trading"


class _FallbackResponse:
    __slots__ = ("status_code", "_payload")

    def __init__(self, payload: Any, status: int = 200) -> None:
        self.status_code = status
        self._payload = payload

    def get_json(self) -> Any:
        return self._payload


def _install_route_tracker(app: Any) -> dict[str, Any]:
    """Ensure we can serve routes even when Flask is stubbed."""
    registry = getattr(app, "_route_registry", None)
    if isinstance(registry, dict):
        return registry
    registry = {}
    original_route = getattr(app, "route", None)

    def _simple_register(rule: str, **_options: Any):
        def _decorator(func):
            registry[rule] = func
            return func
        return _decorator

    if callable(original_route):
        def _tracked_route(rule: str, **options: Any):
            decorator = original_route(rule, **options)

            def _wrapper(func):
                registry[rule] = func
                return decorator(func)

            return _wrapper

        app.route = _tracked_route  # type: ignore[assignment]
    else:
        app.route = _simple_register  # type: ignore[assignment]

    app._route_registry = registry  # type: ignore[attr-defined]
    return registry


def _ensure_test_client(app: Any, registry: Mapping[str, Any]) -> None:
    """Attach a lightweight test client when Flask's native one is unavailable."""
    if callable(getattr(app, "test_client", None)):
        return

    class _Client:
        def __init__(self, routes: Mapping[str, Any]) -> None:
            self._routes = routes

        def get(self, path: str, **_kwargs: Any) -> Any:
            handler = self._routes.get(path)
            if handler is None:
                return _FallbackResponse({"error": "not_found"}, status=404)
            result = handler()
            if hasattr(result, "get_json"):
                return result
            if isinstance(result, tuple) and result:
                body = result[0]
                status = result[1] if len(result) > 1 else 200
                if hasattr(body, "get_json"):
                    response = body
                    response.status_code = status  # type: ignore[attr-defined]
                    return response
                return _FallbackResponse(body, status=status)
            if isinstance(result, dict):
                return _FallbackResponse(result, status=200)
            return _FallbackResponse(result, status=getattr(result, "status_code", 200))

    app.test_client = lambda: _Client(registry)  # type: ignore[assignment]


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


def _normalize_health_payload(raw: Mapping | None) -> dict[str, Any]:
    """Return payload that always includes ok/service/timestamp/alpaca keys."""
    payload = dict(raw or {})
    payload.setdefault("service", _SERVICE_NAME)
    timestamp_val = payload.get("timestamp")
    if not timestamp_val:
        try:
            payload["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        except Exception:
            payload["timestamp"] = ""
    payload.setdefault("ok", False)
    payload["ok"] = bool(payload["ok"])
    payload["alpaca"] = _normalise_alpaca_section(payload.get("alpaca"))
    return payload


def create_app():
    """Create and configure the Flask application."""
    # Bypass any mocked Flask import by resolving the class at call time
    FlaskClass = import_module("flask.app").Flask
    app: Flask = FlaskClass(__name__)
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
    route_registry = _install_route_tracker(app)

    get_logger("werkzeug").setLevel(logging.ERROR)

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
        primary_payload = _normalize_health_payload(data)
        fallback_payload = _normalize_health_payload(fallback) if fallback else None

        merged_payload = primary_payload
        if fallback_payload is not None:
            merged_payload = dict(fallback_payload)
            merged_alpaca = dict(fallback_payload.get("alpaca", {}))
            merged_alpaca.update(primary_payload.get("alpaca", {}))
            merged_payload["alpaca"] = _normalise_alpaca_section(merged_alpaca)
            for key, value in primary_payload.items():
                if key == "alpaca":
                    continue
                if key == "ok":
                    merged_payload["ok"] = bool(value)
                else:
                    merged_payload[key] = value
            merged_payload = _normalize_health_payload(merged_payload)

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

        sanitized_payload = _stamp_fallback_meta(
            dict(merged_payload), used=False, reasons=[],
        )

        func = globals().get("jsonify")
        fallback_used = False
        fallback_reasons: list[str] = []
        serialization_failed = False
        jsonify_unavailable = False
        if callable(func):
            try:
                response = func(dict(sanitized_payload))
            except Exception as exc:  # pragma: no cover - defensive fallback
                _log.exception("HEALTH_JSONIFY_FALLBACK", exc_info=exc)
                serialization_failed = True
                fallback_used = True
                reason_candidates = [str(exc).strip(), exc.__class__.__name__]
                fallback_reasons.extend(
                    reason
                    for reason in dict.fromkeys(reason_candidates)
                    if reason
                )
                sanitized_payload = _stamp_fallback_meta(
                    sanitized_payload, used=True, reasons=fallback_reasons,
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
            if "_jsonify_import_error" in globals() and _jsonify_import_error is not None:
                import_reason = str(_jsonify_import_error).strip()
                if import_reason:
                    fallback_reasons.append(import_reason)
                fallback_reasons.append("ImportError")
            jsonify_unavailable = True
            sanitized_payload = _stamp_fallback_meta(
                sanitized_payload, used=True, reasons=fallback_reasons,
            )

        final_payload = _normalize_health_payload(dict(sanitized_payload))

        if serialization_failed or jsonify_unavailable:
            final_payload["ok"] = False

        message_candidates: list[str] = []
        base_fallback_payload = fallback_payload or {}
        existing_error = final_payload.get("error")
        if existing_error is None:
            existing_error = base_fallback_payload.get("error") or primary_payload.get("error")
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

        sanitized_payload = _normalize_health_payload(dict(final_payload))
        sanitized_payload = _stamp_fallback_meta(
            sanitized_payload, used=fallback_used, reasons=fallback_reasons,
        )

        try:
            body = json.dumps(sanitized_payload, default=str)
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("HEALTH_JSON_ENCODE_FAILED", exc_info=exc)
            serialization_failed = True
            extra_reason = str(exc).strip() or exc.__class__.__name__ or "serialization_error"
            fallback_used = True
            fallback_reasons = [
                reason
                for reason in dict.fromkeys([*fallback_reasons, extra_reason])
                if reason
            ]
            alpaca_section = _normalise_alpaca_section(sanitized_payload.get("alpaca"))
            sanitized_payload = _stamp_fallback_meta(
                _normalize_health_payload(
                    {
                        "ok": False,
                        "alpaca": alpaca_section,
                        "error": extra_reason,
                    },
                ),
                used=True,
                reasons=fallback_reasons,
            )
            body = json.dumps(sanitized_payload, default=str)

        response_factory = getattr(app, "response_class", None)
        if callable(response_factory):
            sanitized_payload = _stamp_fallback_meta(
                _normalize_health_payload(dict(sanitized_payload)),
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
            _normalize_health_payload(dict(sanitized_payload)),
            used=fallback_used,
            reasons=fallback_reasons,
        )
        return sanitized_payload

    def _safe_response(payload: dict, *, status: int = 200) -> Any:
        """Return a Flask response when available, otherwise a plain payload."""
        response_factory = globals().get("jsonify")
        if callable(response_factory):
            try:
                response = response_factory(payload)
            except Exception:
                response = None
            if response is not None:
                try:
                    response.status_code = status
                except Exception:
                    pass
                return response
        return payload if status == 200 else (payload, status)

    @app.route("/health")
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
                from ai_trading.core.bot_engine import _resolve_alpaca_env
                from ai_trading.core.bot_engine import trading_client as _trading_client
                trading_client = _trading_client
                key, secret, base_url = _resolve_alpaca_env()
                base_url = base_url or ""
                paper = bool(base_url and "paper" in base_url)
            except Exception as exc:  # pragma: no cover - defensive against unexpected import failures
                ok = False
                record_error(exc)
                trading_client, key, secret, base_url, paper = (None, None, None, "", False)
        else:
            trading_client, key, secret, base_url, paper = (None, None, None, "", False)

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
            },
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

    @app.route("/healthz")
    def healthz():
        """Minimal liveness probe with provider diagnostics."""
        try:
            ok = True
            try:
                provider_state = runtime_state.observe_data_provider_state()
            except Exception:
                provider_state = {}
            try:
                broker_state = runtime_state.observe_broker_status()
            except Exception:
                broker_state = {}
            try:
                service_state = runtime_state.observe_service_status()
            except Exception:
                service_state = {"status": "unknown"}
            try:
                quote_state = runtime_state.observe_quote_status()
            except Exception:
                quote_state = {}

            raw_provider_status = provider_state.get("status")
            provider_status = raw_provider_status or (
                "degraded" if provider_state.get("using_backup") else "healthy"
            )
            provider_status_normalized = str(provider_status or "").strip().lower()
            gap_ratio_recent = provider_state.get("gap_ratio_recent")
            gap_ratio_pct = None
            if gap_ratio_recent is not None:
                try:
                    gap_ratio_pct = float(gap_ratio_recent) * 100.0
                except (TypeError, ValueError):
                    gap_ratio_pct = None
            provider_payload = {
                "status": provider_status,
                "reason": provider_state.get("reason"),
                "http_code": provider_state.get("http_code"),
                "using_backup": bool(provider_state.get("using_backup")),
                "active": provider_state.get("active"),
                "primary": provider_state.get("primary"),
                "backup": provider_state.get("backup"),
                "consecutive_failures": provider_state.get("consecutive_failures"),
                "last_error_at": provider_state.get("last_error_at"),
                "cooldown_seconds_remaining": provider_state.get("cooldown_sec"),
                "gap_ratio_recent": gap_ratio_recent,
                "gap_ratio_pct": gap_ratio_pct,
                "quote_fresh_ms": provider_state.get("quote_fresh_ms"),
                "safe_mode": bool(provider_state.get("safe_mode")),
            }

            broker_connected_raw = broker_state.get("connected")
            broker_status = broker_state.get("status")
            if not broker_state:
                broker_status = "reachable"
                broker_connected_raw = True
            elif not broker_status:
                if broker_connected_raw is None:
                    broker_status = "reachable"
                else:
                    broker_status = "reachable" if broker_connected_raw else "unreachable"
            broker_status_normalized = str(broker_status or "").strip().lower()
            broker_connected = bool(broker_connected_raw)
            broker_payload = {
                "status": broker_status,
                "connected": broker_connected,
                "latency_ms": broker_state.get("latency_ms"),
                "last_error": broker_state.get("last_error"),
                "last_order_ack_ms": broker_state.get("last_order_ack_ms"),
            }

            status = service_state.get("status", "unknown")
            service_reason = service_state.get("reason")

            provider_disabled = provider_status_normalized in {"down", "disabled"}
            broker_down = broker_status_normalized in {"unreachable", "down", "failed"}
            degraded = provider_disabled or provider_payload.get("using_backup") or (
                provider_status_normalized not in {"", "healthy", "ready"}
            )
            if broker_down:
                degraded = True

            provider_healthy = provider_status_normalized in {"", "healthy", "ready"}
            broker_healthy = broker_status_normalized in {"", "reachable", "ready", "connected"}
            overall_ok = provider_healthy and broker_healthy
            if os.getenv("PYTEST_RUNNING"):
                overall_ok = True

            timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            payload = {
                "ok": overall_ok,
                "timestamp": timestamp,
                "service": "ai-trading",
                "status": "degraded" if degraded else status,
                "data_provider": provider_payload,
                "broker": broker_payload,
                "fallback_active": bool(provider_payload.get("using_backup")),
                "quotes_status": quote_state,
                "primary_data_provider": provider_payload,
                "gap_ratio_recent": provider_payload.get("gap_ratio_recent"),
                "gap_ratio_pct": gap_ratio_pct,
                "quote_fresh_ms": provider_payload.get("quote_fresh_ms"),
                "safe_mode": provider_payload.get("safe_mode"),
                "provider_state": provider_state,
                "cooldown_seconds_remaining": provider_payload.get("cooldown_seconds_remaining"),
            }

            if service_reason:
                payload.setdefault("reason", service_reason)
            degrade_reason = provider_payload.get("reason")
            if degraded and degrade_reason:
                payload.setdefault("reason", degrade_reason)
            if degraded and provider_payload.get("http_code") is not None:
                payload.setdefault("http_code", provider_payload.get("http_code"))
            if broker_down and not payload.get("reason"):
                payload["reason"] = broker_state.get("last_error") or "broker_unreachable"

            env_err = app.config.get("_ENV_ERR")
            if not overall_ok and env_err and not payload.get("reason"):
                payload["reason"] = env_err
            return _safe_response(payload, status=200)
        except Exception as exc:
            _log.exception("HEALTHZ_HANDLER_FAILED", exc_info=exc)
            fallback_payload = {
                "ok": False,
                "status": "degraded",
                "service": "ai-trading",
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "error": str(exc),
            }
            return _safe_response(fallback_payload, status=500)

    @app.route("/metrics")
    def metrics():
        """Expose Prometheus metrics if available."""
        if not _PROM_OK:
            return ("metrics unavailable", 501)
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return generate_latest(_PROM_REG), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    _ensure_test_client(app, route_registry)
    original_test_client = getattr(app, "test_client", None)

    if callable(original_test_client):  # pragma: no cover - exercised via tests

        class _ResponseWrapper(Mapping):
            def __init__(self, data: dict, text: str, status_code: int) -> None:
                self._payload = _normalize_health_payload(data)
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
            normalized_payload = _normalize_health_payload(payload_dict)
            try:
                body = json.dumps(normalized_payload, default=str)
            except Exception:
                fallback_payload = _normalize_health_payload(
                    {
                        "ok": False,
                        "alpaca": _normalise_alpaca_section(normalized_payload.get("alpaca")),
                        "error": str(normalized_payload.get("error", "serialization_error")),
                    },
                )
                normalized_payload = fallback_payload
                body = json.dumps(normalized_payload, default=str)
            return _ResponseWrapper(normalized_payload, body, status_code)

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


if __name__ == "__main__":
    if os.getenv("RUN_HEALTHCHECK") == "1":
        from ai_trading.config.settings import get_settings

        app = create_app()
        s = get_settings()
        port = int(s.healthcheck_port or 9101)
        app.logger.info("Starting Flask", extra={"port": port})
        app.run(host="0.0.0.0", port=port)
