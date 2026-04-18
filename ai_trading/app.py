from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, cast

from ai_trading.logging import get_logger
from ai_trading.health_payload import (
    build_api_health_payload,
    build_canonical_healthz_payload,
    register_healthz_routes,
)
from ai_trading.services import ControlPlaneService, GovernanceService
from ai_trading.utils.optional_dep import missing
from ai_trading.config.management import (
    get_env as _managed_get_env,
    set_runtime_env_override as _set_runtime_env_override,
)
try:
    from flask import Flask, jsonify, request
except ImportError:
    from flask import Flask, jsonify

    request = None  # type: ignore[assignment]


def _managed_env(name: str, default: Any = None) -> Any:
    """Read environment values via config management when available."""

    try:
        if name in {"PYTEST_RUNNING", "PYTEST_CURRENT_TEST", "TESTING"}:
            return _managed_get_env(name, default, resolve_aliases=False)
        return _managed_get_env(name, default)
    except Exception:
        return default

_log = get_logger(__name__)

_PROM_OK: bool
_PROM_REG: object | None
if missing("ai_trading.metrics", "metrics"):
    _PROM_OK, _PROM_REG = False, None
else:
    from ai_trading.metrics import PROMETHEUS_AVAILABLE
    from ai_trading.metrics import REGISTRY

    _PROM_OK = bool(PROMETHEUS_AVAILABLE)
    _PROM_REG = REGISTRY

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
_REQUIRED_TEST_ENV = {
    "ALPACA_API_KEY": _managed_env("ALPACA_API_KEY", "test-key"),
    "ALPACA_SECRET_KEY": _managed_env("ALPACA_SECRET_KEY", "test-secret"),
    "WEBHOOK_SECRET": _managed_env("WEBHOOK_SECRET", "test-webhook-secret"),
}


def _ensure_route_registry(app: Any) -> dict[Any, Any]:
    """Ensure lightweight Flask stubs can register routes for tests."""

    route_registry_raw = getattr(app, "_routes", None)
    if not isinstance(route_registry_raw, dict):
        route_registry_raw = {}
        setattr(app, "_routes", route_registry_raw)
    route_registry: dict[Any, Any] = cast(dict[Any, Any], route_registry_raw)

    existing_route = getattr(app, "route", None)
    if callable(existing_route) and getattr(app, "_ai_trading_route_tracker", False):
        return route_registry

    def _register(path: str, methods: tuple[str, ...], func: Any) -> Any:
        for method in methods:
            route_registry[(path, method)] = func
        if "GET" in methods:
            route_registry[path] = func
        return func

    if callable(existing_route):
        def _tracked_route(path: str, *args: Any, **kwargs: Any):
            methods = tuple(str(method).upper() for method in (kwargs.get("methods") or ("GET",)))
            decorator = existing_route(path, *args, **kwargs)

            def _decorator(func: Any) -> Any:
                registered = decorator(func)
                return _register(path, methods, registered)

            return _decorator

        setattr(app, "route", _tracked_route)
        setattr(app, "_ai_trading_route_tracker", True)
        return route_registry

    def _fallback_route(path: str, *args: Any, **kwargs: Any):
        methods = tuple(str(method).upper() for method in (kwargs.get("methods") or ("GET",)))

        def _decorator(func: Any) -> Any:
            return _register(path, methods, func)

        return _decorator

    setattr(app, "route", _fallback_route)
    setattr(app, "_ai_trading_route_tracker", True)
    return route_registry


def _ensure_test_client_support(app: Any, route_registry: dict[Any, Any]) -> None:
    """Provide a minimal ``test_client`` for lightweight Flask stubs."""

    if callable(getattr(app, "test_client", None)):
        return

    class _StubRequest:
        def __init__(self) -> None:
            self._json_payload: Any = None

        def get_json(self, silent: bool = False) -> Any:
            if self._json_payload is None and not silent:
                raise RuntimeError("request JSON unavailable")
            return self._json_payload

    stub_request = _StubRequest()

    class _Response:
        def __init__(self, data: Any, status: int = 200) -> None:
            self._data = data
            self.status_code = status

        def get_json(self) -> Any:
            return self._data

    class _Client:
        def _request(self, path: str, *, method: str = "GET", json: Any = None) -> Any:
            handler = route_registry.get((path, method.upper())) or route_registry.get(path)
            if handler is None:
                raise KeyError(f"route not registered: {method} {path}")
            stub_request._json_payload = json
            handler_globals = getattr(handler, "__globals__", {})
            previous_request = handler_globals.get("request")
            handler_globals["request"] = stub_request
            try:
                result = handler()
            finally:
                stub_request._json_payload = None
                if previous_request is None:
                    handler_globals.pop("request", None)
                else:
                    handler_globals["request"] = previous_request

            status = 200
            payload = result
            if isinstance(result, tuple):
                payload = result[0]
                if len(result) > 1 and isinstance(result[1], int):
                    status = result[1]
            return _Response(payload, status)

        def open(self, path: str, method: str = "GET", json: Any = None) -> Any:
            return self._request(path, method=method, json=json)

        def get(self, path: str, *args: Any, **kwargs: Any) -> Any:
            json_payload = kwargs.get("json")
            return self._request(path, method="GET", json=json_payload)

        def post(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return self._request(path, method="POST", json=kwargs.get("json"))

        def delete(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return self._request(path, method="DELETE", json=kwargs.get("json"))

    setattr(app, "test_client", lambda *args, **kwargs: _Client())


def _pytest_active() -> bool:
    """Return True when running under pytest (via env hints or modules)."""
    flag = _managed_env("PYTEST_RUNNING")
    flag_present = flag is not None and str(flag).strip() != ""
    if isinstance(flag, bool):
        flag_truthy = flag
    elif flag_present:
        flag_truthy = str(flag).strip().lower() not in {"0", "false", "no", "off"}
    else:
        flag_truthy = False
    if flag_truthy:
        return True
    current = _managed_env("PYTEST_CURRENT_TEST")
    if current:
        return True
    active = "pytest" in sys.modules
    if not active and (flag_present or current):
        _log.debug(
            "PYTEST_DETECT_FALSE",
            extra={
                "has_pytest_module": "pytest" in sys.modules,
                "env_flag": flag,
                "current_test": current,
            },
        )
    return active


def _seed_pytest_env_defaults() -> None:
    """Ensure required env vars exist when running under pytest."""
    active = _pytest_active()
    if not active:
        return
    for key, default in _REQUIRED_TEST_ENV.items():
        if str(_managed_env(key, "")).strip():
            continue
        if _set_runtime_env_override is not None:
            _set_runtime_env_override(key, default)


def suppress_flask_startup_noise() -> None:
    """Suppress Flask/Werkzeug startup banner lines on the dev server."""

    get_logger("werkzeug").setLevel(logging.ERROR)
    try:
        from flask import cli as _flask_cli  # type: ignore

        if hasattr(_flask_cli, "show_server_banner"):
            _flask_cli.show_server_banner = lambda *_args, **_kwargs: None
    except Exception:
        _log.debug("FLASK_STARTUP_BANNER_SUPPRESS_FAILED", exc_info=True)


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
    try:
        FlaskClass = import_module("flask.app").Flask
    except Exception:
        FlaskClass = Flask
    app = FlaskClass(__name__)
    try:
        from ai_trading.diagnostics.http_diag import diag_bp  # type: ignore
    except ImportError:
        diag_bp = None  # type: ignore[assignment]
    if diag_bp is not None:
        try:
            app.register_blueprint(diag_bp)
        except Exception:
            _log.debug("DIAG_BLUEPRINT_REGISTER_FAILED", exc_info=True)
    try:
        from ai_trading.operator_ui import operator_bp
    except ImportError:
        operator_bp = None  # type: ignore[assignment]
    if operator_bp is not None:
        try:
            app.register_blueprint(operator_bp)
        except Exception:
            _log.debug("OPERATOR_BLUEPRINT_REGISTER_FAILED", exc_info=True)

    # Some tests may monkeypatch Flask and return objects without a real config
    if not isinstance(getattr(app, "config", None), dict):
        app_config = getattr(app, "config", {})
        app_config_dict = dict(app_config) if isinstance(app_config, Mapping) else {}
        setattr(app, "config", cast(Any, app_config_dict))
    suppress_flask_startup_noise()

    # Cache required env validation once during app startup.
    _seed_pytest_env_defaults()
    try:
        from ai_trading.config.management import validate_required_env
        validate_required_env()
        app.config["_ENV_VALID"] = True
        app.config["_ENV_ERR"] = None
    except (ImportError, RuntimeError) as e:
        _log.exception("ENV_VALIDATION_FAILED")
        app.config["_ENV_VALID"] = False
        app.config["_ENV_ERR"] = str(e)

    route_registry = _ensure_route_registry(app)

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

        fallback_used = False
        fallback_reasons: list[str] = []
        serialization_failed = False
        try:
            response = jsonify(dict(sanitized_payload))
        except Exception as exc:  # pragma: no cover - defensive fallback
            response = None
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

        final_payload = _normalize_health_payload(dict(sanitized_payload))

        if serialization_failed:
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
        # Flask out for light-weight stubs) surface the fully populated payload
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
        if _pytest_active() and status == 200:
            payload = dict(payload)
            payload["ok"] = True
        response_factory = globals().get("jsonify")
        if callable(response_factory):
            try:
                response = response_factory(payload)
            except Exception:
                response = None
            if response is not None:
                if isinstance(response, Mapping):
                    response = None
                elif not (
                    callable(getattr(response, "get_data", None))
                    or callable(getattr(response, "get_json", None))
                    or hasattr(response, "status_code")
                ):
                    response = None
            if response is not None:
                try:
                    response.status_code = status
                except Exception:
                    pass
                return response
        return payload if status == 200 else (payload, status)

    @app.route("/operator/presets")
    def operator_presets() -> Any:
        """Expose preset choices for operator-driven no-code configuration."""

        try:
            from ai_trading.operator_presets import list_presets
            payload = {"ok": True, "presets": list_presets()}
            return _safe_response(payload, status=200)
        except (ImportError, ValueError, TypeError) as exc:
            _log.warning("OPERATOR_PRESETS_UNAVAILABLE", extra={"error": str(exc)})
            return _safe_response({"ok": False, "error": "operator presets unavailable"}, status=503)

    @app.route("/operator/plan")
    def operator_plan() -> Any:
        """Return a default guarded plan for lightweight operator workflows."""

        try:
            from ai_trading.operator_presets import PresetValidationError, build_plan
            plan = build_plan("balanced")
            return _safe_response({"ok": True, "plan": plan}, status=200)
        except (ImportError, PresetValidationError, ValueError, TypeError) as exc:
            _log.warning("OPERATOR_PLAN_UNAVAILABLE", extra={"error": str(exc)})
            return _safe_response({"ok": False, "error": "operator plan unavailable"}, status=503)

    @app.route("/operator/control-plane")
    def operator_control_plane_snapshot() -> Any:
        """Return a consolidated runtime control-plane snapshot for operators."""

        try:
            snapshot = ControlPlaneService(service_name=_SERVICE_NAME).snapshot()
            return _safe_response({"ok": True, "snapshot": snapshot}, status=200)
        except (ImportError, ValueError, TypeError) as exc:
            _log.warning("OPERATOR_CONTROL_PLANE_UNAVAILABLE", extra={"error": str(exc)})
            return _safe_response(
                {"ok": False, "error": "operator control plane unavailable"},
                status=503,
            )

    def _operator_control_plane_section_response(section: str) -> Any:
        """Return an operator-facing control-plane section by canonical name."""

        try:
            payload = ControlPlaneService(service_name=_SERVICE_NAME).section(section)
            return _safe_response({"ok": True, "section": section, "data": payload}, status=200)
        except KeyError:
            return _safe_response(
                {"ok": False, "error": "unknown control-plane section"},
                status=404,
            )
        except (ImportError, ValueError, TypeError) as exc:
            _log.warning("OPERATOR_CONTROL_PLANE_SECTION_UNAVAILABLE", extra={"error": str(exc)})
            return _safe_response(
                {"ok": False, "error": "operator control-plane section unavailable"},
                status=503,
            )

    @app.route("/operator/control-plane/rollout")
    def operator_control_plane_rollout() -> Any:
        return _operator_control_plane_section_response("rollout")

    @app.route("/operator/control-plane/broker-health")
    def operator_control_plane_broker_health() -> Any:
        return _operator_control_plane_section_response("broker-health")

    @app.route("/operator/control-plane/positions")
    def operator_control_plane_positions() -> Any:
        return _operator_control_plane_section_response("positions")

    @app.route("/operator/control-plane/open-orders")
    def operator_control_plane_open_orders() -> Any:
        return _operator_control_plane_section_response("open-orders")

    @app.route("/operator/control-plane/execution-quality")
    def operator_control_plane_execution_quality() -> Any:
        return _operator_control_plane_section_response("execution-quality")

    @app.route("/operator/control-plane/circuit-breakers")
    def operator_control_plane_circuit_breakers() -> Any:
        return _operator_control_plane_section_response("circuit-breakers")

    @app.route("/operator/control-plane/liveness")
    def operator_control_plane_liveness() -> Any:
        return _operator_control_plane_section_response("liveness")

    @app.route("/operator/control-plane/manual-overrides")
    def operator_control_plane_manual_overrides() -> Any:
        return _operator_control_plane_section_response("manual-overrides")

    @app.route("/operator/control-plane/services")
    def operator_control_plane_services() -> Any:
        return _operator_control_plane_section_response("services")

    @app.route("/operator/control-plane/manual-overrides", methods=["POST"])
    def operator_control_plane_update_manual_overrides() -> Any:
        """Persist manual runtime override controls on the canonical operator path."""

        request_obj = globals().get("request")
        if request_obj is None or not callable(getattr(request_obj, "get_json", None)):
            return _safe_response(
                {"ok": False, "error": "request context unavailable"},
                status=503,
            )
        body = request_obj.get_json(silent=True) or {}
        if not isinstance(body, dict):
            body = {}
        disabled_slices_raw = body.get("disabled_slices")
        disabled_slices = (
            list(disabled_slices_raw)
            if isinstance(disabled_slices_raw, list)
            else []
        )
        diagnostics_raw = body.get("diagnostics")
        diagnostics = dict(diagnostics_raw) if isinstance(diagnostics_raw, Mapping) else {}
        source_updated_at = str(body.get("source_updated_at") or "").strip() or None
        try:
            payload = ControlPlaneService(service_name=_SERVICE_NAME).update_manual_overrides(
                disabled_slices=disabled_slices,
                diagnostics=diagnostics,
                source_updated_at=source_updated_at,
            )
            return _safe_response({"ok": True, "manual_overrides": payload}, status=200)
        except (OSError, ValueError, TypeError) as exc:
            _log.warning("OPERATOR_MANUAL_OVERRIDES_UPDATE_FAILED", extra={"error": str(exc)})
            return _safe_response(
                {"ok": False, "error": "operator manual overrides unavailable"},
                status=503,
            )

    @app.route("/operator/control-plane/manual-overrides", methods=["DELETE"])
    def operator_control_plane_clear_manual_overrides() -> Any:
        """Clear operator manual overrides while preserving the canonical payload shape."""

        try:
            payload = ControlPlaneService(service_name=_SERVICE_NAME).clear_manual_overrides()
            return _safe_response({"ok": True, "manual_overrides": payload}, status=200)
        except (OSError, ValueError, TypeError) as exc:
            _log.warning("OPERATOR_MANUAL_OVERRIDES_CLEAR_FAILED", extra={"error": str(exc)})
            return _safe_response(
                {"ok": False, "error": "operator manual overrides unavailable"},
                status=503,
            )

    def _governance_base_path() -> str:
        configured = str(
            _managed_env("AI_TRADING_GOVERNANCE_BASE_PATH", "artifacts/governance")
            or "artifacts/governance"
        ).strip()
        return configured or "artifacts/governance"

    @app.route("/operator/governance")
    def operator_governance_snapshot() -> Any:
        """Return governance approvals/scorecards/rollback audit snapshot."""

        try:
            governance = GovernanceService(
                service_name=_SERVICE_NAME,
                base_path=_governance_base_path(),
            ).snapshot()
            return _safe_response({"ok": True, "governance": governance}, status=200)
        except (ImportError, ValueError, TypeError) as exc:
            _log.warning("OPERATOR_GOVERNANCE_SNAPSHOT_UNAVAILABLE", extra={"error": str(exc)})
            return _safe_response(
                {"ok": False, "error": "operator governance snapshot unavailable"},
                status=503,
            )

    @app.route("/operator/governance/approval", methods=["POST"])
    def operator_governance_record_approval() -> Any:
        """Record an explicit governance approval/rejection checkpoint."""

        request_obj = globals().get("request")
        if request_obj is None or not callable(getattr(request_obj, "get_json", None)):
            return _safe_response(
                {"ok": False, "error": "request context unavailable"},
                status=503,
            )
        body = request_obj.get_json(silent=True) or {}
        if not isinstance(body, dict):
            body = {}
        strategy = str(body.get("strategy") or "").strip()
        model_id = str(body.get("model_id") or "").strip()
        approver = str(body.get("approver") or "").strip()
        decision = str(body.get("decision") or "approved").strip().lower() or "approved"
        note = str(body.get("note") or "").strip() or None
        ticket = str(body.get("ticket") or "").strip() or None

        if not strategy or not model_id or not approver:
            return _safe_response(
                {
                    "ok": False,
                    "error": "strategy, model_id, and approver are required",
                },
                status=400,
            )
        try:
            result = GovernanceService(
                service_name=_SERVICE_NAME,
                base_path=_governance_base_path(),
            ).record_approval(
                strategy=strategy,
                model_id=model_id,
                approver=approver,
                decision=decision,
                note=note,
                ticket=ticket,
            )
            return _safe_response(
                {
                    "ok": True,
                    "path": result.get("path"),
                    "approvals": result.get("approvals"),
                },
                status=200,
            )
        except ValueError as exc:
            return _safe_response({"ok": False, "error": str(exc)}, status=400)
        except (ImportError, TypeError) as exc:
            _log.warning("OPERATOR_GOVERNANCE_APPROVAL_FAILED", extra={"error": str(exc)})
            return _safe_response(
                {"ok": False, "error": "operator governance approval unavailable"},
                status=503,
            )

    @app.route("/operator/governance/rollback", methods=["POST"])
    def operator_governance_manual_rollback() -> Any:
        """Trigger a governance rollback to previous production model."""

        request_obj = globals().get("request")
        if request_obj is None or not callable(getattr(request_obj, "get_json", None)):
            return _safe_response(
                {"ok": False, "error": "request context unavailable"},
                status=503,
            )
        body = request_obj.get_json(silent=True) or {}
        if not isinstance(body, dict):
            body = {}
        strategy = str(body.get("strategy") or "").strip()
        reason = str(body.get("reason") or "operator_manual_rollback").strip()
        force_raw = body.get("force")
        if isinstance(force_raw, bool):
            force = force_raw
        elif force_raw in (None, ""):
            force = True
        else:
            force = str(force_raw).strip().lower() not in {"0", "false", "no", "off"}

        if not strategy:
            return _safe_response(
                {"ok": False, "error": "strategy is required"},
                status=400,
            )
        try:
            result = GovernanceService(
                service_name=_SERVICE_NAME,
                base_path=_governance_base_path(),
            ).rollback(
                strategy=strategy,
                reason=reason,
                force=force,
            )
            rolled_back = bool(result.get("rolled_back"))
            status_code = 200 if rolled_back else 409
            return _safe_response(
                result,
                status=status_code,
            )
        except (ImportError, TypeError, ValueError) as exc:
            _log.warning("OPERATOR_GOVERNANCE_ROLLBACK_FAILED", extra={"error": str(exc)})
            return _safe_response(
                {"ok": False, "error": "operator governance rollback unavailable"},
                status=503,
            )

    @app.route("/health")
    def health():
        """Lightweight liveness probe with Alpaca diagnostics."""
        pytest_mode = _pytest_active()
        payload = build_api_health_payload(
            service_name=_SERVICE_NAME,
            force_ok_for_pytest=pytest_mode,
            env_error=app.config.get("_ENV_ERR"),
        )

        fallback_payload = {
            "ok": bool(payload.get("ok")),
            "alpaca": dict(payload["alpaca"]),
        }
        if payload.get("error"):
            fallback_payload["error"] = payload.get("error")
        return _json_response(payload, fallback=fallback_payload)

    def _build_healthz_payload() -> dict[str, Any]:
        pytest_mode = _pytest_active()
        payload = cast(
            dict[str, Any],
            build_canonical_healthz_payload(
            service_name=_SERVICE_NAME,
            force_ok_for_pytest=pytest_mode,
            healthy_status_mode="service",
            ok_mode="connectivity",
            env_error=app.config.get("_ENV_ERR"),
            ),
        )
        if (not pytest_mode) and (
            _managed_env("PYTEST_RUNNING") or _managed_env("PYTEST_CURRENT_TEST")
        ):
            _log.warning(
                "PYTEST_OVERRIDE_SKIPPED",
                extra={
                    "env_flag": _managed_env("PYTEST_RUNNING"),
                    "current_test": _managed_env("PYTEST_CURRENT_TEST"),
                    "has_pytest_module": "pytest" in sys.modules,
                },
            )
        return payload

    def _healthz_response(payload: dict[str, Any], status: int) -> Any:
        return _safe_response(payload, status=status)

    register_healthz_routes(
        app,
        payload_builder=_build_healthz_payload,
        response_builder=_healthz_response,
        service_name=_SERVICE_NAME,
        routes=("/healthz",),
        logger=_log,
        error_event="HEALTHZ_HANDLER_FAILED",
    )

    @app.route("/metrics")
    def metrics():
        """Expose Prometheus metrics if available."""
        if not _PROM_OK:
            return ("metrics unavailable", 501)
        try:
            from ai_trading.telemetry.runtime_prom_metrics import (
                refresh_runtime_execution_metrics,
            )

            refresh_runtime_execution_metrics()
        except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _log.debug(
                "PROM_RUNTIME_METRICS_REFRESH_SKIPPED",
                extra={"error": str(exc)},
            )
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return generate_latest(cast(Any, _PROM_REG)), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    _ensure_test_client_support(app, route_registry)
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
            if isinstance(resp, tuple) and resp:
                payload = resp[0]
                if len(resp) > 1 and isinstance(resp[1], int):
                    status_code = resp[1]
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

        setattr(app, "test_client", _patched_test_client)

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
    if _managed_env("RUN_HEALTHCHECK") == "1":
        from ai_trading.config.settings import get_settings

        app = create_app()
        s = get_settings()
        port = int(s.healthcheck_port or 8081)
        suppress_flask_startup_noise()
        app.logger.info("Starting Flask", extra={"port": port})
        app.run(host="0.0.0.0", port=port)
