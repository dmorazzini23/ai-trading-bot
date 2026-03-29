from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable, Mapping

from ai_trading.telemetry import runtime_state


def _safe_observe(observer: Callable[[], Any], default: Any) -> Any:
    try:
        return observer()
    except Exception:
        return default


def _timestamp_age_seconds(raw_value: Any) -> float | None:
    if raw_value in (None, ""):
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return max((datetime.now(UTC) - parsed).total_seconds(), 0.0)


def _env_bool(name: str, default: bool) -> bool:
    try:
        from ai_trading.config.management import get_env

        return bool(get_env(name, default, cast=bool))
    except Exception:
        return bool(default)


def _env_float(name: str, default: float) -> float:
    try:
        from ai_trading.config.management import get_env

        raw = get_env(name, default, cast=float)
        return float(raw if raw is not None else default)
    except Exception:
        return float(default)


def _market_is_closed_now() -> bool:
    try:
        from ai_trading.utils.base import is_market_open as _is_market_open_base

        return not bool(_is_market_open_base())
    except Exception:
        return False


def _model_liveness_snapshot() -> dict[str, Any]:
    try:
        from ai_trading.monitoring.model_liveness import (
            get_model_liveness_snapshot,
        )

        snapshot = get_model_liveness_snapshot()
        if isinstance(snapshot, dict):
            return snapshot
    except Exception:
        pass
    return {}


def build_alpaca_health_payload(
    context: Mapping[str, Any] | None = None,
    *,
    enrich_from_runtime_env: bool = True,
) -> dict[str, Any]:
    """Build a normalized Alpaca diagnostic section for health payloads."""

    payload: dict[str, Any] = {
        "sdk_ok": False,
        "initialized": False,
        "client_attached": False,
        "has_key": False,
        "has_secret": False,
        "base_url": "",
        "paper": False,
        "shadow_mode": False,
    }
    if isinstance(context, Mapping):
        for key, value in context.items():
            if key not in payload:
                continue
            if isinstance(payload[key], bool):
                payload[key] = bool(value)
            elif value is not None:
                payload[key] = value

    if enrich_from_runtime_env:
        try:
            from ai_trading.utils.env import alpaca_credential_status, get_alpaca_base_url

            has_key, has_secret = alpaca_credential_status()
            payload["has_key"] = bool(payload["has_key"] or has_key)
            payload["has_secret"] = bool(payload["has_secret"] or has_secret)
            if not payload.get("base_url"):
                payload["base_url"] = str(get_alpaca_base_url() or "")
        except Exception:
            pass

        try:
            from ai_trading.alpaca_api import ALPACA_AVAILABLE as alpaca_sdk_ok

            payload["sdk_ok"] = bool(payload["sdk_ok"] or alpaca_sdk_ok)
        except Exception:
            pass

    if payload.get("base_url"):
        payload["paper"] = bool(payload["paper"] or ("paper" in str(payload["base_url"]).lower()))

    return payload


def build_runtime_health_payload(
    *,
    service_name: str = "ai-trading",
    force_ok_for_pytest: bool = False,
    healthy_status_mode: str = "service",
    ok_mode: str = "strict",
) -> dict[str, Any]:
    """Build a normalized runtime health payload shared by API/health surfaces."""

    provider_state: dict[str, Any] = _safe_observe(
        runtime_state.observe_data_provider_state,
        {},
    )
    broker_state: dict[str, Any] = _safe_observe(runtime_state.observe_broker_status, {})
    service_state: dict[str, Any] = _safe_observe(
        runtime_state.observe_service_status,
        {"status": "unknown"},
    )
    quote_state: dict[str, Any] = _safe_observe(runtime_state.observe_quote_status, {})
    model_liveness = _model_liveness_snapshot()

    raw_provider_status = provider_state.get("status")
    provider_status = raw_provider_status or (
        "degraded" if provider_state.get("using_backup") else "unknown"
    )
    provider_status_normalized = str(provider_status or "").strip().lower()
    data_status = provider_state.get("data_status")
    data_status_normalized = str(data_status or "").strip().lower()
    gap_ratio_recent = provider_state.get("gap_ratio_recent")
    gap_ratio_pct: float | None = None
    if gap_ratio_recent is not None:
        try:
            gap_ratio_pct = float(gap_ratio_recent) * 100.0
        except (TypeError, ValueError):
            gap_ratio_pct = None

    provider_payload: dict[str, Any] = {
        "status": provider_status,
        "reason": provider_state.get("reason"),
        "reason_code": provider_state.get("reason_code"),
        "reason_detail": provider_state.get("reason_detail"),
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
        "data_status": data_status,
    }
    broker_connected_raw = broker_state.get("connected")
    broker_status = broker_state.get("status")
    if not broker_status:
        if broker_connected_raw is None:
            broker_status = "unknown"
        else:
            broker_status = "reachable" if bool(broker_connected_raw) else "unreachable"
    broker_status_normalized = str(broker_status or "").strip().lower()
    broker_connected: bool | None
    if broker_connected_raw is None:
        broker_connected = None
    else:
        broker_connected = bool(broker_connected_raw)
    broker_payload = {
        "status": broker_status,
        "connected": broker_connected,
        "latency_ms": broker_state.get("latency_ms"),
        "last_error": broker_state.get("last_error"),
        "last_order_ack_ms": broker_state.get("last_order_ack_ms"),
    }

    service_status = service_state.get("status", "unknown")
    service_reason = service_state.get("reason")
    provider_reason_normalized = str(provider_state.get("reason") or "").strip().lower()
    service_reason_normalized = str(service_reason or "").strip().lower()
    service_phase_normalized = str(service_state.get("phase") or "").strip().lower()
    service_phase_age_s = _timestamp_age_seconds(service_state.get("phase_since"))
    market_closed_mode = (
        provider_reason_normalized == "market_closed"
        or service_reason_normalized == "market_closed"
    )
    provider_disabled = provider_status_normalized in {"down", "disabled", "failed", "unreachable"}
    provider_unknown = provider_status_normalized in {"", "unknown"}
    broker_down = broker_status_normalized in {"unreachable", "down", "failed"}
    broker_degraded = broker_status_normalized in {"degraded"}
    broker_unknown = broker_status_normalized in {"", "unknown"}
    data_degraded = data_status_normalized in {"empty", "degraded"}

    degraded = provider_disabled or provider_payload.get("using_backup") or (
        provider_status_normalized not in {"healthy", "ready"}
    )
    if broker_down or broker_degraded:
        degraded = True
    if provider_unknown or broker_unknown:
        degraded = True
    if data_degraded:
        degraded = True

    provider_healthy = provider_status_normalized in {"healthy", "ready"} and not data_degraded
    broker_healthy = broker_status_normalized in {"reachable", "ready", "connected"}
    overall_ok = provider_healthy and broker_healthy
    if str(ok_mode).strip().lower() == "connectivity":
        provider_connectivity_ok = (
            provider_status_normalized in {"healthy", "ready", "degraded"}
            and not data_degraded
        )
        broker_connectivity_ok = broker_status_normalized in {"reachable", "ready", "connected"}
        overall_ok = provider_connectivity_ok and broker_connectivity_ok
    offhours_market_closed_ready = (
        market_closed_mode
        and broker_healthy
        and not broker_down
        and not broker_degraded
        and not data_degraded
        and not provider_disabled
        and not bool(provider_payload.get("using_backup"))
    )
    warmup_fast_path_enabled = _env_bool(
        "AI_TRADING_HEALTH_WARMUP_MARKET_CLOSED_FASTPATH_ENABLED",
        True,
    )
    warmup_fast_path_max_age_s = max(
        30.0,
        min(
            _env_float(
                "AI_TRADING_HEALTH_WARMUP_MARKET_CLOSED_FASTPATH_MAX_AGE_SEC",
                420.0,
            ),
            3600.0,
        ),
    )
    warmup_market_closed_ready = (
        warmup_fast_path_enabled
        and service_phase_normalized == "warmup"
        and service_reason_normalized == "warmup_cycle"
        and (market_closed_mode or _market_is_closed_now())
        and not provider_disabled
        and not data_degraded
        and not bool(provider_payload.get("using_backup"))
        and not broker_down
        and not broker_degraded
        and (broker_healthy or broker_unknown)
        and (
            service_phase_age_s is None
            or float(service_phase_age_s) <= float(warmup_fast_path_max_age_s)
        )
    )
    if offhours_market_closed_ready:
        overall_ok = True
        degraded = False
    elif warmup_market_closed_ready:
        overall_ok = True
        degraded = False
    if force_ok_for_pytest:
        overall_ok = True
    if not overall_ok:
        degraded = True

    if offhours_market_closed_ready:
        resolved_status = "healthy"
    elif warmup_market_closed_ready:
        resolved_status = "healthy"
    elif degraded:
        resolved_status = "degraded"
    elif healthy_status_mode == "healthy":
        resolved_status = "healthy"
    else:
        resolved_status = service_status

    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "ok": overall_ok,
        "timestamp": timestamp,
        "service": service_name,
        "status": resolved_status,
        "data_provider": provider_payload,
        "broker": broker_payload,
        "broker_connectivity": broker_payload,
        "fallback_active": bool(provider_payload.get("using_backup")),
        "quotes_status": quote_state,
        "primary_data_provider": provider_payload,
        "gap_ratio_recent": provider_payload.get("gap_ratio_recent"),
        "gap_ratio_pct": gap_ratio_pct,
        "quote_fresh_ms": provider_payload.get("quote_fresh_ms"),
        "safe_mode": provider_payload.get("safe_mode"),
        "provider_state": provider_state,
        "cooldown_seconds_remaining": provider_payload.get("cooldown_seconds_remaining"),
        "data_status": data_status,
        "model_liveness": model_liveness,
    }
    if offhours_market_closed_ready:
        payload["reason"] = "market_closed"
    elif warmup_market_closed_ready:
        payload["reason"] = "market_closed"
    elif service_reason:
        payload.setdefault("reason", service_reason)
    degrade_reason = provider_payload.get("reason")
    if degraded and degrade_reason:
        payload.setdefault("reason", degrade_reason)
    if degraded and provider_payload.get("http_code") is not None:
        payload.setdefault("http_code", provider_payload.get("http_code"))
    if data_degraded and not payload.get("reason"):
        payload["reason"] = "data_unavailable"
    if broker_down and not payload.get("reason"):
        payload["reason"] = broker_state.get("last_error") or "broker_unreachable"
    if provider_unknown and not payload.get("reason"):
        payload["reason"] = "provider_status_unknown"
    if broker_unknown and not payload.get("reason"):
        payload["reason"] = "broker_status_unknown"
    if force_ok_for_pytest:
        payload["ok"] = True
        payload.setdefault("status", payload.get("status") or "healthy")
    return payload


def build_service_health_payload(
    *,
    service_name: str = "ai-trading",
    force_ok_for_pytest: bool = False,
    healthy_status_mode: str = "service",
    ok_mode: str = "connectivity",
    env_error: Any | None = None,
    alpaca_context: Mapping[str, Any] | None = None,
    enrich_alpaca_from_runtime_env: bool = True,
) -> dict[str, Any]:
    """Build canonical API/health-service payload shared by all health entrypoints."""

    payload = build_runtime_health_payload(
        service_name=service_name,
        force_ok_for_pytest=force_ok_for_pytest,
        healthy_status_mode=healthy_status_mode,
        ok_mode=ok_mode,
    )
    payload["alpaca"] = build_alpaca_health_payload(
        alpaca_context,
        enrich_from_runtime_env=enrich_alpaca_from_runtime_env,
    )

    env_error_text = str(env_error or "").strip()
    if env_error_text and not payload.get("reason"):
        payload["reason"] = env_error_text

    if force_ok_for_pytest:
        payload["ok"] = True
        payload.setdefault("status", payload.get("status") or "healthy")
    return payload


def build_api_health_payload(
    *,
    service_name: str = "ai-trading",
    force_ok_for_pytest: bool = False,
    env_error: Any | None = None,
) -> dict[str, Any]:
    """Build the canonical `/health` payload used by API entrypoints."""

    errors: list[str] = []
    alpaca_import_ok = True
    sdk_ok = False
    trading_client: Any = None
    key = secret = None
    base_url = ""
    paper = False
    shadow = False

    try:
        from ai_trading.alpaca_api import ALPACA_AVAILABLE as alpaca_sdk_available
    except Exception as exc:  # pragma: no cover - defensive
        alpaca_import_ok = False
        errors.append(str(exc) or exc.__class__.__name__)
        sdk_ok = False
    else:
        sdk_ok = bool(alpaca_sdk_available)

    if alpaca_import_ok:
        try:
            from ai_trading.core.bot_engine import _resolve_alpaca_env
            from ai_trading.core.bot_engine import trading_client as _trading_client

            trading_client = _trading_client
            key, secret, base_url = _resolve_alpaca_env()
            base_url = str(base_url or "")
            paper = bool(base_url and "paper" in base_url.lower())
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(str(exc) or exc.__class__.__name__)
            trading_client, key, secret, base_url, paper = (None, None, None, "", False)

    try:
        from ai_trading.config.management import is_shadow_mode

        shadow = bool(is_shadow_mode())
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(str(exc) or exc.__class__.__name__)
        shadow = False

    if (not alpaca_import_ok) or (not key) or (not secret):
        shadow = False

    payload = build_service_health_payload(
        service_name=service_name,
        force_ok_for_pytest=force_ok_for_pytest,
        healthy_status_mode="service",
        ok_mode="connectivity",
        env_error=env_error,
        alpaca_context={
            "sdk_ok": sdk_ok,
            "initialized": bool(trading_client),
            "client_attached": bool(trading_client),
            "has_key": bool(key),
            "has_secret": bool(secret),
            "base_url": base_url,
            "paper": paper,
            "shadow_mode": shadow,
        },
        enrich_alpaca_from_runtime_env=False,
    )
    env_error_text = str(env_error or "").strip()
    if env_error_text:
        errors.append(env_error_text)

    if errors:
        payload["ok"] = False
        payload["status"] = "degraded"
        payload["error"] = "; ".join(dict.fromkeys(errors))
    elif force_ok_for_pytest:
        payload["ok"] = True
        payload.setdefault("status", payload.get("status") or "healthy")
    return payload


def build_canonical_healthz_payload(
    *,
    service_name: str = "ai-trading",
    force_ok_for_pytest: bool = False,
    healthy_status_mode: str = "healthy",
    ok_mode: str = "connectivity",
    env_error: Any | None = None,
    alpaca_context: Mapping[str, Any] | None = None,
    enrich_alpaca_from_runtime_env: bool = True,
    error: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical ``/healthz`` payload shared across runtime entrypoints."""

    payload = build_service_health_payload(
        service_name=service_name,
        force_ok_for_pytest=force_ok_for_pytest,
        healthy_status_mode=healthy_status_mode,
        ok_mode=ok_mode,
        env_error=env_error,
        alpaca_context=alpaca_context,
        enrich_alpaca_from_runtime_env=enrich_alpaca_from_runtime_env,
    )
    error_text = str(error or "").strip()
    if error_text:
        payload["ok"] = False
        payload["status"] = "degraded"
        payload["error"] = error_text
    if extras:
        payload.update(dict(extras))
    return payload


def build_health_exception_payload(
    exc: Exception,
    *,
    service_name: str = "ai-trading",
) -> dict[str, Any]:
    """Build a normalized degraded payload for health handler exceptions."""

    return {
        "ok": False,
        "status": "degraded",
        "service": service_name,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "error": str(exc),
    }


def build_health_json_response(
    payload: Mapping[str, Any],
    status: int,
    *,
    jsonify_fn: Callable[[Mapping[str, Any]], Any],
) -> Any:
    """Build a robust Flask-compatible JSON response for health handlers."""

    response = jsonify_fn(dict(payload))
    if response is None or isinstance(response, Mapping):
        return dict(payload) if status == 200 else (dict(payload), status)
    if not (
        callable(getattr(response, "get_data", None))
        or callable(getattr(response, "get_json", None))
        or hasattr(response, "status_code")
    ):
        return dict(payload) if status == 200 else (dict(payload), status)
    try:
        response.status_code = status
    except Exception:
        pass
    return response


def register_healthz_routes(
    app: Any,
    *,
    payload_builder: Callable[[], Mapping[str, Any]],
    response_builder: Callable[[dict[str, Any], int], Any],
    service_name: str = "ai-trading",
    routes: tuple[str, ...] = ("/healthz",),
    methods: tuple[str, ...] = ("GET",),
    logger: Any | None = None,
    error_event: str = "HEALTH_CHECK_FAILED",
) -> None:
    """Register canonical health routes on ``app`` using shared runtime logic."""

    def _healthz_handler() -> Any:
        try:
            payload = dict(payload_builder())
            return response_builder(payload, 200)
        except Exception as exc:
            if logger is not None:
                try:
                    logger.exception(error_event, exc_info=exc)
                except Exception:
                    pass
            fallback_payload = build_health_exception_payload(exc, service_name=service_name)
            return response_builder(fallback_payload, 500)

    for route in routes:
        decorator = app.route(route, methods=list(methods))
        decorator(_healthz_handler)


__all__ = [
    "build_runtime_health_payload",
    "build_alpaca_health_payload",
    "build_service_health_payload",
    "build_api_health_payload",
    "build_canonical_healthz_payload",
    "build_health_exception_payload",
    "build_health_json_response",
    "register_healthz_routes",
]
