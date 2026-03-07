from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable, Mapping

from ai_trading.telemetry import runtime_state


def _safe_observe(observer: Callable[[], Any], default: Any) -> Any:
    try:
        return observer()
    except Exception:
        return default


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
        "degraded" if provider_state.get("using_backup") else "healthy"
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
    primary_name = str(provider_payload.get("primary") or "").strip().lower()
    active_name = str(provider_payload.get("active") or "").strip().lower()
    try:
        consecutive_failures = int(provider_payload.get("consecutive_failures") or 0)
    except (TypeError, ValueError):
        consecutive_failures = 0
    provider_unknown = provider_status_normalized in {"", "unknown"}
    provider_primary_steady = (
        not provider_payload.get("using_backup")
        and (not primary_name or not active_name or primary_name == active_name)
        and consecutive_failures <= 0
        and not provider_payload.get("last_error_at")
    )
    if provider_unknown and provider_primary_steady:
        provider_status = "healthy"
        provider_status_normalized = "healthy"
        provider_payload["status"] = "healthy"

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
    broker_payload = {
        "status": broker_status,
        "connected": bool(broker_connected_raw),
        "latency_ms": broker_state.get("latency_ms"),
        "last_error": broker_state.get("last_error"),
        "last_order_ack_ms": broker_state.get("last_order_ack_ms"),
    }

    service_status = service_state.get("status", "unknown")
    service_reason = service_state.get("reason")
    provider_disabled = provider_status_normalized in {"down", "disabled"}
    broker_down = broker_status_normalized in {"unreachable", "down", "failed"}
    data_degraded = data_status_normalized in {"empty", "degraded"}

    degraded = provider_disabled or provider_payload.get("using_backup") or (
        provider_status_normalized not in {"", "healthy", "ready"}
    )
    if broker_down:
        degraded = True
    if data_degraded:
        degraded = True

    provider_healthy = provider_status_normalized in {"", "healthy", "ready"} and not data_degraded
    broker_healthy = broker_status_normalized in {"", "reachable", "ready", "connected"}
    overall_ok = provider_healthy and broker_healthy
    if str(ok_mode).strip().lower() == "connectivity":
        overall_ok = not provider_disabled and not broker_down
    if force_ok_for_pytest:
        overall_ok = True

    if degraded:
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
    if service_reason:
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
    if force_ok_for_pytest:
        payload["ok"] = True
        payload.setdefault("status", payload.get("status") or "healthy")
    return payload


__all__ = ["build_runtime_health_payload", "build_alpaca_health_payload"]
