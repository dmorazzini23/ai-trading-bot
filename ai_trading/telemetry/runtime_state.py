from __future__ import annotations

"""Lightweight runtime telemetry state shared across health surfaces."""

from datetime import UTC, datetime
from threading import RLock
from typing import Any

__all__ = [
    "update_data_provider_state",
    "observe_data_provider_state",
    "reset_data_provider_state",
    "update_quote_status",
    "observe_quote_status",
    "reset_quote_status",
    "update_broker_status",
    "observe_broker_status",
    "reset_broker_status",
    "update_service_status",
    "observe_service_status",
    "reset_service_status",
    "reset_all_states",
]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


_LOCK = RLock()
_DEFAULT_PROVIDER_STATE: dict[str, Any] = {
    "primary": "alpaca",
    "active": "alpaca",
    "backup": None,
    "using_backup": False,
    "reason": None,
    "data_status": None,
    "updated": None,
    "status": "unknown",
    "consecutive_failures": 0,
    "last_error_at": None,
    "http_code": None,
    "gap_ratio_recent": None,
    "quote_fresh_ms": None,
    "safe_mode": False,
    "timeframes": {},
}
_DEFAULT_QUOTE_STATE: dict[str, Any] = {
    "status": "unknown",
    "allowed": False,
    "reason": None,
    "age_sec": None,
    "synthetic": False,
    "bid": None,
    "ask": None,
    "updated": None,
    "source": None,
    "last_price": None,
    "quote_age_ms": None,
}
_DEFAULT_BROKER_STATE: dict[str, Any] = {
    "connected": False,
    "latency_ms": None,
    "last_error": None,
    "updated": None,
    "status": "unknown",
    "last_order_ack_ms": None,
}

_provider_state: dict[str, Any] = dict(_DEFAULT_PROVIDER_STATE)
_quote_status: dict[str, Any] = dict(_DEFAULT_QUOTE_STATE)
_broker_status: dict[str, Any] = dict(_DEFAULT_BROKER_STATE)
_service_status: dict[str, Any] = {
    "status": "unknown",
    "phase": "unknown",
    "phase_since": _now_iso(),
    "updated": _now_iso(),
}


def _merge_state(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(target)
    merged.update({k: v for k, v in updates.items() if v is not None or k not in merged})
    merged["updated"] = _now_iso()
    return merged


def update_data_provider_state(
    *,
    primary: str | None = None,
    active: str | None = None,
    backup: str | None = None,
    using_backup: bool | None = None,
    reason: str | None = None,
    cooldown_sec: float | None = None,
    status: str | None = None,
    consecutive_failures: int | None = None,
    last_error_at: str | None = None,
    http_code: int | None = None,
    timeframe: str | None = None,
    gap_ratio_recent: float | None = None,
    quote_fresh_ms: float | None = None,
    safe_mode: bool | None = None,
    data_status: str | None = None,
) -> None:
    """Record current data provider routing."""

    updates: dict[str, Any] = {}
    if primary is not None:
        updates["primary"] = str(primary)
    if active is not None:
        updates["active"] = str(active)
    if backup is not None:
        updates["backup"] = str(backup)
    if using_backup is not None:
        updates["using_backup"] = bool(using_backup)
    if reason is not None:
        updates["reason"] = reason
    if cooldown_sec is not None:
        try:
            updates["cooldown_sec"] = max(0.0, float(cooldown_sec))
        except (TypeError, ValueError):
            pass
    if status is not None:
        updates["status"] = status
    if consecutive_failures is not None:
        try:
            updates["consecutive_failures"] = max(0, int(consecutive_failures))
        except (TypeError, ValueError):
            pass
    if last_error_at is not None:
        updates["last_error_at"] = last_error_at
    if http_code is not None:
        try:
            updates["http_code"] = int(http_code)
        except (TypeError, ValueError):
            pass
    if gap_ratio_recent is not None:
        try:
            updates["gap_ratio_recent"] = max(0.0, float(gap_ratio_recent))
        except (TypeError, ValueError):
            pass
    if quote_fresh_ms is not None:
        try:
            updates["quote_fresh_ms"] = max(0.0, float(quote_fresh_ms))
        except (TypeError, ValueError):
            pass
    if safe_mode is not None:
        updates["safe_mode"] = bool(safe_mode)
    if data_status is not None:
        try:
            updates["data_status"] = str(data_status)
        except Exception:
            updates["data_status"] = data_status
    timeframe_key: str | None = None
    if timeframe is not None:
        try:
            candidate = str(timeframe).strip()
        except Exception:
            candidate = str(timeframe)
        timeframe_key = candidate or None

    with _LOCK:
        global _provider_state
        if timeframe_key:
            current = dict(_provider_state.get("timeframes") or {})
            if using_backup is not None:
                current[timeframe_key] = bool(using_backup)
            elif timeframe_key not in current:
                current[timeframe_key] = False
            updates["timeframes"] = current
        _provider_state = _merge_state(_provider_state, updates)


def observe_data_provider_state() -> dict[str, Any]:
    with _LOCK:
        return dict(_provider_state)


def reset_data_provider_state() -> None:
    """Reset provider telemetry to defaults."""

    with _LOCK:
        global _provider_state
        _provider_state = dict(_DEFAULT_PROVIDER_STATE)


def update_service_status(
    *,
    status: str,
    reason: str | None = None,
    phase: str | None = None,
    cycle_index: int | None = None,
) -> None:
    updates: dict[str, Any] = {"status": str(status or "unknown")}
    if reason is not None:
        updates["reason"] = reason
    phase_token: str | None = None
    if phase is not None:
        phase_token = str(phase or "").strip().lower() or "unknown"
        updates["phase"] = phase_token
    if cycle_index is not None:
        try:
            updates["cycle_index"] = max(0, int(cycle_index))
        except (TypeError, ValueError):
            pass
    with _LOCK:
        global _service_status
        if phase_token is not None:
            previous_phase = str(_service_status.get("phase") or "").strip().lower()
            if phase_token != previous_phase:
                updates["phase_since"] = _now_iso()
        _service_status = _merge_state(_service_status, updates)


def observe_service_status() -> dict[str, Any]:
    with _LOCK:
        return dict(_service_status)


def reset_service_status() -> None:
    """Reset service telemetry to defaults."""

    with _LOCK:
        global _service_status
        now_iso = _now_iso()
        _service_status = {
            "status": "unknown",
            "phase": "unknown",
            "phase_since": now_iso,
            "updated": now_iso,
        }


def update_quote_status(
    *,
    allowed: bool,
    reason: str | None = None,
    age_sec: float | None = None,
    synthetic: bool | None = None,
    bid: float | None = None,
    ask: float | None = None,
    status: str | None = None,
    source: str | None = None,
    last_price: float | None = None,
    quote_age_ms: float | None = None,
) -> None:
    """Update snapshot of the most recent quote gate decision."""

    updates: dict[str, Any] = {
        "allowed": bool(allowed),
    }
    if status is not None:
        updates["status"] = status
    if reason is not None:
        updates["reason"] = reason
    if age_sec is not None:
        try:
            updates["age_sec"] = max(0.0, float(age_sec))
        except (TypeError, ValueError):
            pass
    if synthetic is not None:
        updates["synthetic"] = bool(synthetic)
    if bid is not None:
        updates["bid"] = bid
    if ask is not None:
        updates["ask"] = ask
    if source is not None:
        updates["source"] = source
    if last_price is not None:
        updates["last_price"] = last_price
    if quote_age_ms is not None:
        try:
            updates["quote_age_ms"] = max(0.0, float(quote_age_ms))
        except (TypeError, ValueError):
            pass
    with _LOCK:
        global _quote_status
        _quote_status = _merge_state(_quote_status, updates)


def observe_quote_status() -> dict[str, Any]:
    with _LOCK:
        return dict(_quote_status)


def reset_quote_status() -> None:
    """Reset quote telemetry to defaults."""

    with _LOCK:
        global _quote_status
        _quote_status = dict(_DEFAULT_QUOTE_STATE)


def update_broker_status(
    *,
    connected: bool | None = None,
    latency_ms: float | None = None,
    last_error: str | None = None,
    status: str | None = None,
    last_order_ack_ms: float | None = None,
) -> None:
    """Record recent broker connectivity observations."""

    updates: dict[str, Any] = {}
    if connected is not None:
        updates["connected"] = bool(connected)
    if latency_ms is not None:
        try:
            updates["latency_ms"] = max(0.0, float(latency_ms))
        except (TypeError, ValueError):
            pass
    if last_error is not None:
        updates["last_error"] = last_error
    if status is not None:
        updates["status"] = status
    if last_order_ack_ms is not None:
        try:
            updates["last_order_ack_ms"] = max(0.0, float(last_order_ack_ms))
        except (TypeError, ValueError):
            pass
    with _LOCK:
        global _broker_status
        _broker_status = _merge_state(_broker_status, updates)


def observe_broker_status() -> dict[str, Any]:
    with _LOCK:
        return dict(_broker_status)


def reset_broker_status() -> None:
    """Reset broker telemetry to defaults."""

    with _LOCK:
        global _broker_status
        _broker_status = dict(_DEFAULT_BROKER_STATE)


def reset_all_states() -> None:
    """Reset all runtime telemetry snapshots."""

    with _LOCK:
        global _provider_state, _quote_status, _broker_status, _service_status
        _provider_state = dict(_DEFAULT_PROVIDER_STATE)
        _quote_status = dict(_DEFAULT_QUOTE_STATE)
        _broker_status = dict(_DEFAULT_BROKER_STATE)
        now_iso = _now_iso()
        _service_status = {
            "status": "unknown",
            "phase": "unknown",
            "phase_since": now_iso,
            "updated": now_iso,
        }
