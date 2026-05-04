from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

"""Lightweight runtime telemetry state shared across health surfaces."""

from copy import deepcopy
from datetime import UTC, datetime
from threading import RLock
from typing import Any, TypeVar, cast

T = TypeVar("T")

__all__ = [
    "update_data_provider_state",
    "observe_data_provider_state",
    "reset_data_provider_state",
    "update_quote_status",
    "observe_quote_status",
    "observe_symbol_quote_status",
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
    "reason_code": None,
    "reason_detail": None,
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
    "quote_timestamp": None,
    "symbol": None,
    "spread_bps": None,
    "max_spread_bps": None,
    "max_quote_age_ms": None,
    "gate_reason": None,
}
_DEFAULT_BROKER_STATE: dict[str, Any] = {
    "connected": False,
    "latency_ms": None,
    "last_error": None,
    "updated": None,
    "status": "unknown",
    "last_order_ack_ms": None,
    "open_orders_count": None,
    "positions_count": None,
}
_DEFAULT_SERVICE_STATUS: dict[str, Any] = {
    "status": "unknown",
    "phase": "unknown",
    "phase_since": _now_iso(),
    "updated": _now_iso(),
}


def _clone_state(value: T) -> T:
    return cast(T, deepcopy(value))


def _fresh_provider_state() -> dict[str, Any]:
    return _clone_state(_DEFAULT_PROVIDER_STATE)


def _fresh_quote_state() -> dict[str, Any]:
    return _clone_state(_DEFAULT_QUOTE_STATE)


def _fresh_broker_state() -> dict[str, Any]:
    return _clone_state(_DEFAULT_BROKER_STATE)


def _fresh_service_status() -> dict[str, Any]:
    snapshot = _clone_state(_DEFAULT_SERVICE_STATUS)
    now_iso = _now_iso()
    snapshot["phase_since"] = now_iso
    snapshot["updated"] = now_iso
    return snapshot


_provider_state: dict[str, Any] = _fresh_provider_state()
_quote_status: dict[str, Any] = _fresh_quote_state()
_quote_status_by_symbol: dict[str, dict[str, Any]] = {}
_broker_status: dict[str, Any] = _fresh_broker_state()
_service_status: dict[str, Any] = _fresh_service_status()


def _merge_state(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = _clone_state(target)
    merged.update({k: _clone_state(v) for k, v in updates.items()})
    merged["updated"] = _now_iso()
    return merged


def _normalize_provider_reason(reason: Any) -> tuple[str | None, str | None]:
    if reason in (None, ""):
        return None, None
    detail = str(reason).strip()
    if not detail:
        return None, None
    normalized = detail.lower()
    code = "unknown"
    if "market_closed" in normalized or "market closed" in normalized:
        code = "market_closed"
    elif "startup" in normalized or "warmup" in normalized:
        code = "startup"
    elif "timeout" in normalized:
        code = "timeout"
    elif any(token in normalized for token in ("unauthorized", "forbidden", "auth", "invalid_api")):
        code = "auth"
    elif any(token in normalized for token in ("rate_limited", "too many requests", "429")):
        code = "rate_limit"
    elif any(token in normalized for token in ("gap_ratio", "data_gap", "gap_ratio_exceeded")):
        code = "gap_ratio"
    elif "stale" in normalized:
        code = "stale"
    elif any(token in normalized for token in ("5xx", "server_error", "upstream_unavailable")):
        code = "upstream_5xx"
    elif any(token in normalized for token in ("4xx", "invalid_request", "bad_request")):
        code = "upstream_4xx"
    elif "backup" in normalized:
        code = "backup_active"
    elif "quote" in normalized:
        code = "quote_quality"
    return code, detail[:256]


def update_data_provider_state(
    *,
    primary: str | None = None,
    active: str | None = None,
    backup: str | None = None,
    using_backup: bool | None = None,
    reason: str | None = None,
    reason_code: str | None = None,
    reason_detail: str | None = None,
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
        derived_code, derived_detail = _normalize_provider_reason(reason)
        if derived_code is not None:
            updates["reason_code"] = derived_code
        if derived_detail is not None:
            updates["reason_detail"] = derived_detail
    if reason_code is not None:
        token = str(reason_code).strip().lower()
        updates["reason_code"] = token or None
    if reason_detail is not None:
        detail = str(reason_detail).strip()
        updates["reason_detail"] = detail[:256] if detail else None
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
            if "reason_code" not in updates:
                if int(http_code) >= 500:
                    updates["reason_code"] = "upstream_5xx"
                elif int(http_code) >= 400:
                    updates["reason_code"] = "upstream_4xx"
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
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            updates["data_status"] = data_status
    if "reason_code" not in updates:
        if using_backup is True:
            updates["reason_code"] = "backup_active"
        elif status is not None:
            status_token = str(status).strip().lower()
            if status_token in {"down", "disabled", "failed", "unreachable"}:
                updates["reason_code"] = "provider_down"
    timeframe_key: str | None = None
    if timeframe is not None:
        try:
            candidate = str(timeframe).strip()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
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
        status_token = str(updates.get("status", _provider_state.get("status")) or "").strip().lower()
        effective_using_backup = bool(
            updates.get("using_backup", _provider_state.get("using_backup", False))
        )
        if status_token in {"healthy", "ready"} and not effective_using_backup:
            clear_defaults: dict[str, Any] = {
                "reason": None,
                "reason_code": None,
                "reason_detail": None,
                "last_error_at": None,
                "http_code": None,
                "cooldown_sec": None,
                "data_status": None,
                "safe_mode": False,
                "consecutive_failures": 0,
            }
            for key, value in clear_defaults.items():
                if key not in updates:
                    updates[key] = value
        _provider_state = _merge_state(_provider_state, updates)


def observe_data_provider_state() -> dict[str, Any]:
    with _LOCK:
        return _clone_state(_provider_state)


def reset_data_provider_state() -> None:
    """Reset provider telemetry to defaults."""

    with _LOCK:
        global _provider_state
        _provider_state = _fresh_provider_state()


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
        return _clone_state(_service_status)


def reset_service_status() -> None:
    """Reset service telemetry to defaults."""

    with _LOCK:
        global _service_status
        _service_status = _fresh_service_status()


def update_quote_status(
    *,
    allowed: bool,
    symbol: str | None = None,
    reason: str | None = None,
    age_sec: float | None = None,
    synthetic: bool | None = None,
    bid: float | None = None,
    ask: float | None = None,
    status: str | None = None,
    source: str | None = None,
    last_price: float | None = None,
    quote_age_ms: float | None = None,
    quote_timestamp: str | None = None,
    spread_bps: float | None = None,
    max_spread_bps: float | None = None,
    max_quote_age_ms: float | None = None,
    gate_reason: str | None = None,
) -> None:
    """Update snapshot of the most recent quote gate decision."""

    updates: dict[str, Any] = {
        "allowed": bool(allowed),
    }
    normalized_symbol = str(symbol or "").strip().upper()
    if normalized_symbol:
        updates["symbol"] = normalized_symbol
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
    if quote_timestamp is not None:
        updates["quote_timestamp"] = str(quote_timestamp)
    if spread_bps is not None:
        try:
            updates["spread_bps"] = max(0.0, float(spread_bps))
        except (TypeError, ValueError):
            pass
    if max_spread_bps is not None:
        try:
            updates["max_spread_bps"] = max(0.0, float(max_spread_bps))
        except (TypeError, ValueError):
            pass
    if max_quote_age_ms is not None:
        try:
            updates["max_quote_age_ms"] = max(0.0, float(max_quote_age_ms))
        except (TypeError, ValueError):
            pass
    if gate_reason is not None:
        updates["gate_reason"] = str(gate_reason)
    with _LOCK:
        global _quote_status, _quote_status_by_symbol
        latest_base = _quote_status
        if normalized_symbol:
            current_latest_symbol = str(_quote_status.get("symbol") or "").strip().upper()
            if current_latest_symbol and current_latest_symbol != normalized_symbol:
                latest_base = _fresh_quote_state()
        _quote_status = _merge_state(latest_base, updates)
        if normalized_symbol:
            current = _quote_status_by_symbol.get(normalized_symbol, _fresh_quote_state())
            _quote_status_by_symbol[normalized_symbol] = _merge_state(current, updates)


def observe_quote_status() -> dict[str, Any]:
    with _LOCK:
        return _clone_state(_quote_status)


def observe_symbol_quote_status(symbol: str) -> dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    with _LOCK:
        if not normalized_symbol:
            return _clone_state(_quote_status)
        return _clone_state(_quote_status_by_symbol.get(normalized_symbol, _fresh_quote_state()))


def reset_quote_status() -> None:
    """Reset quote telemetry to defaults."""

    with _LOCK:
        global _quote_status, _quote_status_by_symbol
        _quote_status = _fresh_quote_state()
        _quote_status_by_symbol = {}


def update_broker_status(
    *,
    connected: bool | None = None,
    latency_ms: float | None = None,
    last_error: str | None = None,
    status: str | None = None,
    last_order_ack_ms: float | None = None,
    open_orders_count: int | None = None,
    positions_count: int | None = None,
) -> None:
    """Record recent broker connectivity observations."""

    updates: dict[str, Any] = {}
    status_token: str | None = None
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
        status_token = str(status).strip().lower()
    if last_error is None and (
        connected is True or status_token in {"reachable", "healthy", "ok", "connected"}
    ):
        updates["last_error"] = None
    if last_order_ack_ms is not None:
        try:
            updates["last_order_ack_ms"] = max(0.0, float(last_order_ack_ms))
        except (TypeError, ValueError):
            pass
    if open_orders_count is not None:
        try:
            updates["open_orders_count"] = max(0, int(open_orders_count))
        except (TypeError, ValueError):
            pass
    if positions_count is not None:
        try:
            updates["positions_count"] = max(0, int(positions_count))
        except (TypeError, ValueError):
            pass
    with _LOCK:
        global _broker_status
        _broker_status = _merge_state(_broker_status, updates)


def observe_broker_status() -> dict[str, Any]:
    with _LOCK:
        return _clone_state(_broker_status)


def reset_broker_status() -> None:
    """Reset broker telemetry to defaults."""

    with _LOCK:
        global _broker_status
        _broker_status = _fresh_broker_state()


def reset_all_states() -> None:
    """Reset all runtime telemetry snapshots."""

    with _LOCK:
        global _provider_state, _quote_status, _quote_status_by_symbol, _broker_status, _service_status
        _provider_state = _fresh_provider_state()
        _quote_status = _fresh_quote_state()
        _quote_status_by_symbol = {}
        _broker_status = _fresh_broker_state()
        _service_status = _fresh_service_status()
