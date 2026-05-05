"""Live-canary state and pretrade gating helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.config.launch_profiles import (
    LaunchProfile,
    provider_authority_allows,
    resolve_launch_profile,
)
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.telemetry import runtime_state


def _now() -> datetime:
    return datetime.now(UTC)


def _today_key(now: datetime | None = None) -> str:
    return (now or _now()).strftime("%Y-%m-%d")


def _state_path() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/live_canary_state_latest.json",
        default_relative="runtime/live_canary_state_latest.json",
        for_write=True,
    )


def _events_path() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/live_canary_events.jsonl",
        default_relative="runtime/live_canary_events.jsonl",
        for_write=True,
    )


def _load_state() -> dict[str, Any]:
    path = _state_path()
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_state(state: Mapping[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(dict(state), indent=2, sort_keys=True) + "\n")


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _side_is_short(side: Any) -> bool:
    return str(side or "").strip().lower().replace("-", "_") in {"short", "sell_short"}


def _append_event(payload: Mapping[str, Any]) -> None:
    path = _events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def observe_live_canary_state() -> dict[str, Any]:
    state = _load_state()
    profile = resolve_launch_profile()
    if not state:
        state = {
            "artifact_type": "live_canary_state",
            "date": _today_key(),
            "profile": profile.name,
            "status": "ready" if profile.name == "live_canary" else "inactive",
            "entry_attempts": 0,
            "blocked_attempts": 0,
            "last_event": None,
        }
    return state


def evaluate_canary_order(
    order: Mapping[str, Any],
    *,
    profile: LaunchProfile | None = None,
    execution_mode: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    resolved = profile or resolve_launch_profile()
    if resolved.name != "live_canary":
        return True, {"enabled": False, "profile": resolved.name, "reason": "not_live_canary"}

    symbol = str(order.get("symbol") or "").strip().upper()
    side = str(order.get("side") or "").strip().lower()
    quantity = _float(order.get("quantity") if order.get("quantity") not in (None, "") else order.get("qty"))
    price_hint = _float(order.get("price_hint") or order.get("limit_price") or order.get("price"))
    notional = float(quantity or 0.0) * float(price_hint or 0.0)
    spread_bps = _float(order.get("spread_bps"))
    quote_age_ms = _float(order.get("quote_age_ms"))
    provider_state = runtime_state.observe_data_provider_state()
    quote_state = runtime_state.observe_symbol_quote_status(symbol) if symbol else runtime_state.observe_quote_status()
    provider_ok, provider_context = provider_authority_allows(
        profile=resolved,
        provider_state=provider_state,
        quote_state=quote_state,
        execution_mode=execution_mode,
    )
    state = observe_live_canary_state()
    today = _today_key()
    if str(state.get("date") or "") != today:
        state = {
            "artifact_type": "live_canary_state",
            "date": today,
            "profile": resolved.name,
            "status": "ready",
            "entry_attempts": 0,
            "blocked_attempts": 0,
            "last_event": None,
        }
    attempts = int(_float(state.get("entry_attempts")) or 0.0)
    reasons: list[str] = []
    if resolved.allowed_symbols and symbol not in resolved.allowed_symbols:
        reasons.append("symbol_not_allowlisted")
    if not resolved.shorts_allowed and _side_is_short(side):
        reasons.append("shorts_disabled")
    if resolved.max_notional_per_order is not None and notional > float(resolved.max_notional_per_order):
        reasons.append("notional_cap_exceeded")
    if attempts >= int(resolved.max_order_count):
        reasons.append("daily_order_count_cap_exceeded")
    if resolved.max_quote_age_ms is not None and quote_age_ms is not None and quote_age_ms > float(resolved.max_quote_age_ms):
        reasons.append("quote_age_cap_exceeded")
    if resolved.max_spread_bps is not None and spread_bps is not None and spread_bps > float(resolved.max_spread_bps):
        reasons.append("spread_cap_exceeded")
    if not provider_ok:
        reasons.extend(str(reason) for reason in provider_context.get("reasons", []))

    allowed = not reasons
    event = {
        "ts": _now().isoformat().replace("+00:00", "Z"),
        "profile": resolved.name,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "notional": notional if notional > 0.0 else None,
        "allowed": allowed,
        "reasons": reasons,
        "provider_authority": provider_context,
    }
    state["profile"] = resolved.name
    state["entry_attempts"] = attempts + (1 if allowed else 0)
    state["blocked_attempts"] = int(_float(state.get("blocked_attempts")) or 0.0) + (0 if allowed else 1)
    state["status"] = "ready" if allowed else "blocked"
    state["last_event"] = event
    try:
        _write_state(state)
        _append_event(event)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        if allowed:
            return False, event | {"reasons": ["live_canary_state_write_failed"]}
    return allowed, event


__all__ = ["evaluate_canary_order", "observe_live_canary_state"]
