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
from ai_trading.config.management import get_env
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.telemetry import runtime_state


def _now() -> datetime:
    return datetime.now(UTC)


def _today_key(now: datetime | None = None) -> str:
    return (now or _now()).strftime("%Y-%m-%d")


def _state_path(*, profile_name: str = "live_canary") -> Path:
    if str(profile_name or "").strip().lower() != "live_canary":
        return resolve_runtime_artifact_path(
            "runtime/launch_profile_state_latest.json",
            default_relative="runtime/launch_profile_state_latest.json",
            for_write=True,
        )
    return resolve_runtime_artifact_path(
        "runtime/live_canary_state_latest.json",
        default_relative="runtime/live_canary_state_latest.json",
        for_write=True,
    )


def _events_path(*, profile_name: str = "live_canary") -> Path:
    if str(profile_name or "").strip().lower() != "live_canary":
        return resolve_runtime_artifact_path(
            "runtime/launch_profile_events.jsonl",
            default_relative="runtime/launch_profile_events.jsonl",
            for_write=True,
        )
    return resolve_runtime_artifact_path(
        "runtime/live_canary_events.jsonl",
        default_relative="runtime/live_canary_events.jsonl",
        for_write=True,
    )


def _load_state(*, profile_name: str = "live_canary") -> dict[str, Any]:
    path = _state_path(profile_name=profile_name)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_state(state: Mapping[str, Any], *, profile_name: str = "live_canary") -> None:
    path = _state_path(profile_name=profile_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(dict(state), indent=2, sort_keys=True) + "\n")


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _side_is_short(side: Any) -> bool:
    return str(side or "").strip().lower().replace("-", "_") in {"sell", "short", "sell_short"}


def _append_event(payload: Mapping[str, Any], *, profile_name: str = "live_canary") -> None:
    path = _events_path(profile_name=profile_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _profile_gate_active(profile: LaunchProfile) -> bool:
    if profile.name == "paper_trade":
        return False
    return profile.name == "paper_observe" or profile.name.startswith("live_")


def _readiness_status_allows(
    profile: LaunchProfile,
    *,
    live_capital_active: bool,
) -> tuple[bool, dict[str, Any]]:
    if not profile.name.startswith("live_") or not live_capital_active:
        return True, {
            "required": False,
            "reason": "not_live_capital" if not live_capital_active else "not_live_profile",
        }
    path = resolve_runtime_artifact_path(
        "runtime/live_capital_readiness_latest.json",
        default_relative="runtime/live_capital_readiness_latest.json",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, {"required": True, "path": str(path), "status": None}
    if not isinstance(payload, Mapping):
        return False, {"required": True, "path": str(path), "status": None}
    status = str(payload.get("status") or "").strip().lower()
    allowed_statuses = (
        {"live_canary_allowed"}
        if profile.name == "live_canary"
        else {"live_canary_allowed", "live_allowed"}
    )
    return status in allowed_statuses, {
        "required": True,
        "path": str(path),
        "status": status or None,
        "allowed_statuses": sorted(allowed_statuses),
    }


def _operator_override_allows(profile: LaunchProfile, *, live_capital_active: bool) -> bool:
    if not profile.manual_approval_required or not live_capital_active:
        return True
    raw = get_env(
        "AI_TRADING_LIVE_CAPITAL_OPERATOR_APPROVED",
        "0",
        cast=str,
        resolve_aliases=False,
    )
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _price_hint_from_quote(side: str, quote_state: Mapping[str, Any]) -> float | None:
    bid = _float(quote_state.get("bid"))
    ask = _float(quote_state.get("ask"))
    side_token = str(side or "").strip().lower()
    if side_token in {"buy", "cover"} and ask is not None and ask > 0.0:
        return ask
    if side_token in {"sell", "short", "sell_short"} and bid is not None and bid > 0.0:
        return bid
    if bid is not None and ask is not None and bid > 0.0 and ask > 0.0:
        return (bid + ask) / 2.0
    last_price = _float(quote_state.get("last_price") or quote_state.get("price"))
    return last_price if last_price is not None and last_price > 0.0 else None


def observe_launch_profile_state(profile: LaunchProfile | None = None) -> dict[str, Any]:
    resolved = profile or resolve_launch_profile()
    state = _load_state(profile_name=resolved.name)
    if not state:
        state = {
            "artifact_type": (
                "live_canary_state"
                if resolved.name == "live_canary"
                else "launch_profile_state"
            ),
            "date": _today_key(),
            "profile": resolved.name,
            "status": "ready" if _profile_gate_active(resolved) else "inactive",
            "entry_attempts": 0,
            "blocked_attempts": 0,
            "last_event": None,
        }
    return state


def observe_live_canary_state() -> dict[str, Any]:
    return observe_launch_profile_state(resolve_launch_profile("live_canary"))


def evaluate_launch_profile_order(
    order: Mapping[str, Any],
    *,
    profile: LaunchProfile | None = None,
    execution_mode: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    resolved = profile or resolve_launch_profile()
    mode = str(execution_mode or "").strip().lower()
    if mode not in {"paper", "live"}:
        return True, {
            "enabled": False,
            "profile": resolved.name,
            "reason": "execution_mode_not_enforced",
            "execution_mode": mode or None,
        }
    if not _profile_gate_active(resolved):
        return True, {"enabled": False, "profile": resolved.name, "reason": "profile_not_enforced"}
    live_capital_active = mode == "live"

    symbol = str(order.get("symbol") or "").strip().upper()
    side = str(order.get("side") or "").strip().lower()
    quantity = _float(order.get("quantity") if order.get("quantity") not in (None, "") else order.get("qty"))
    explicit_notional = _float(order.get("notional"))
    price_hint = _float(order.get("price_hint") or order.get("limit_price") or order.get("price"))
    spread_bps = _float(order.get("spread_bps"))
    quote_age_ms = _float(order.get("quote_age_ms"))
    provider_state = runtime_state.observe_data_provider_state()
    quote_state = runtime_state.observe_symbol_quote_status(symbol) if symbol else runtime_state.observe_quote_status()
    if price_hint is None:
        price_hint = _price_hint_from_quote(side, quote_state)
    notional = (
        float(explicit_notional)
        if explicit_notional is not None and explicit_notional > 0.0
        else float(quantity or 0.0) * float(price_hint or 0.0)
    )
    if quote_age_ms is None:
        quote_age_ms = _float(
            quote_state.get("quote_age_ms")
            if quote_state.get("quote_age_ms") not in (None, "")
            else quote_state.get("age_ms")
        )
    if spread_bps is None:
        spread_bps = _float(quote_state.get("spread_bps"))
        bid = _float(quote_state.get("bid"))
        ask = _float(quote_state.get("ask"))
        if spread_bps is None and bid is not None and ask is not None and bid > 0.0:
            midpoint = (bid + ask) / 2.0
            if midpoint > 0.0:
                spread_bps = ((ask - bid) / midpoint) * 10000.0
    provider_ok, provider_context = provider_authority_allows(
        profile=resolved,
        provider_state=provider_state,
        quote_state=quote_state,
        execution_mode=execution_mode,
    )
    readiness_ok, readiness_context = _readiness_status_allows(
        resolved,
        live_capital_active=live_capital_active,
    )
    operator_ok = _operator_override_allows(
        resolved,
        live_capital_active=live_capital_active,
    )
    state = observe_launch_profile_state(resolved)
    today = _today_key()
    if str(state.get("date") or "") != today:
        state = {
            "artifact_type": (
                "live_canary_state"
                if resolved.name == "live_canary"
                else "launch_profile_state"
            ),
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
    if (
        resolved.max_notional_per_order is not None
        and notional <= 0.0
        and resolved.name.startswith("live_")
        and live_capital_active
    ):
        reasons.append("notional_unknown")
    if attempts >= int(resolved.max_order_count):
        reasons.append("daily_order_count_cap_exceeded")
    if resolved.max_quote_age_ms is not None and quote_age_ms is not None and quote_age_ms > float(resolved.max_quote_age_ms):
        reasons.append("quote_age_cap_exceeded")
    if resolved.max_spread_bps is not None and spread_bps is not None and spread_bps > float(resolved.max_spread_bps):
        reasons.append("spread_cap_exceeded")
    if not provider_ok:
        reasons.extend(str(reason) for reason in provider_context.get("reasons", []))
    if resolved.promotion_required and not readiness_ok:
        reasons.append("live_capital_readiness_not_allowed")
    if resolved.manual_approval_required and not operator_ok:
        reasons.append("operator_approval_missing")

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
        "live_capital_readiness": readiness_context,
        "operator_approval_required": bool(resolved.manual_approval_required),
    }
    state["profile"] = resolved.name
    state["entry_attempts"] = attempts + (1 if allowed else 0)
    state["blocked_attempts"] = int(_float(state.get("blocked_attempts")) or 0.0) + (0 if allowed else 1)
    state["status"] = "ready" if allowed else "blocked"
    state["last_event"] = event
    try:
        _write_state(state, profile_name=resolved.name)
        _append_event(event, profile_name=resolved.name)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        if allowed:
            return False, event | {"reasons": ["launch_profile_state_write_failed"]}
    return allowed, event


def evaluate_canary_order(
    order: Mapping[str, Any],
    *,
    profile: LaunchProfile | None = None,
    execution_mode: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    resolved = profile or resolve_launch_profile()
    if resolved.name != "live_canary":
        return True, {"enabled": False, "profile": resolved.name, "reason": "not_live_canary"}
    return evaluate_launch_profile_order(
        order,
        profile=resolved,
        execution_mode=execution_mode,
    )


__all__ = [
    "evaluate_canary_order",
    "evaluate_launch_profile_order",
    "observe_launch_profile_state",
    "observe_live_canary_state",
]
