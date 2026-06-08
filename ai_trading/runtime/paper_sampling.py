"""Diagnostic paper-order sampling gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.runtime.atomic_io import atomic_write_text


_STATE_LOCK = RLock()


@dataclass(frozen=True)
class PaperSamplingDecision:
    enabled: bool
    allowed: bool
    qty: int
    reason: str
    details: dict[str, Any]


def _state_path() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/paper_sampling_state_latest.json",
        default_relative="runtime/paper_sampling_state_latest.json",
        for_write=True,
    )


def _today_key(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC).date().isoformat()


def _session_bucket(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    local = current.astimezone(ZoneInfo("America/New_York"))
    minutes = local.hour * 60 + local.minute
    if minutes < (9 * 60 + 30) or minutes >= (16 * 60):
        return "offhours"
    if minutes < (11 * 60):
        return "opening"
    if minutes >= (15 * 60):
        return "closing"
    return "midday"


def _load_state(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(dict(state), sort_keys=True, separators=(",", ":")),
    )


def _allowed_symbols(cfg: Any) -> set[str]:
    raw_symbols = getattr(cfg, "paper_sampling_allowed_symbols", ("AAPL", "AMZN"))
    candidates: Iterable[Any]
    if isinstance(raw_symbols, str):
        candidates = raw_symbols.split(",")
    else:
        candidates = raw_symbols or ()
    symbols = {str(symbol).strip().upper() for symbol in candidates if str(symbol).strip()}
    return symbols or {"AAPL", "AMZN"}


def _is_paper_sampling_active(cfg: Any) -> tuple[bool, str | None]:
    if not bool(getattr(cfg, "paper_sampling_enabled", False)):
        return False, None
    execution_mode = str(getattr(cfg, "execution_mode", "sim") or "sim").strip().lower()
    paper_mode = bool(getattr(cfg, "paper", False))
    base_url = str(getattr(cfg, "alpaca_base_url", "") or "").strip().lower()
    launch_profile = str(
        getattr(
            cfg,
            "launch_profile",
            get_env("AI_TRADING_LAUNCH_PROFILE", "", cast=str),
        )
        or ""
    ).strip().lower()
    if execution_mode != "paper":
        return False, "execution_mode_not_paper"
    if not paper_mode or "paper" not in base_url:
        return False, "live_money_endpoint"
    if launch_profile == "live_canary" or launch_profile.startswith("live_"):
        return False, "live_launch_profile"
    return True, None


def _cfg_int(cfg: Any, field: str, default: int) -> int:
    try:
        value = int(getattr(cfg, field, default) or default)
    except (TypeError, ValueError):
        return int(default)
    return max(0, value)


def _state_count(state: Mapping[str, Any], key: str, bucket: str) -> int:
    raw = state.get(key)
    if not isinstance(raw, Mapping):
        return 0
    try:
        return int(raw.get(bucket, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _quota_block(
    *,
    decision: PaperSamplingDecision,
    reason: str,
    today: str,
    count: int,
    quota: int,
    quota_key: str,
) -> PaperSamplingDecision:
    details = dict(decision.details)
    details.update(
        {
            "date": today,
            "count": int(count),
            "quota": int(quota),
            "quota_key": str(quota_key),
        }
    )
    return PaperSamplingDecision(True, False, decision.qty, reason, details)


def evaluate_paper_sampling_order(
    cfg: Any,
    *,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    consumes_daily_slot: bool = True,
) -> PaperSamplingDecision:
    """Apply diagnostic paper-sampling narrowing without bypassing hard gates."""

    active, inactive_reason = _is_paper_sampling_active(cfg)
    if not active:
        return PaperSamplingDecision(
            enabled=bool(getattr(cfg, "paper_sampling_enabled", False)),
            allowed=inactive_reason is None,
            qty=int(max(0, qty)),
            reason=inactive_reason or "paper_sampling_disabled",
            details={"reason": inactive_reason} if inactive_reason else {},
        )

    symbol_key = str(symbol).strip().upper()
    side_key = str(side).strip().lower()
    details: dict[str, Any] = {
        "symbol": symbol_key,
        "side": side_key,
        "mode": "paper_sampling",
        "consumes_daily_slot": bool(consumes_daily_slot),
    }
    if not bool(consumes_daily_slot):
        requested_qty = int(max(0, qty))
        details.update({"requested_qty": requested_qty, "adjusted_qty": requested_qty})
        return PaperSamplingDecision(True, True, requested_qty, "OK", details)
    if symbol_key not in _allowed_symbols(cfg):
        details["allowed_symbols"] = sorted(_allowed_symbols(cfg))
        return PaperSamplingDecision(True, False, 0, "PAPER_SAMPLING_SYMBOL_BLOCK", details)
    if side_key in {"short", "sell_short", "sell-short", "sell short"}:
        return PaperSamplingDecision(True, False, 0, "PAPER_SAMPLING_SHORT_BLOCK", details)

    requested_qty = int(max(0, qty))
    price_value = float(price)
    max_notional = float(getattr(cfg, "paper_sampling_max_notional_per_order", 250.0) or 250.0)
    if not math.isfinite(price_value) or price_value <= 0.0 or requested_qty <= 0:
        details.update({"qty": requested_qty, "price": price})
        return PaperSamplingDecision(True, False, 0, "PAPER_SAMPLING_INPUT_BLOCK", details)

    if not math.isfinite(max_notional) or max_notional <= 0.0:
        details.update({"qty": requested_qty, "price": price_value, "max_notional_per_order": max_notional})
        return PaperSamplingDecision(True, False, 0, "PAPER_SAMPLING_INPUT_BLOCK", details)
    if price_value > max_notional:
        details.update(
            {
                "requested_qty": requested_qty,
                "adjusted_qty": 0,
                "price": price_value,
                "max_notional_per_order": max_notional,
            }
        )
        return PaperSamplingDecision(True, False, 0, "PAPER_SAMPLING_MAX_NOTIONAL_BLOCK", details)

    max_qty = int(math.floor(max_notional / price_value))
    adjusted_qty = min(requested_qty, max_qty)
    details.update(
        {
            "requested_qty": requested_qty,
            "adjusted_qty": adjusted_qty,
            "price": price_value,
            "max_notional_per_order": max_notional,
            "one_share_fallback": bool(max_notional < price_value and adjusted_qty == 1),
        }
    )
    return PaperSamplingDecision(True, True, adjusted_qty, "OK", details)


def reserve_paper_sampling_order(
    cfg: Any,
    *,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    now: datetime | None = None,
    consumes_daily_slot: bool = True,
) -> PaperSamplingDecision:
    """Reserve a diagnostic paper-sampling daily slot after upstream gates pass."""

    decision = evaluate_paper_sampling_order(
        cfg,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        consumes_daily_slot=consumes_daily_slot,
    )
    if not decision.enabled or not decision.allowed:
        return decision
    if not bool(consumes_daily_slot):
        return decision

    max_trades = int(getattr(cfg, "paper_sampling_max_trades_per_day", 2) or 2)
    today = _today_key(now)
    session = _session_bucket(now)
    path = _state_path()
    with _STATE_LOCK:
        state = _load_state(path)
        state_date = str(state.get("date") or "")
        count = int(state.get("count", 0) or 0) if state_date == today else 0
        if count >= max_trades:
            details = dict(decision.details)
            details.update({"date": today, "count": count, "max_trades_per_day": max_trades})
            return PaperSamplingDecision(
                True,
                False,
                decision.qty,
                "PAPER_SAMPLING_DAILY_CAP_BLOCK",
                details,
            )
        symbol_key = str(symbol).strip().upper()
        side_key = str(side).strip().lower()
        by_symbol = state.get("by_symbol") if state_date == today else {}
        by_side = state.get("by_side") if state_date == today else {}
        by_session = state.get("by_session") if state_date == today else {}
        if not isinstance(by_symbol, Mapping):
            by_symbol = {}
        if not isinstance(by_side, Mapping):
            by_side = {}
        if not isinstance(by_session, Mapping):
            by_session = {}

        symbol_quota = _cfg_int(cfg, "paper_sampling_max_trades_per_symbol_per_day", 4)
        if symbol_quota > 0 and _state_count({"by_symbol": by_symbol}, "by_symbol", symbol_key) >= symbol_quota:
            return _quota_block(
                decision=decision,
                reason="PAPER_SAMPLING_SYMBOL_DAILY_QUOTA_BLOCK",
                today=today,
                count=_state_count({"by_symbol": by_symbol}, "by_symbol", symbol_key),
                quota=symbol_quota,
                quota_key=f"symbol:{symbol_key}",
            )

        side_quota = _cfg_int(cfg, "paper_sampling_max_trades_per_side_per_day", 6)
        if side_quota > 0 and _state_count({"by_side": by_side}, "by_side", side_key) >= side_quota:
            return _quota_block(
                decision=decision,
                reason="PAPER_SAMPLING_SIDE_DAILY_QUOTA_BLOCK",
                today=today,
                count=_state_count({"by_side": by_side}, "by_side", side_key),
                quota=side_quota,
                quota_key=f"side:{side_key}",
            )

        session_quota = {
            "opening": _cfg_int(cfg, "paper_sampling_max_opening_trades_per_day", 3),
            "midday": _cfg_int(cfg, "paper_sampling_max_midday_trades_per_day", 4),
            "closing": _cfg_int(cfg, "paper_sampling_max_closing_trades_per_day", 3),
        }.get(session, 0)
        if session_quota > 0 and _state_count({"by_session": by_session}, "by_session", session) >= session_quota:
            return _quota_block(
                decision=decision,
                reason="PAPER_SAMPLING_SESSION_DAILY_QUOTA_BLOCK",
                today=today,
                count=_state_count({"by_session": by_session}, "by_session", session),
                quota=session_quota,
                quota_key=f"session:{session}",
            )

        next_by_symbol = dict(by_symbol)
        next_by_side = dict(by_side)
        next_by_session = dict(by_session)
        next_by_symbol[symbol_key] = _state_count({"by_symbol": by_symbol}, "by_symbol", symbol_key) + 1
        next_by_side[side_key] = _state_count({"by_side": by_side}, "by_side", side_key) + 1
        next_by_session[session] = _state_count({"by_session": by_session}, "by_session", session) + 1
        state = {
            "artifact_type": "paper_sampling_state",
            "date": today,
            "count": count + 1,
            "by_symbol": next_by_symbol,
            "by_side": next_by_side,
            "by_session": next_by_session,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _write_state(path, state)
    details = dict(decision.details)
    details.update(
        {
            "date": today,
            "count": count + 1,
            "max_trades_per_day": max_trades,
            "session_bucket": session,
        }
    )
    return PaperSamplingDecision(True, True, decision.qty, "OK", details)


def release_paper_sampling_order(
    cfg: Any,
    *,
    symbol: str,
    side: str,
    now: datetime | None = None,
    consumes_daily_slot: bool = True,
) -> None:
    """Release a diagnostic paper-sampling slot when submit is not accepted."""

    active, _ = _is_paper_sampling_active(cfg)
    if not active or not bool(consumes_daily_slot):
        return
    today = _today_key(now)
    session = _session_bucket(now)
    symbol_key = str(symbol).strip().upper()
    side_key = str(side).strip().lower()
    if not symbol_key or not side_key:
        return
    path = _state_path()
    with _STATE_LOCK:
        state = _load_state(path)
        if str(state.get("date") or "") != today:
            return
        count = max(0, int(state.get("count", 0) or 0) - 1)

        def _decrement_map(key: str, bucket: str) -> dict[str, int]:
            raw = state.get(key)
            if not isinstance(raw, Mapping):
                return {}
            updated = {str(k): int(v or 0) for k, v in raw.items()}
            next_count = max(0, int(updated.get(bucket, 0) or 0) - 1)
            if next_count > 0:
                updated[bucket] = next_count
            else:
                updated.pop(bucket, None)
            return updated

        _write_state(
            path,
            {
                "artifact_type": "paper_sampling_state",
                "date": today,
                "count": count,
                "by_symbol": _decrement_map("by_symbol", symbol_key),
                "by_side": _decrement_map("by_side", side_key),
                "by_session": _decrement_map("by_session", session),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )


__all__ = [
    "PaperSamplingDecision",
    "evaluate_paper_sampling_order",
    "release_paper_sampling_order",
    "reserve_paper_sampling_order",
]
