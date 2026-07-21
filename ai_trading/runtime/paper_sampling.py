"""Diagnostic paper-order sampling gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping
from uuid import uuid4
from zoneinfo import ZoneInfo

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.runtime.atomic_io import atomic_write_text


_STATE_LOCK = RLock()
_GOVERNED_PAPER_SAMPLING_SYMBOLS = frozenset({"AAPL", "AMZN", "MSFT"})
_REGULAR_SESSIONS = ("opening", "midday", "closing")


@dataclass(frozen=True)
class PaperSamplingDecision:
    enabled: bool
    allowed: bool
    qty: int
    reason: str
    details: dict[str, Any]


def _state_path(*, for_write: bool = True) -> Path:
    return resolve_runtime_artifact_path(
        "runtime/paper_sampling_state_latest.json",
        default_relative="runtime/paper_sampling_state_latest.json",
        for_write=for_write,
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
    raw_symbols = getattr(
        cfg,
        "paper_sampling_allowed_symbols",
        tuple(sorted(_GOVERNED_PAPER_SAMPLING_SYMBOLS)),
    )
    candidates: Iterable[Any]
    if isinstance(raw_symbols, str):
        candidates = raw_symbols.split(",")
    else:
        candidates = raw_symbols or ()
    symbols = {str(symbol).strip().upper() for symbol in candidates if str(symbol).strip()}
    return symbols.intersection(_GOVERNED_PAPER_SAMPLING_SYMBOLS)


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
        raw_value = getattr(cfg, field, default)
        value = int(default if raw_value in (None, "") else raw_value)
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


def _count_map(state: Mapping[str, Any], key: str) -> dict[str, int]:
    raw = state.get(key)
    if not isinstance(raw, Mapping):
        return {}
    counts: dict[str, int] = {}
    for bucket, value in raw.items():
        try:
            count = max(0, int(value or 0))
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[str(bucket)] = count
    return counts


def _increment_count(counts: Mapping[str, int], bucket: str) -> dict[str, int]:
    updated = dict(counts)
    updated[bucket] = max(0, int(updated.get(bucket, 0))) + 1
    return updated


def _decrement_count(counts: Mapping[str, int], bucket: str) -> dict[str, int]:
    updated = dict(counts)
    next_count = max(0, int(updated.get(bucket, 0)) - 1)
    if next_count > 0:
        updated[bucket] = next_count
    else:
        updated.pop(bucket, None)
    return updated


def _sampling_role(role: str | None, *, consumes_daily_slot: bool) -> str:
    normalized = str(role or "").strip().lower()
    if normalized in {"entry", "exit"}:
        return normalized
    return "entry" if consumes_daily_slot else "exit"


def _rotated_symbols(symbols: Iterable[str], today: str) -> list[str]:
    ordered = sorted(
        {
            str(symbol).strip().upper()
            for symbol in symbols
            if str(symbol).strip()
        }
    )
    if len(ordered) <= 1:
        return ordered
    try:
        rotation = datetime.fromisoformat(today).date().toordinal() % len(ordered)
    except ValueError:
        rotation = 0
    return ordered[rotation:] + ordered[:rotation]


def _symbol_targets(
    cfg: Any,
    *,
    today: str,
    max_trades: int,
) -> dict[str, int]:
    """Return deterministic balanced daily targets within every hard cap."""

    symbols = _rotated_symbols(_allowed_symbols(cfg), today)
    if not symbols or max_trades <= 0:
        return {}
    configured_quota = _cfg_int(
        cfg,
        "paper_sampling_max_trades_per_symbol_per_day",
        4,
    )
    per_symbol_cap = configured_quota if configured_quota > 0 else max_trades
    targets = {symbol: 0 for symbol in symbols}
    remaining = int(max_trades)
    while remaining > 0:
        progressed = False
        for symbol in symbols:
            if remaining <= 0:
                break
            if targets[symbol] >= per_symbol_cap:
                continue
            targets[symbol] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return targets


def _session_reserved_minima(cfg: Any) -> dict[str, int]:
    return {
        "opening": _cfg_int(
            cfg,
            "paper_sampling_reserved_opening_trades_per_day",
            0,
        ),
        "midday": _cfg_int(
            cfg,
            "paper_sampling_reserved_midday_trades_per_day",
            0,
        ),
        "closing": _cfg_int(
            cfg,
            "paper_sampling_reserved_closing_trades_per_day",
            0,
        ),
    }


def _reservation_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("reservations")
    if not isinstance(raw, list):
        return []
    return [dict(row) for row in raw if isinstance(row, Mapping)]


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


def paper_sampling_deficit_snapshot(
    cfg: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a read-only governed sampling-deficit snapshot for candidate ranking."""

    active, inactive_reason = _is_paper_sampling_active(cfg)
    today = _today_key(now)
    session = _session_bucket(now)
    configured_symbols = _rotated_symbols(_allowed_symbols(cfg), today)
    fairness_enabled = bool(
        getattr(cfg, "paper_sampling_stratified_fairness_enabled", False)
    )
    base: dict[str, Any] = {
        "active": bool(active),
        "fairness_enabled": fairness_enabled,
        "reason": inactive_reason or ("active" if active else "paper_sampling_disabled"),
        "date": today,
        "session_bucket": session,
        "configured_symbols": configured_symbols,
        "counts": {symbol: 0 for symbol in configured_symbols},
        "session_counts": {symbol: 0 for symbol in configured_symbols},
        "targets": {symbol: 0 for symbol in configured_symbols},
        "deficits": {symbol: 0 for symbol in configured_symbols},
        "priority_symbols": [],
        "ranked_underfilled_symbols": [],
        "priority_reason": "inactive" if not active else "balanced",
    }
    if not active or not configured_symbols:
        return base

    max_trades = _cfg_int(cfg, "paper_sampling_max_trades_per_day", 12)
    targets = _symbol_targets(cfg, today=today, max_trades=max_trades)
    path = _state_path(for_write=False)
    with _STATE_LOCK:
        state = _load_state(path)
    current_state = state if str(state.get("date") or "") == today else {}
    raw_counts = _count_map(current_state, "by_symbol")
    raw_session_counts = _count_map(
        current_state,
        "observed_by_symbol_session",
    )
    counts = {
        symbol: int(raw_counts.get(symbol, 0))
        for symbol in configured_symbols
    }
    session_counts = {
        symbol: int(raw_session_counts.get(f"{symbol}:{session}", 0))
        for symbol in configured_symbols
    }
    deficits = {
        symbol: max(0, int(targets.get(symbol, 0)) - counts[symbol])
        for symbol in configured_symbols
    }
    rotation_index = {
        symbol: index for index, symbol in enumerate(configured_symbols)
    }
    underfilled = [
        symbol
        for symbol in configured_symbols
        if int(targets.get(symbol, 0)) > 0 and deficits[symbol] > 0
    ]
    ranked_underfilled = sorted(
        underfilled,
        key=lambda symbol: (
            -deficits[symbol],
            session_counts[symbol],
            rotation_index[symbol],
        ),
    )
    priority_symbols: list[str] = []
    priority_reason = "balanced"
    if underfilled:
        distinct_deficits = {deficits[symbol] for symbol in underfilled}
        if len(distinct_deficits) > 1:
            maximum_deficit = max(distinct_deficits)
            priority_symbols = [
                symbol
                for symbol in ranked_underfilled
                if deficits[symbol] == maximum_deficit
            ]
            priority_reason = "symbol_deficit"
        else:
            distinct_session_counts = {
                session_counts[symbol] for symbol in underfilled
            }
            if len(distinct_session_counts) > 1:
                minimum_session_count = min(distinct_session_counts)
                priority_symbols = [
                    symbol
                    for symbol in ranked_underfilled
                    if session_counts[symbol] == minimum_session_count
                ]
                priority_reason = "session_deficit"

    return base | {
        "counts": counts,
        "session_counts": session_counts,
        "targets": {
            symbol: int(targets.get(symbol, 0))
            for symbol in configured_symbols
        },
        "deficits": deficits,
        "priority_symbols": priority_symbols,
        "ranked_underfilled_symbols": ranked_underfilled,
        "priority_reason": priority_reason,
    }


def evaluate_paper_sampling_order(
    cfg: Any,
    *,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    consumes_daily_slot: bool = True,
    role: str | None = None,
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
    role_key = _sampling_role(role, consumes_daily_slot=consumes_daily_slot)
    details: dict[str, Any] = {
        "symbol": symbol_key,
        "side": side_key,
        "role": role_key,
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
    configured_max_notional = float(
        getattr(cfg, "paper_sampling_max_notional_per_order", 750.0) or 750.0
    )
    max_notional = min(configured_max_notional, 750.0)
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
    role: str | None = None,
) -> PaperSamplingDecision:
    """Reserve a diagnostic paper-sampling daily slot after upstream gates pass."""

    decision = evaluate_paper_sampling_order(
        cfg,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        consumes_daily_slot=consumes_daily_slot,
        role=role,
    )
    if not decision.enabled or not decision.allowed:
        return decision

    max_trades = _cfg_int(cfg, "paper_sampling_max_trades_per_day", 12)
    today = _today_key(now)
    session = _session_bucket(now)
    symbol_key = str(symbol).strip().upper()
    side_key = str(side).strip().lower()
    role_key = _sampling_role(role, consumes_daily_slot=consumes_daily_slot)
    if not consumes_daily_slot and symbol_key not in _allowed_symbols(cfg):
        return decision
    path = _state_path()
    with _STATE_LOCK:
        state = _load_state(path)
        state_date = str(state.get("date") or "")
        if not consumes_daily_slot and state and state_date != today:
            return decision
        count = int(state.get("count", 0) or 0) if state_date == today else 0
        if consumes_daily_slot and count >= max_trades:
            details = dict(decision.details)
            details.update({"date": today, "count": count, "max_trades_per_day": max_trades})
            return PaperSamplingDecision(
                True,
                False,
                decision.qty,
                "PAPER_SAMPLING_DAILY_CAP_BLOCK",
                details,
            )
        current_state = state if state_date == today else {}
        by_symbol = _count_map(current_state, "by_symbol")
        by_side = _count_map(current_state, "by_side")
        by_session = _count_map(current_state, "by_session")

        symbol_quota = _cfg_int(cfg, "paper_sampling_max_trades_per_symbol_per_day", 4)
        if (
            consumes_daily_slot
            and symbol_quota > 0
            and int(by_symbol.get(symbol_key, 0)) >= symbol_quota
        ):
            return _quota_block(
                decision=decision,
                reason="PAPER_SAMPLING_SYMBOL_DAILY_QUOTA_BLOCK",
                today=today,
                count=int(by_symbol.get(symbol_key, 0)),
                quota=symbol_quota,
                quota_key=f"symbol:{symbol_key}",
            )

        side_quota = _cfg_int(cfg, "paper_sampling_max_trades_per_side_per_day", 6)
        if (
            consumes_daily_slot
            and side_quota > 0
            and int(by_side.get(side_key, 0)) >= side_quota
        ):
            return _quota_block(
                decision=decision,
                reason="PAPER_SAMPLING_SIDE_DAILY_QUOTA_BLOCK",
                today=today,
                count=int(by_side.get(side_key, 0)),
                quota=side_quota,
                quota_key=f"side:{side_key}",
            )

        session_quota = {
            "opening": _cfg_int(cfg, "paper_sampling_max_opening_trades_per_day", 3),
            "midday": _cfg_int(cfg, "paper_sampling_max_midday_trades_per_day", 4),
            "closing": _cfg_int(cfg, "paper_sampling_max_closing_trades_per_day", 3),
        }.get(session, 0)
        if consumes_daily_slot and session not in _REGULAR_SESSIONS:
            details = dict(decision.details)
            details.update({"date": today, "session_bucket": session})
            return PaperSamplingDecision(
                True,
                False,
                decision.qty,
                "PAPER_SAMPLING_SESSION_BLOCK",
                details,
            )
        if (
            consumes_daily_slot
            and session_quota > 0
            and int(by_session.get(session, 0)) >= session_quota
        ):
            return _quota_block(
                decision=decision,
                reason="PAPER_SAMPLING_SESSION_DAILY_QUOTA_BLOCK",
                today=today,
                count=int(by_session.get(session, 0)),
                quota=session_quota,
                quota_key=f"session:{session}",
            )

        fairness_enabled = bool(
            getattr(cfg, "paper_sampling_stratified_fairness_enabled", False)
        )
        symbol_targets = _symbol_targets(
            cfg,
            today=today,
            max_trades=max_trades,
        )
        max_symbol_lead = max(
            1,
            _cfg_int(cfg, "paper_sampling_symbol_fairness_max_lead", 1),
        )
        if consumes_daily_slot and fairness_enabled:
            symbol_target = int(symbol_targets.get(symbol_key, 0))
            symbol_count = int(by_symbol.get(symbol_key, 0))
            targeted_counts = [
                int(by_symbol.get(candidate, 0))
                for candidate, target in symbol_targets.items()
                if int(target) > 0
            ]
            minimum_count = min(targeted_counts, default=0)
            if symbol_target <= 0 or symbol_count >= symbol_target:
                details = dict(decision.details)
                details.update(
                    {
                        "date": today,
                        "count": symbol_count,
                        "quota": symbol_target,
                        "quota_key": f"symbol_reservation:{symbol_key}",
                        "symbol_targets": symbol_targets,
                    }
                )
                return PaperSamplingDecision(
                    True,
                    False,
                    decision.qty,
                    "PAPER_SAMPLING_SYMBOL_RESERVATION_BLOCK",
                    details,
                )
            if symbol_count - minimum_count >= max_symbol_lead:
                details = dict(decision.details)
                details.update(
                    {
                        "date": today,
                        "count": symbol_count,
                        "minimum_governed_symbol_count": minimum_count,
                        "max_symbol_lead": max_symbol_lead,
                        "quota_key": f"symbol_fairness:{symbol_key}",
                        "symbol_targets": symbol_targets,
                    }
                )
                return PaperSamplingDecision(
                    True,
                    False,
                    decision.qty,
                    "PAPER_SAMPLING_SYMBOL_FAIRNESS_BLOCK",
                    details,
                )

            reserved_minima = _session_reserved_minima(cfg)
            session_index = _REGULAR_SESSIONS.index(session)
            future_reserved = sum(
                int(reserved_minima.get(future_session, 0))
                for future_session in _REGULAR_SESSIONS[session_index + 1 :]
            )
            capacity_before_future = max(0, max_trades - future_reserved)
            if count >= capacity_before_future:
                details = dict(decision.details)
                details.update(
                    {
                        "date": today,
                        "count": count,
                        "max_trades_per_day": max_trades,
                        "session_bucket": session,
                        "future_reserved_slots": future_reserved,
                        "reserved_session_minima": reserved_minima,
                    }
                )
                return PaperSamplingDecision(
                    True,
                    False,
                    decision.qty,
                    "PAPER_SAMPLING_FUTURE_SESSION_RESERVATION_BLOCK",
                    details,
                )
        else:
            reserved_minima = _session_reserved_minima(cfg)

        observed_by_symbol = _count_map(current_state, "observed_by_symbol")
        observed_by_side = _count_map(current_state, "observed_by_side")
        observed_by_role = _count_map(current_state, "observed_by_role")
        observed_by_session = _count_map(current_state, "observed_by_session")
        observed_by_side_role = _count_map(current_state, "observed_by_side_role")
        observed_by_symbol_session = _count_map(
            current_state,
            "observed_by_symbol_session",
        )
        observed_by_stratum = _count_map(current_state, "observed_by_stratum")
        side_role_key = f"{side_key}:{role_key}"
        symbol_session_key = f"{symbol_key}:{session}"
        stratum_key = f"{symbol_key}:{side_key}:{role_key}:{session}"
        reservation_token = uuid4().hex
        reservations = _reservation_rows(current_state)
        reservations.append(
            {
                "reservation_token": reservation_token,
                "symbol": symbol_key,
                "side": side_key,
                "role": role_key,
                "session_bucket": session,
                "consumes_daily_slot": bool(consumes_daily_slot),
                "reserved_at": (now or datetime.now(UTC)).isoformat(),
            }
        )
        next_count = count + 1 if consumes_daily_slot else count
        next_by_symbol = (
            _increment_count(by_symbol, symbol_key)
            if consumes_daily_slot
            else by_symbol
        )
        next_by_side = (
            _increment_count(by_side, side_key)
            if consumes_daily_slot
            else by_side
        )
        next_by_session = (
            _increment_count(by_session, session)
            if consumes_daily_slot
            else by_session
        )
        symbol_deficits = {
            candidate: max(
                0,
                int(target) - int(next_by_symbol.get(candidate, 0)),
            )
            for candidate, target in symbol_targets.items()
        }
        state = {
            "schema_version": "2.0.0",
            "artifact_type": "paper_sampling_state",
            "date": today,
            "count": next_count,
            "by_symbol": next_by_symbol,
            "by_side": next_by_side,
            "by_session": next_by_session,
            "observed_count": int(current_state.get("observed_count", 0) or 0) + 1,
            "observed_by_symbol": _increment_count(
                observed_by_symbol,
                symbol_key,
            ),
            "observed_by_side": _increment_count(observed_by_side, side_key),
            "observed_by_role": _increment_count(observed_by_role, role_key),
            "observed_by_session": _increment_count(
                observed_by_session,
                session,
            ),
            "observed_by_side_role": _increment_count(
                observed_by_side_role,
                side_role_key,
            ),
            "observed_by_symbol_session": _increment_count(
                observed_by_symbol_session,
                symbol_session_key,
            ),
            "observed_by_stratum": _increment_count(
                observed_by_stratum,
                stratum_key,
            ),
            "symbol_targets": symbol_targets,
            "symbol_deficits": symbol_deficits,
            "reserved_session_minima": reserved_minima,
            "stratified_fairness_enabled": fairness_enabled,
            "reservations": reservations,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _write_state(path, state)
    details = dict(decision.details)
    details.update(
        {
            "date": today,
            "count": next_count,
            "max_trades_per_day": max_trades,
            "session_bucket": session,
            "role": role_key,
            "reservation_token": reservation_token,
            "symbol_targets": symbol_targets,
            "symbol_deficits": symbol_deficits,
            "reserved_session_minima": reserved_minima,
            "stratified_fairness_enabled": fairness_enabled,
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
    role: str | None = None,
    reservation_token: str | None = None,
) -> None:
    """Release a diagnostic paper-sampling slot when submit is not accepted."""

    active, _ = _is_paper_sampling_active(cfg)
    if not active:
        return
    today = _today_key(now)
    symbol_key = str(symbol).strip().upper()
    side_key = str(side).strip().lower()
    role_key = _sampling_role(role, consumes_daily_slot=consumes_daily_slot)
    if not symbol_key or not side_key:
        return
    path = _state_path()
    with _STATE_LOCK:
        state = _load_state(path)
        if str(state.get("date") or "") != today:
            return
        reservations = _reservation_rows(state)
        matched_index: int | None = None
        token_key = str(reservation_token or "").strip()
        for index in range(len(reservations) - 1, -1, -1):
            row = reservations[index]
            if token_key:
                if str(row.get("reservation_token") or "") == token_key:
                    matched_index = index
                    break
                continue
            if (
                str(row.get("symbol") or "").strip().upper() == symbol_key
                and str(row.get("side") or "").strip().lower() == side_key
                and str(row.get("role") or "").strip().lower() == role_key
                and bool(row.get("consumes_daily_slot"))
                == bool(consumes_daily_slot)
            ):
                matched_index = index
                break

        if matched_index is None:
            if not consumes_daily_slot:
                return
            # Legacy v1 state has no reservation records. Retain its release
            # behavior while all v2 reservations use their original session.
            session = _session_bucket(now)
            state["count"] = max(0, int(state.get("count", 0) or 0) - 1)
            state["by_symbol"] = _decrement_count(
                _count_map(state, "by_symbol"),
                symbol_key,
            )
            state["by_side"] = _decrement_count(
                _count_map(state, "by_side"),
                side_key,
            )
            state["by_session"] = _decrement_count(
                _count_map(state, "by_session"),
                session,
            )
            state["updated_at"] = datetime.now(UTC).isoformat()
            _write_state(path, state)
            return

        reservation = reservations.pop(matched_index)
        reserved_symbol = str(reservation.get("symbol") or symbol_key).strip().upper()
        reserved_side = str(reservation.get("side") or side_key).strip().lower()
        reserved_role = str(reservation.get("role") or role_key).strip().lower()
        reserved_session = str(
            reservation.get("session_bucket") or _session_bucket(now)
        ).strip().lower()
        reserved_consumes = bool(reservation.get("consumes_daily_slot"))

        if reserved_consumes:
            state["count"] = max(0, int(state.get("count", 0) or 0) - 1)
            state["by_symbol"] = _decrement_count(
                _count_map(state, "by_symbol"),
                reserved_symbol,
            )
            state["by_side"] = _decrement_count(
                _count_map(state, "by_side"),
                reserved_side,
            )
            state["by_session"] = _decrement_count(
                _count_map(state, "by_session"),
                reserved_session,
            )

        side_role_key = f"{reserved_side}:{reserved_role}"
        symbol_session_key = f"{reserved_symbol}:{reserved_session}"
        stratum_key = (
            f"{reserved_symbol}:{reserved_side}:{reserved_role}:{reserved_session}"
        )
        state["observed_count"] = max(
            0,
            int(state.get("observed_count", 0) or 0) - 1,
        )
        state["observed_by_symbol"] = _decrement_count(
            _count_map(state, "observed_by_symbol"),
            reserved_symbol,
        )
        state["observed_by_side"] = _decrement_count(
            _count_map(state, "observed_by_side"),
            reserved_side,
        )
        state["observed_by_role"] = _decrement_count(
            _count_map(state, "observed_by_role"),
            reserved_role,
        )
        state["observed_by_session"] = _decrement_count(
            _count_map(state, "observed_by_session"),
            reserved_session,
        )
        state["observed_by_side_role"] = _decrement_count(
            _count_map(state, "observed_by_side_role"),
            side_role_key,
        )
        state["observed_by_symbol_session"] = _decrement_count(
            _count_map(state, "observed_by_symbol_session"),
            symbol_session_key,
        )
        state["observed_by_stratum"] = _decrement_count(
            _count_map(state, "observed_by_stratum"),
            stratum_key,
        )
        targets = _count_map(state, "symbol_targets")
        current_by_symbol = _count_map(state, "by_symbol")
        state["symbol_deficits"] = {
            candidate: max(
                0,
                int(target) - int(current_by_symbol.get(candidate, 0)),
            )
            for candidate, target in targets.items()
        }
        state["reservations"] = reservations
        state["updated_at"] = datetime.now(UTC).isoformat()
        _write_state(path, state)


__all__ = [
    "PaperSamplingDecision",
    "evaluate_paper_sampling_order",
    "paper_sampling_deficit_snapshot",
    "release_paper_sampling_order",
    "reserve_paper_sampling_order",
]
