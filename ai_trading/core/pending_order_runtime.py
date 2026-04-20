"""Pending-order runtime helpers extracted from ``bot_engine.py``."""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, cast


def _bot_engine() -> Any:
    return importlib.import_module("ai_trading.core.bot_engine")


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def record_pending_order_slo_metrics(
    *,
    pending_count: int,
    oldest_pending_age_s: float | None,
) -> None:
    """Record pending backlog SLO metrics without raising."""

    be = _bot_engine()
    try:
        from ai_trading.monitoring.slo import get_slo_monitor

        monitor = get_slo_monitor()
        monitor.record_metric("pending_orders_count", float(max(int(pending_count), 0)))
        monitor.record_metric(
            "pending_oldest_age_sec",
            float(max(float(oldest_pending_age_s or 0.0), 0.0)),
        )
    except Exception:
        be.logger.debug("PENDING_ORDER_SLO_RECORD_FAILED", exc_info=True)


def maybe_apply_pending_stale_sweep(
    *,
    runtime: Any,
    pending_orders: Iterable[Any],
    now_dt: datetime,
    now_ts: float,
) -> dict[str, Any] | None:
    """Attempt bounded cancellation of stale pending orders."""

    be = _bot_engine()
    if not be._pending_stale_sweep_enabled():
        return None
    max_cancels = be._pending_stale_sweep_max_cancels()
    if max_cancels <= 0:
        return None

    runtime_state_map = be._ensure_runtime_state(runtime)
    cooldown_s = be._pending_stale_sweep_cooldown_seconds()
    last_run_raw = runtime_state_map.get(be._PENDING_STALE_SWEEP_LAST_TS_KEY)
    try:
        last_run_ts = float(last_run_raw) if last_run_raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        last_run_ts = 0.0
    if cooldown_s > 0.0 and last_run_ts > 0.0 and (now_ts - last_run_ts) < cooldown_s:
        return None

    stale_after_s = be._pending_stale_sweep_age_seconds()
    include_partial_fills = be._pending_stale_sweep_include_partially_filled()
    partial_fill_stale_after_s = be._pending_stale_sweep_partial_fill_age_seconds(
        stale_after_s
    )
    stale_candidates: list[tuple[float, Any]] = []
    for order in pending_orders:
        status = be._normalize_broker_order_status(getattr(order, "status", None))
        age_s = be._pending_order_broker_age_seconds(order, now_dt)
        if age_s is None:
            continue
        if status in be._PENDING_ORDER_STUCK_STATUSES:
            if age_s < stale_after_s:
                continue
        elif bool(include_partial_fills) and status == "partially_filled":
            if age_s < partial_fill_stale_after_s:
                continue
        else:
            continue
        stale_candidates.append((float(age_s), order))
    if not stale_candidates:
        return None

    stale_candidates.sort(key=lambda item: item[0], reverse=True)
    selected_orders = [order for _age, order in stale_candidates[:max_cancels]]
    cancel_result = be._cancel_open_orders_subset(
        runtime,
        orders=selected_orders,
        reason_code="PENDING_STALE_SWEEP",
    )
    runtime_state_map[be._PENDING_STALE_SWEEP_LAST_TS_KEY] = float(now_ts)
    selected_ids: list[str] = []
    for order in selected_orders:
        order_id = be._extract_order_identifier(order)
        if order_id:
            selected_ids.append(order_id)
    return {
        "stale_after_s": int(stale_after_s),
        "partial_fill_stale_after_s": int(partial_fill_stale_after_s),
        "include_partial_fills": bool(include_partial_fills),
        "attempted": int(len(selected_orders)),
        "cancelled": int(cancel_result.cancelled),
        "failed": int(cancel_result.failed),
        "selected_ids": selected_ids[: be._PENDING_ORDER_SAMPLE_LIMIT],
        "oldest_candidate_age_s": int(max(stale_candidates[0][0], 0.0)),
        "errors": list(cancel_result.errors or []),
    }


def _extract_pending_order_symbol(order: Any) -> str | None:
    symbol_raw: Any = None
    if isinstance(order, Mapping):
        symbol_raw = order.get("symbol")
    if symbol_raw in (None, ""):
        symbol_raw = getattr(order, "symbol", None)
    if symbol_raw in (None, ""):
        return None
    symbol = str(symbol_raw).strip().upper()
    return symbol or None


def _collect_pending_blocked_symbols(orders: Iterable[Any]) -> set[str]:
    blocked: set[str] = set()
    for order in orders:
        symbol = _extract_pending_order_symbol(order)
        if symbol:
            blocked.add(symbol)
    return blocked


def set_pending_blocked_symbols(runtime: Any, symbols: Iterable[str]) -> None:
    """Persist normalized pending-blocked symbols on runtime and global state."""

    be = _bot_engine()
    normalized = sorted({str(sym).strip().upper() for sym in symbols if str(sym).strip()})
    setattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR, tuple(normalized))
    state_obj = getattr(be, "state", None)
    if state_obj is not None:
        try:
            setattr(state_obj, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR, tuple(normalized))
        except Exception:
            be.logger.debug("PENDING_BLOCKED_SYMBOLS_STATE_SET_FAILED", exc_info=True)


def resolve_runtime_info_log_ttl_seconds(
    env_name: str,
    default_seconds: float,
) -> float:
    """Resolve throttled INFO cadence from managed environment config."""

    be = _bot_engine()
    try:
        ttl = float(be.get_env(env_name, default_seconds, cast=float))
    except be.COMMON_EXC:
        ttl = float(default_seconds)
    if not math.isfinite(ttl):
        ttl = float(default_seconds)
    return max(0.0, min(ttl, 3600.0))


def should_emit_runtime_info_log(
    runtime: Any,
    key: str,
    *,
    ttl_seconds: float,
    now_mono: float | None = None,
) -> bool:
    """Return ``True`` when INFO log ``key`` should emit under TTL coalescing."""

    be = _bot_engine()
    ttl = max(float(ttl_seconds), 0.0)
    if ttl <= 0.0:
        return True
    state = be._ensure_runtime_state(runtime)
    tracker_raw = state.get(be._RUNTIME_INFO_LOG_TRACKER_KEY)
    tracker: dict[str, float]
    if isinstance(tracker_raw, dict):
        tracker = tracker_raw
    else:
        tracker = {}
        state[be._RUNTIME_INFO_LOG_TRACKER_KEY] = tracker
    now_value = float(now_mono if now_mono is not None else be.monotonic_time())
    try:
        last_value = float(tracker.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        last_value = 0.0
    if last_value <= 0.0 or now_value - last_value >= ttl:
        tracker[key] = now_value
        return True
    return False


def _get_pending_symbol_decay_tracker(runtime: Any) -> dict[str, dict[str, Any]]:
    be = _bot_engine()
    state = be._ensure_runtime_state(runtime)
    tracker = state.get(be._PENDING_SYMBOL_DECAY_TRACKER_KEY)
    if not isinstance(tracker, dict):
        tracker = {}
        state[be._PENDING_SYMBOL_DECAY_TRACKER_KEY] = tracker
    return cast(dict[str, dict[str, Any]], tracker)


def _pending_symbol_decay_config(*, force_cleanup_after: float) -> dict[str, Any]:
    be = _bot_engine()
    try:
        enabled = bool(
            be.get_env("AI_TRADING_PENDING_SYMBOL_DECAY_ENABLED", False, cast=bool)
        )
    except be.COMMON_EXC:
        enabled = False

    try:
        ack_timeout_s = float(be.get_env("ORDER_ACK_TIMEOUT_SECONDS", 20.0, cast=float))
    except be.COMMON_EXC:
        ack_timeout_s = 20.0
    ack_timeout_s = max(1.0, min(ack_timeout_s, 3600.0))

    try:
        ack_mult = float(
            be.get_env(
                "AI_TRADING_PENDING_SYMBOL_DECAY_ACK_MULT",
                be._PENDING_SYMBOL_DECAY_ACK_MULT_DEFAULT,
                cast=float,
            )
        )
    except be.COMMON_EXC:
        ack_mult = be._PENDING_SYMBOL_DECAY_ACK_MULT_DEFAULT
    ack_mult = max(1.0, min(ack_mult, 100.0))

    try:
        min_sec = float(
            be.get_env(
                "AI_TRADING_PENDING_SYMBOL_DECAY_MIN_SEC",
                be._PENDING_SYMBOL_DECAY_MIN_SEC_DEFAULT,
                cast=float,
            )
        )
    except be.COMMON_EXC:
        min_sec = be._PENDING_SYMBOL_DECAY_MIN_SEC_DEFAULT
    min_sec = max(1.0, min(min_sec, 86400.0))

    try:
        max_sec = float(
            be.get_env(
                "AI_TRADING_PENDING_SYMBOL_DECAY_MAX_SEC",
                force_cleanup_after,
                cast=float,
            )
        )
    except be.COMMON_EXC:
        max_sec = float(force_cleanup_after)
    max_sec = max(min_sec, min(max_sec, 86400.0))

    release_after_s = max(min_sec, ack_timeout_s * ack_mult)
    release_after_s = min(release_after_s, max_sec)

    try:
        min_cycles = int(
            be.get_env(
                "AI_TRADING_PENDING_SYMBOL_DECAY_MIN_CYCLES",
                be._PENDING_SYMBOL_DECAY_MIN_CYCLES_DEFAULT,
                cast=int,
            )
        )
    except be.COMMON_EXC:
        min_cycles = be._PENDING_SYMBOL_DECAY_MIN_CYCLES_DEFAULT
    min_cycles = max(1, min(min_cycles, 120))

    try:
        release_cooldown_s = float(
            be.get_env(
                "AI_TRADING_PENDING_SYMBOL_RELEASE_COOLDOWN_SEC",
                be._PENDING_SYMBOL_RELEASE_COOLDOWN_SEC_DEFAULT,
                cast=float,
            )
        )
    except be.COMMON_EXC:
        release_cooldown_s = be._PENDING_SYMBOL_RELEASE_COOLDOWN_SEC_DEFAULT
    release_cooldown_s = max(0.0, min(release_cooldown_s, 86400.0))

    return {
        "enabled": enabled,
        "ack_timeout_s": ack_timeout_s,
        "ack_mult": ack_mult,
        "release_after_s": release_after_s,
        "min_cycles": min_cycles,
        "release_cooldown_s": release_cooldown_s,
    }


def apply_pending_symbol_block_decay(
    runtime: Any,
    raw_symbols: Iterable[str],
    *,
    symbol_oldest_age_s: Mapping[str, float],
    symbol_open_order_count: Mapping[str, int],
    symbol_oldest_open_age_s: Mapping[str, float],
    symbol_statuses: Mapping[str, set[str]],
    now: float,
    block_scope: str,
    force_cleanup_after: float,
) -> tuple[set[str], dict[str, Any] | None]:
    """Return effective blocked symbols after optional stale-age decay."""

    be = _bot_engine()
    raw_blocked = sorted(
        {str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()}
    )
    if block_scope != "symbol":
        return set(raw_blocked), None

    tracker = _get_pending_symbol_decay_tracker(runtime)
    config = _pending_symbol_decay_config(force_cleanup_after=force_cleanup_after)
    decay_enabled = bool(config["enabled"])
    release_after_s = float(config["release_after_s"])
    min_cycles_required = int(config["min_cycles"])
    try:
        sample_limit = int(
            be.get_env(
                "AI_TRADING_PENDING_SYMBOL_COOLDOWN_TELEMETRY_SAMPLE_LIMIT",
                be._PENDING_SYMBOL_COOLDOWN_TELEMETRY_SAMPLE_LIMIT_DEFAULT,
                cast=int,
            )
        )
    except be.COMMON_EXC:
        sample_limit = be._PENDING_SYMBOL_COOLDOWN_TELEMETRY_SAMPLE_LIMIT_DEFAULT
    sample_limit = max(1, min(sample_limit, be._PENDING_ORDER_SAMPLE_LIMIT))

    effective_blocked: set[str] = set()
    released_symbols: list[str] = []
    cooldown_symbols: list[str] = []
    symbol_states: list[dict[str, Any]] = []
    now_s = float(now)

    for symbol in raw_blocked:
        raw_entry = tracker.get(symbol)
        entry = raw_entry if isinstance(raw_entry, dict) else {}

        raw_first_seen = entry.get("first_seen_ts", now_s)
        first_seen_ts = _safe_float(raw_first_seen)
        if first_seen_ts is None or first_seen_ts <= 0.0:
            first_seen_ts = now_s

        raw_cycles = entry.get("cycles_seen", 0)
        try:
            cycles_seen = int(raw_cycles)
        except (TypeError, ValueError):
            cycles_seen = 0
        cycles_seen = max(cycles_seen + 1, 1)

        prev_max_age_s = _safe_float(entry.get("max_age_s", 0.0))
        prev_max_age_s = max(prev_max_age_s or 0.0, 0.0)

        current_age_s = max(float(symbol_oldest_age_s.get(symbol, 0.0) or 0.0), 0.0)
        max_age_s = max(prev_max_age_s, current_age_s)
        open_orders_count = int(symbol_open_order_count.get(symbol, 0) or 0)
        oldest_open_age_s = float(symbol_oldest_open_age_s.get(symbol, 0.0) or 0.0)

        statuses = {
            str(status).strip().lower()
            for status in symbol_statuses.get(symbol, set())
            if str(status).strip()
        }
        statuses_stuck = (not statuses) or statuses.issubset(be._PENDING_ORDER_STUCK_STATUSES)

        released_until_ts = _safe_float(entry.get("released_until_ts", 0.0)) or 0.0

        entry.update(
            {
                "first_seen_ts": first_seen_ts,
                "last_seen_ts": now_s,
                "cycles_seen": cycles_seen,
                "max_age_s": max_age_s,
                "statuses": sorted(statuses),
            }
        )
        statuses_sample = sorted(statuses)[:4]

        if released_until_ts > now_s:
            cooldown_symbols.append(symbol)
            tracker[symbol] = entry
            if len(symbol_states) < sample_limit:
                symbol_states.append(
                    {
                        "symbol": symbol,
                        "state": "cooldown",
                        "cycles_seen": cycles_seen,
                        "max_age_s": round(max_age_s, 3),
                        "release_after_s": round(release_after_s, 3),
                        "age_to_release_s": round(max(release_after_s - max_age_s, 0.0), 3),
                        "cooldown_remaining_s": round(max(released_until_ts - now_s, 0.0), 3),
                        "statuses": statuses_sample,
                        "statuses_stuck": bool(statuses_stuck),
                        "eligible_for_release": False,
                    }
                )
            continue

        if (
            decay_enabled
            and statuses_stuck
            and cycles_seen >= min_cycles_required
            and max_age_s >= release_after_s
            and open_orders_count <= 0
        ):
            released_symbols.append(symbol)
            release_count_raw = entry.get("release_count", 0)
            try:
                release_count = int(release_count_raw)
            except (TypeError, ValueError):
                release_count = 0
            entry["release_count"] = max(release_count + 1, 1)
            entry["last_released_ts"] = now_s
            cooldown_s = float(config["release_cooldown_s"])
            entry["released_until_ts"] = now_s + cooldown_s if cooldown_s > 0.0 else now_s
            tracker[symbol] = entry
            if len(symbol_states) < sample_limit:
                symbol_states.append(
                    {
                        "symbol": symbol,
                        "state": "released",
                        "cycles_seen": cycles_seen,
                        "max_age_s": round(max_age_s, 3),
                        "release_after_s": round(release_after_s, 3),
                        "age_to_release_s": 0.0,
                        "cooldown_remaining_s": round(
                            max(float(entry["released_until_ts"]) - now_s, 0.0),
                            3,
                        ),
                        "statuses": statuses_sample,
                        "statuses_stuck": bool(statuses_stuck),
                        "eligible_for_release": True,
                    }
                )
            continue

        if (
            decay_enabled
            and statuses_stuck
            and cycles_seen >= min_cycles_required
            and max_age_s >= release_after_s
            and open_orders_count > 0
        ):
            be.logger.warning(
                "PENDING_SYMBOL_BLOCK_DECAY_DEFERRED",
                extra={
                    "symbol": symbol,
                    "open_orders_count": open_orders_count,
                    "oldest_open_order_age_s": round(max(oldest_open_age_s, 0.0), 3),
                },
            )
            entry["released_until_ts"] = 0.0
            tracker[symbol] = entry
            effective_blocked.add(symbol)
            if len(symbol_states) < sample_limit:
                symbol_states.append(
                    {
                        "symbol": symbol,
                        "state": "deferred",
                        "cycles_seen": cycles_seen,
                        "max_age_s": round(max_age_s, 3),
                        "release_after_s": round(release_after_s, 3),
                        "age_to_release_s": 0.0,
                        "cooldown_remaining_s": 0.0,
                        "statuses": statuses_sample,
                        "statuses_stuck": bool(statuses_stuck),
                        "eligible_for_release": False,
                        "open_orders_count": open_orders_count,
                        "oldest_open_order_age_s": round(max(oldest_open_age_s, 0.0), 3),
                    }
                )
            continue

        entry["released_until_ts"] = 0.0
        tracker[symbol] = entry
        effective_blocked.add(symbol)
        if len(symbol_states) < sample_limit:
            eligible_for_release = (
                decay_enabled
                and statuses_stuck
                and cycles_seen >= min_cycles_required
                and max_age_s >= release_after_s
            )
            symbol_states.append(
                {
                    "symbol": symbol,
                    "state": "blocked",
                    "cycles_seen": cycles_seen,
                    "max_age_s": round(max_age_s, 3),
                    "release_after_s": round(release_after_s, 3),
                    "age_to_release_s": round(max(release_after_s - max_age_s, 0.0), 3),
                    "cooldown_remaining_s": 0.0,
                    "statuses": statuses_sample,
                    "statuses_stuck": bool(statuses_stuck),
                    "eligible_for_release": bool(eligible_for_release),
                }
            )

    for symbol in list(tracker.keys()):
        if symbol in raw_blocked:
            continue
        entry = tracker.get(symbol)
        if not isinstance(entry, dict):
            tracker.pop(symbol, None)
            continue
        released_until_ts = _safe_float(entry.get("released_until_ts", 0.0)) or 0.0
        if released_until_ts > now_s:
            continue
        tracker.pop(symbol, None)

    if not raw_blocked:
        return set(), None

    telemetry: dict[str, Any] = {
        "raw_blocked_count": len(raw_blocked),
        "effective_blocked_count": len(effective_blocked),
        "released_count": len(released_symbols),
        "cooldown_active_count": len(cooldown_symbols),
        "release_after_s": round(float(config["release_after_s"]), 3),
        "min_cycles": int(config["min_cycles"]),
        "cooldown_s": round(float(config["release_cooldown_s"]), 3),
        "ack_timeout_s": round(float(config["ack_timeout_s"]), 3),
        "ack_mult": round(float(config["ack_mult"]), 3),
        "telemetry_sample_limit": sample_limit,
    }
    if released_symbols:
        telemetry["released_symbols"] = released_symbols[: be._PENDING_ORDER_SAMPLE_LIMIT]
    if cooldown_symbols:
        telemetry["cooldown_symbols"] = cooldown_symbols[: be._PENDING_ORDER_SAMPLE_LIMIT]
    if symbol_states:
        telemetry["symbol_states"] = symbol_states
    return effective_blocked, telemetry


def apply_pending_new_timeout_policy(runtime: Any) -> bool:
    """Best-effort per-order pending policy action using execution engine hooks."""

    be = _bot_engine()
    engine = getattr(runtime, "execution_engine", None) or getattr(runtime, "exec_engine", None)
    if engine is None:
        return False
    policy_hook = getattr(engine, "_apply_pending_new_timeout_policy", None)
    if not callable(policy_hook):
        return False
    try:
        result = policy_hook()
    except be.COMMON_EXC as exc:
        be.logger.warning(
            "PENDING_NEW_POLICY_APPLY_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
            exc_info=True,
        )
        return False
    if isinstance(result, bool):
        return result
    return True


def handle_pending_orders(open_orders: Iterable[Any], runtime: Any) -> bool:
    """Handle pending orders and decide whether to skip the current cycle."""

    be = _bot_engine()
    if isinstance(open_orders, list):
        open_list = open_orders
    else:
        open_list = list(open_orders)

    api = getattr(runtime, "api", None)
    confirmed_pending = (
        be.get_confirmed_pending_orders(
            api,
            open_list,
            require_confirmation=False,
        )
        if api is not None
        else []
    )

    if not confirmed_pending and open_list:
        confirmed_pending = open_list

    blocked_symbols = _collect_pending_blocked_symbols(confirmed_pending)

    now = be.time.time()
    now_dt = datetime.fromtimestamp(now, tz=UTC)

    open_count = 0
    counts_by_status: dict[str, int] = {}
    oldest_open_age_s: float | None = None
    symbol_open_order_count: dict[str, int] = {}
    symbol_oldest_open_age_s: dict[str, float] = {}
    sample_candidates: list[dict[str, Any]] = []
    for order in open_list:
        if isinstance(order, Mapping):
            status_raw = order.get("status")
        else:
            status_raw = getattr(order, "status", None)
        status = be._normalize_broker_order_status(status_raw)
        if status in be._OPEN_ORDER_TERMINAL_STATUSES:
            continue
        open_count += 1
        status_key = status or "unknown"
        counts_by_status[status_key] = int(counts_by_status.get(status_key, 0)) + 1

        symbol = _extract_pending_order_symbol(order)
        if symbol:
            symbol_open_order_count[symbol] = int(symbol_open_order_count.get(symbol, 0)) + 1

        age_s = be._pending_order_broker_age_seconds(order, now_dt)
        if age_s is not None:
            if oldest_open_age_s is None or age_s > oldest_open_age_s:
                oldest_open_age_s = age_s
            if symbol:
                symbol_oldest_open_age_s[symbol] = max(
                    symbol_oldest_open_age_s.get(symbol, 0.0),
                    float(age_s),
                )

        order_id = None
        if isinstance(order, Mapping):
            order_id = order.get("id") or order.get("order_id") or order.get("client_order_id")
            submitted_at = order.get("submitted_at")
            updated_at = order.get("updated_at")
        else:
            order_id = (
                getattr(order, "id", None)
                or getattr(order, "order_id", None)
                or getattr(order, "client_order_id", None)
            )
            submitted_at = getattr(order, "submitted_at", None)
            updated_at = getattr(order, "updated_at", None)
        sample_candidates.append(
            {
                "order_id": str(order_id) if order_id not in (None, "") else None,
                "symbol": symbol,
                "status": status_key,
                "submitted_at": str(submitted_at) if submitted_at not in (None, "") else None,
                "updated_at": str(updated_at) if updated_at not in (None, "") else None,
                "age_s": age_s,
            }
        )

    def _sample_sort_key(entry: Mapping[str, Any]) -> tuple[int, float, str]:
        age_val = entry.get("age_s")
        if isinstance(age_val, (int, float)):
            return (0, -float(age_val), str(entry.get("order_id") or ""))
        return (1, 0.0, str(entry.get("order_id") or ""))

    sample_orders: list[dict[str, Any]] = []
    for entry in sorted(sample_candidates, key=_sample_sort_key)[:5]:
        sample_orders.append(
            {
                "order_id": entry.get("order_id"),
                "symbol": entry.get("symbol"),
                "status": entry.get("status"),
                "submitted_at": entry.get("submitted_at"),
                "updated_at": entry.get("updated_at"),
                "age_s": (
                    round(float(entry["age_s"]), 3)
                    if isinstance(entry.get("age_s"), (int, float))
                    else None
                ),
            }
        )
    affected_symbols_count = len({sym for sym in symbol_open_order_count if sym})

    pending_ids: list[str] = []
    pending_statuses: set[str] = set()
    oldest_pending_age_s: float | None = None
    oldest_stuck_age_s: float | None = None
    stuck_pending_count = 0
    symbol_oldest_age_s: dict[str, float] = {}
    symbol_statuses: dict[str, set[str]] = {}
    for order in confirmed_pending:
        status = be._normalize_broker_order_status(getattr(order, "status", None))
        symbol = _extract_pending_order_symbol(order)
        pending_ids.append(str(getattr(order, "id", "?")))
        if status:
            pending_statuses.add(status)
            if symbol:
                symbol_statuses.setdefault(symbol, set()).add(status)
            if status in be._PENDING_ORDER_STUCK_STATUSES:
                stuck_pending_count += 1
        broker_age_s = be._pending_order_broker_age_seconds(order, now_dt)
        if broker_age_s is not None:
            if oldest_pending_age_s is None or broker_age_s > oldest_pending_age_s:
                oldest_pending_age_s = broker_age_s
            if status in be._PENDING_ORDER_STUCK_STATUSES and (
                oldest_stuck_age_s is None or broker_age_s > oldest_stuck_age_s
            ):
                oldest_stuck_age_s = broker_age_s
            if symbol:
                symbol_oldest_age_s[symbol] = max(
                    symbol_oldest_age_s.get(symbol, 0.0),
                    float(broker_age_s),
                )

    tracker = be._get_pending_tracker(runtime)
    first_seen = _safe_float(tracker.get(be._PENDING_ORDER_FIRST_SEEN_KEY))
    last_log = _safe_float(tracker.get(be._PENDING_ORDER_LAST_LOG_KEY))
    runtime_state_map = be._ensure_runtime_state(runtime)
    backlog_active = bool(runtime_state_map.get(be._PENDING_BACKLOG_ACTIVE_KEY, False))
    last_warn_ts = _safe_float(runtime_state_map.get(be._PENDING_BACKLOG_LAST_WARN_TS_KEY))

    if not pending_ids:
        be._record_pending_order_slo_metrics(pending_count=0, oldest_pending_age_s=0.0)
        be._set_pending_blocked_symbols(runtime, ())
        if backlog_active or first_seen is not None:
            resolved_age = max(now - first_seen, 0.0) if first_seen is not None else 0.0
            be.logger.info(
                "PENDING_ORDERS_CLEARED",
                extra={
                    "open_count": int(open_count),
                    "pending_count": 0,
                    "counts_by_status": counts_by_status,
                    "oldest_open_age_s": None,
                    "affected_symbols_count": 0,
                    "sample_orders": [],
                    "resolved_age_s": int(max(resolved_age, 0.0)),
                },
            )
        tracker[be._PENDING_ORDER_FIRST_SEEN_KEY] = None
        tracker[be._PENDING_ORDER_LAST_LOG_KEY] = None
        runtime_state_map[be._PENDING_BACKLOG_ACTIVE_KEY] = False
        runtime_state_map[be._PENDING_BACKLOG_LAST_WARN_TS_KEY] = None
        if be._consume_pending_cleanup_warmup(runtime, open_count=len(open_list)):
            return True
        return False

    try:
        cfg = be.get_trading_config()
    except be.COMMON_EXC:
        cfg = None
    cfg_interval = getattr(cfg, "order_stale_cleanup_interval", 120)
    warn_after_s = be._pending_orders_warn_after_seconds()
    warn_every_s = be._pending_orders_warn_every_seconds()
    block_scope = be._pending_orders_block_scope()
    try:
        cleanup_after = float(cfg_interval)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        cleanup_after = 120.0
    cleanup_after = max(5.0, min(cleanup_after, 3600.0))
    force_cleanup_after = be._pending_order_force_cleanup_seconds()
    if pending_statuses and pending_statuses.issubset(be._PENDING_ORDER_STUCK_STATUSES):
        cleanup_after = min(cleanup_after, force_cleanup_after)
    stale_stuck_detected = (
        oldest_stuck_age_s is not None and oldest_stuck_age_s >= force_cleanup_after
    )
    if stale_stuck_detected:
        cleanup_after = min(cleanup_after, force_cleanup_after)

    had_blocked_symbols = bool(blocked_symbols)
    blocked_symbols, decay_telemetry = be._apply_pending_symbol_block_decay(
        runtime,
        blocked_symbols,
        symbol_oldest_age_s=symbol_oldest_age_s,
        symbol_open_order_count=symbol_open_order_count,
        symbol_oldest_open_age_s=symbol_oldest_open_age_s,
        symbol_statuses=symbol_statuses,
        now=now,
        block_scope=block_scope,
        force_cleanup_after=force_cleanup_after,
    )
    be._set_pending_blocked_symbols(runtime, blocked_symbols)
    if decay_telemetry is not None:
        decay_payload = dict(decay_telemetry)
        if decay_payload.get("released_count", 0) > 0:
            be.logger.warning(
                "PENDING_SYMBOL_BLOCK_DECAY_RELEASED",
                extra=decay_payload,
            )
        else:
            be.log_throttled_event(
                be.logger,
                "PENDING_SYMBOL_BLOCK_DECAY_METRIC",
                level=logging.INFO,
                extra=decay_payload,
                message="PENDING_SYMBOL_BLOCK_DECAY_METRIC",
            )
        symbol_states = decay_payload.get("symbol_states")
        if isinstance(symbol_states, list) and symbol_states:
            cooldown_ttl_s = be._resolve_runtime_info_log_ttl_seconds(
                "AI_TRADING_PENDING_SYMBOL_COOLDOWN_TELEMETRY_LOG_TTL_SEC",
                be._PENDING_SYMBOL_COOLDOWN_TELEMETRY_LOG_TTL_SEC_DEFAULT,
            )
            signature_states: list[str] = []
            for entry in symbol_states[:3]:
                if not isinstance(entry, dict):
                    continue
                symbol_token = str(entry.get("symbol") or "?").strip().upper() or "?"
                state_token = str(entry.get("state") or "unknown").strip().lower() or "unknown"
                signature_states.append(f"{symbol_token}:{state_token}")
            cooldown_signature = (
                f"{int(decay_payload.get('raw_blocked_count', 0))}:"
                f"{int(decay_payload.get('effective_blocked_count', 0))}:"
                f"{int(decay_payload.get('released_count', 0))}:"
                f"{int(decay_payload.get('cooldown_active_count', 0))}"
            )
            if signature_states:
                cooldown_signature = f"{cooldown_signature}:{','.join(signature_states)}"
            if be._should_emit_runtime_info_log(
                runtime,
                f"PENDING_SYMBOL_COOLDOWN_TELEMETRY:{cooldown_signature}",
                ttl_seconds=cooldown_ttl_s,
            ):
                be.log_throttled_event(
                    be.logger,
                    f"PENDING_SYMBOL_COOLDOWN_TELEMETRY_{cooldown_signature}",
                    level=logging.INFO,
                    extra=decay_payload,
                    message="PENDING_SYMBOL_COOLDOWN_TELEMETRY",
                )

    sample_ids = pending_ids[: be._PENDING_ORDER_SAMPLE_LIMIT]
    statuses = sorted(pending_statuses)
    allow_symbol_scope_continue = (
        block_scope == "symbol" and had_blocked_symbols and not blocked_symbols
    )
    payload_base: dict[str, Any] = {
        "open_count": int(open_count),
        "pending_count": len(pending_ids),
        "pending_stuck_count": int(max(stuck_pending_count, 0)),
        "counts_by_status": dict(sorted(counts_by_status.items())),
        "oldest_open_age_s": (
            round(float(oldest_open_age_s), 3) if oldest_open_age_s is not None else None
        ),
        "affected_symbols_count": int(affected_symbols_count),
        "sample_orders": sample_orders,
        "pending_ids": sample_ids,
        "pending_statuses": statuses,
        "cleanup_after_s": int(cleanup_after),
        "warn_after_s": int(warn_after_s),
        "warn_every_s": int(warn_every_s),
        "oldest_pending_age_s": (
            int(max(oldest_pending_age_s, 0))
            if oldest_pending_age_s is not None
            else None
        ),
        "oldest_stuck_age_s": (
            int(max(oldest_stuck_age_s, 0))
            if oldest_stuck_age_s is not None
            else None
        ),
        "stale_stuck_detected": bool(stale_stuck_detected),
        "blocked_symbols_count": len(blocked_symbols),
        "blocked_symbols": sorted(blocked_symbols)[: be._PENDING_ORDER_SAMPLE_LIMIT],
        "open_count_definition": "broker-active non-terminal orders",
        "pending_count_definition": "confirmed pending-ack/stuck orders under policy tracking",
    }
    slo_pending_oldest_age_s = (
        float(oldest_stuck_age_s)
        if oldest_stuck_age_s is not None
        else 0.0
    )
    be._record_pending_order_slo_metrics(
        pending_count=int(max(stuck_pending_count, 0)),
        oldest_pending_age_s=slo_pending_oldest_age_s,
    )

    if first_seen is None:
        first_seen = now - cleanup_after if stale_stuck_detected else now
        tracker[be._PENDING_ORDER_FIRST_SEEN_KEY] = first_seen
        tracker[be._PENDING_ORDER_LAST_LOG_KEY] = now
        be.logger.info(
            "PENDING_ORDERS_DETECTED",
            extra=payload_base
            | {
                "transition": "started",
                "age_s": int(max(float(oldest_pending_age_s or 0.0), 0.0)),
            },
        )

    age = now - float(first_seen)

    if not backlog_active:
        runtime_state_map[be._PENDING_BACKLOG_ACTIVE_KEY] = True
        backlog_level = (
            logging.WARNING
            if oldest_open_age_s is not None and float(oldest_open_age_s) >= warn_after_s
            else logging.INFO
        )
        be.logger.log(
            backlog_level,
            "PENDING_ORDERS_BACKLOG_STARTED",
            extra=payload_base
            | {
                "transition": "started",
                "age_s": int(max(age, 0)),
            },
        )
        tracker[be._PENDING_ORDER_LAST_LOG_KEY] = now
        if backlog_level >= logging.WARNING:
            runtime_state_map[be._PENDING_BACKLOG_LAST_WARN_TS_KEY] = now
        else:
            runtime_state_map[be._PENDING_BACKLOG_LAST_WARN_TS_KEY] = None
    else:
        should_warn = (
            oldest_open_age_s is not None
            and float(oldest_open_age_s) >= warn_after_s
            and (
                last_warn_ts is None
                or warn_every_s <= 0.0
                or (now - float(last_warn_ts)) >= warn_every_s
            )
        )
        if should_warn:
            be.logger.warning(
                "PENDING_ORDERS_STILL_PRESENT",
                extra=payload_base
                | {
                    "transition": "heartbeat",
                    "age_s": int(max(age, 0)),
                },
            )
            tracker[be._PENDING_ORDER_LAST_LOG_KEY] = now
            runtime_state_map[be._PENDING_BACKLOG_LAST_WARN_TS_KEY] = now
        elif last_log is None or now - float(last_log) >= be._PENDING_ORDER_LOG_INTERVAL_SECONDS:
            be.logger.info(
                "PENDING_ORDERS_STILL_PRESENT",
                extra=payload_base
                | {
                    "transition": "heartbeat_info",
                    "age_s": int(max(age, 0)),
                },
            )
            tracker[be._PENDING_ORDER_LAST_LOG_KEY] = now

    stale_sweep_result = be._maybe_apply_pending_stale_sweep(
        runtime=runtime,
        pending_orders=confirmed_pending,
        now_dt=now_dt,
        now_ts=now,
    )
    if stale_sweep_result is not None:
        stale_sweep_payload = payload_base | {
            "age_s": int(max(age, 0)),
            "stale_sweep": stale_sweep_result,
        }
        if stale_sweep_result.get("cancelled", 0) > 0:
            be.logger.warning("PENDING_STALE_SWEEP_APPLIED", extra=stale_sweep_payload)
            tracker[be._PENDING_ORDER_LAST_LOG_KEY] = now
            return True
        if stale_sweep_result.get("failed", 0) > 0:
            be.logger.warning("PENDING_STALE_SWEEP_FAILED", extra=stale_sweep_payload)
            tracker[be._PENDING_ORDER_LAST_LOG_KEY] = now

    if age < cleanup_after and not stale_stuck_detected:
        if allow_symbol_scope_continue:
            return False
        return True

    if block_scope == "symbol":
        if be._apply_pending_new_timeout_policy(runtime):
            tracker[be._PENDING_ORDER_LAST_LOG_KEY] = now
            policy_payload = payload_base | {"age_s": int(max(age, 0))}
            policy_ttl_s = be._resolve_runtime_info_log_ttl_seconds(
                "AI_TRADING_PENDING_POLICY_APPLIED_LOG_TTL_SEC",
                be._PENDING_POLICY_APPLIED_LOG_TTL_SEC_DEFAULT,
            )
            policy_signature = (
                f"{policy_payload['pending_count']}:"
                f"{policy_payload['blocked_symbols_count']}:"
                f"{','.join(statuses[:3])}"
            )
            if be._should_emit_runtime_info_log(
                runtime,
                f"PENDING_ORDERS_POLICY_APPLIED:{policy_signature}",
                ttl_seconds=policy_ttl_s,
            ):
                be.log_throttled_event(
                    be.logger,
                    "PENDING_ORDERS_POLICY_APPLIED",
                    level=logging.INFO,
                    message="PENDING_ORDERS_POLICY_APPLIED",
                    extra=policy_payload,
                )
            if allow_symbol_scope_continue:
                return False
            return True

    try:
        be.cancel_all_open_orders(runtime)
    except be.COMMON_EXC as exc:  # pragma: no cover - network/API failure
        tracker[be._PENDING_ORDER_LAST_LOG_KEY] = now
        be.logger.warning(
            "PENDING_ORDERS_CLEANUP_FAILED",
            extra=payload_base
            | {
                "age_s": int(max(age, 0)),
                "detail": str(exc),
            },
            exc_info=True,
        )
        return True

    be.logger.info(
        "PENDING_ORDERS_CANCELED",
        extra=payload_base
        | {
            "canceled_ids": sample_ids,
            "age_s": int(max(age, 0)),
        },
    )
    tracker[be._PENDING_ORDER_FIRST_SEEN_KEY] = None
    tracker[be._PENDING_ORDER_LAST_LOG_KEY] = None
    runtime_state_map[be._PENDING_BACKLOG_ACTIVE_KEY] = False
    runtime_state_map[be._PENDING_BACKLOG_LAST_WARN_TS_KEY] = None
    if be._arm_pending_cleanup_warmup(
        runtime,
        source="pending_cleanup",
        open_count=len(open_list),
        pending_count=len(pending_ids),
    ):
        return True
    return False


__all__ = [
    "apply_pending_new_timeout_policy",
    "apply_pending_symbol_block_decay",
    "handle_pending_orders",
    "maybe_apply_pending_stale_sweep",
    "record_pending_order_slo_metrics",
    "resolve_runtime_info_log_ttl_seconds",
    "set_pending_blocked_symbols",
    "should_emit_runtime_info_log",
]
