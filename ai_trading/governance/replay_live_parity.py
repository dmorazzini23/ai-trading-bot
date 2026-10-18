from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from ai_trading.config.management import get_env
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.logging import get_logger
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.utils import market_calendar


REPLAY_GOVERNANCE_SCHEMA_VERSION = "3.0.0"
REPLAY_FRESHNESS_CALENDAR = "XNYS"
_EXCHANGE_TZ = ZoneInfo("America/New_York")
# Calendar-aware freshness only matters while at most one completed session has
# been missed; spans longer than this are always judged on wall-clock age.
_MAX_CALENDAR_SPAN_DAYS = 15
_CALENDAR_FALLBACK_EXCEPTIONS: tuple[type[Exception], ...] = (
    *AI_TRADING_FALLBACK_EXCEPTIONS,
    ValueError,
)

logger = get_logger(__name__)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _default_fail_closed_outside_tests() -> bool:
    return not bool(
        str(get_env("PYTEST_CURRENT_TEST", "", cast=str) or "").strip()
        or bool(get_env("PYTEST_RUNNING", False, cast=bool))
    )


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@lru_cache(maxsize=256)
def _exchange_session_close_utc(session_date: date) -> datetime | None:
    """Return the XNYS close for ``session_date`` or ``None`` when closed."""

    try:
        session = market_calendar.session_info(session_date)
    except ValueError:
        return None
    return session.end_utc.astimezone(UTC)


def _exchange_day_start_utc(session_date: date) -> datetime:
    return datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        tzinfo=_EXCHANGE_TZ,
    ).astimezone(UTC)


def _replay_session_age(ts: datetime, now: datetime) -> dict[str, Any]:
    """Measure replay artifact age on the exchange calendar.

    Hours that fall on non-trading exchange days (weekends and holidays) are
    excluded from ``session_age_hours``; on spans made only of trading days the
    result equals wall-clock age. ``sessions_since_replay`` counts exchange
    sessions whose close falls in ``(ts, now]`` and ``covered_session`` is the
    latest session that had closed when the artifact was written.
    """

    wall_age_hours = max((now - ts).total_seconds(), 0.0) / 3600.0
    start_day = ts.astimezone(_EXCHANGE_TZ).date()
    end_day = max(now, ts).astimezone(_EXCHANGE_TZ).date()
    span_days = (end_day - start_day).days
    if span_days > _MAX_CALENDAR_SPAN_DAYS:
        return {
            "basis": "wall_clock",
            "reason": "replay_age_span_exceeds_calendar_window",
            "session_age_hours": float(wall_age_hours),
            "non_trading_hours_excluded": 0.0,
            "sessions_since_replay": None,
            "covered_session": None,
        }

    excluded_seconds = 0.0
    sessions_since = 0
    for offset in range(span_days + 1):
        day = start_day + timedelta(days=offset)
        close_utc = _exchange_session_close_utc(day)
        if close_utc is None:
            window_start = max(_exchange_day_start_utc(day), ts)
            window_end = min(
                _exchange_day_start_utc(day + timedelta(days=1)),
                now,
            )
            if window_end > window_start:
                excluded_seconds += (window_end - window_start).total_seconds()
        elif ts < close_utc <= now:
            sessions_since += 1

    covered_session: date | None = None
    probe = start_day
    for _ in range(_MAX_CALENDAR_SPAN_DAYS):
        close_utc = _exchange_session_close_utc(probe)
        if close_utc is not None and close_utc <= ts:
            covered_session = probe
            break
        probe -= timedelta(days=1)

    excluded_hours = excluded_seconds / 3600.0
    return {
        "basis": "exchange_calendar",
        "reason": None,
        "session_age_hours": float(max(wall_age_hours - excluded_hours, 0.0)),
        "non_trading_hours_excluded": float(excluded_hours),
        "sessions_since_replay": int(sessions_since),
        "covered_session": (
            covered_session.isoformat() if covered_session is not None else None
        ),
    }


def _load_latest_replay_governance_snapshot() -> dict[str, Any]:
    enabled = bool(
        get_env("AI_TRADING_REPLAY_LIVE_PARITY_GATE_ENABLED", True, cast=bool)
    )
    if not enabled:
        return {"enabled": False, "available": False}

    configured_output_dir = str(
        get_env(
            "AI_TRADING_REPLAY_OUTPUT_DIR",
            "runtime/replay_outputs",
            cast=str,
            resolve_aliases=False,
        )
        or "runtime/replay_outputs"
    ).strip() or "runtime/replay_outputs"
    output_dir = resolve_runtime_artifact_path(
        configured_output_dir,
        default_relative="runtime/replay_outputs",
    )
    max_age_hours = max(
        1.0,
        min(
            _as_float(
                get_env(
                    "AI_TRADING_REPLAY_LIVE_PARITY_MAX_REPLAY_AGE_HOURS",
                    24.0,
                    cast=float,
                ),
                24.0,
            ),
            24.0 * 14.0,
        ),
    )
    max_future_skew_seconds = max(
        0.0,
        _as_float(
            get_env(
                "AI_TRADING_REPLAY_LIVE_PARITY_MAX_FUTURE_SKEW_SECONDS",
                300.0,
                cast=float,
            ),
            300.0,
        ),
    )
    candidates = list(output_dir.glob("replay_hash_*.json"))
    if not candidates:
        return {
            "enabled": True,
            "available": False,
            "fresh": False,
            "path": str(output_dir),
            "reason": "missing_replay_governance_artifact",
            "max_age_hours": float(max_age_hours),
        }

    ranked_candidates: list[
        tuple[datetime, str, Path, dict[str, Any] | None, str | None]
    ] = []
    missing_ts = datetime.fromtimestamp(0.0, UTC)
    for path in candidates:
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            ranked_candidates.append(
                (missing_ts, str(path), path, None, "replay_governance_artifact_invalid")
            )
            continue
        if not isinstance(raw_payload, Mapping):
            ranked_candidates.append(
                (missing_ts, str(path), path, None, "replay_governance_artifact_invalid")
            )
            continue
        payload = dict(raw_payload)
        payload_ts = _parse_ts(payload.get("ts"))
        ranked_candidates.append(
            (
                payload_ts if payload_ts is not None else missing_ts,
                str(path),
                path,
                payload,
                None,
            )
        )

    _, _, candidate, payload, load_error = max(
        ranked_candidates,
        key=lambda item: (item[0], item[1]),
    )
    if payload is None:
        return {
            "enabled": True,
            "available": False,
            "fresh": False,
            "path": str(candidate),
            "reason": load_error or "replay_governance_artifact_invalid",
            "max_age_hours": float(max_age_hours),
        }

    ts = _parse_ts(payload.get("ts"))
    age_hours: float | None = None
    future_skew_seconds = 0.0
    future_dated = False
    session_age: dict[str, Any] = {
        "basis": None,
        "reason": None,
        "session_age_hours": None,
        "non_trading_hours_excluded": None,
        "sessions_since_replay": None,
        "covered_session": None,
    }
    if ts is not None:
        now = _utcnow()
        delta_seconds = (now - ts).total_seconds()
        future_skew_seconds = max(-delta_seconds, 0.0)
        future_dated = bool(future_skew_seconds > float(max_future_skew_seconds))
        age_hours = max(delta_seconds, 0.0) / 3600.0
        try:
            session_age = _replay_session_age(ts, now)
        except _CALENDAR_FALLBACK_EXCEPTIONS as exc:
            # Degrade to wall-clock age (today's rule); never fail health.
            logger.warning(
                "REPLAY_FRESHNESS_CALENDAR_UNAVAILABLE",
                extra={
                    "calendar": REPLAY_FRESHNESS_CALENDAR,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            session_age = {
                "basis": "wall_clock",
                "reason": "replay_freshness_calendar_unavailable",
                "session_age_hours": float(age_hours),
                "non_trading_hours_excluded": 0.0,
                "sessions_since_replay": None,
                "covered_session": None,
            }
    wall_clock_fresh = bool(
        age_hours is not None and age_hours <= float(max_age_hours)
    )
    # Exchange-calendar freshness: weekend/holiday hours do not age the
    # artifact, but only while it has missed at most one completed session.
    # On spans of trading days only, session age equals wall-clock age, so the
    # weekday threshold is unchanged and this never makes freshness stricter.
    sessions_since_replay = session_age.get("sessions_since_replay")
    session_age_hours = session_age.get("session_age_hours")
    calendar_fresh = bool(
        session_age.get("basis") == "exchange_calendar"
        and session_age_hours is not None
        and float(session_age_hours) <= float(max_age_hours)
        and sessions_since_replay is not None
        and int(sessions_since_replay) <= 1
    )
    fresh = bool(
        age_hours is not None
        and (wall_clock_fresh or calendar_fresh)
        and not future_dated
    )
    violations_raw = payload.get("violations")
    violations_count = (
        len(violations_raw)
        if isinstance(violations_raw, list)
        else _as_int(payload.get("violations_count"), 0)
    )
    cap_adjustments_raw = payload.get("cap_adjustments")
    cap_adjustments_count = (
        len(cap_adjustments_raw)
        if isinstance(cap_adjustments_raw, list)
        else _as_int(payload.get("cap_adjustments_count"), 0)
    )
    violations_by_code = dict(payload.get("violations_by_code", {}) or {})
    counterfactual_raw = payload.get("counterfactual")
    counterfactual = (
        dict(counterfactual_raw)
        if isinstance(counterfactual_raw, Mapping)
        else {}
    )
    counterfactual_candidate_raw = counterfactual.get("candidate")
    counterfactual_candidate = (
        dict(counterfactual_candidate_raw)
        if isinstance(counterfactual_candidate_raw, Mapping)
        else {}
    )
    counterfactual_required_raw = counterfactual.get("required")
    counterfactual_required = (
        dict(counterfactual_required_raw)
        if isinstance(counterfactual_required_raw, Mapping)
        else {}
    )
    counterfactual_passed = counterfactual.get("passed") is True
    schema_version = str(payload.get("schema_version") or "").strip()
    policy_hash = str(payload.get("policy_hash") or "").strip()

    source_data_raw = payload.get("source_data")
    source_data = (
        dict(source_data_raw)
        if isinstance(source_data_raw, Mapping)
        else {}
    )
    return {
        "enabled": True,
        "available": True,
        "fresh": bool(fresh),
        "path": str(candidate),
        "ts": ts.isoformat() if ts is not None else None,
        "ts_source": "payload" if ts is not None else None,
        "age_hours": float(age_hours) if age_hours is not None else None,
        "age_basis": session_age.get("basis"),
        "age_basis_reason": session_age.get("reason"),
        "freshness_calendar": REPLAY_FRESHNESS_CALENDAR,
        "session_age_hours": session_age_hours,
        "non_trading_hours_excluded": session_age.get("non_trading_hours_excluded"),
        "sessions_since_replay": sessions_since_replay,
        "covered_session": session_age.get("covered_session"),
        "wall_clock_fresh": bool(wall_clock_fresh),
        "calendar_fresh": bool(calendar_fresh),
        "future_dated": bool(future_dated),
        "future_skew_seconds": float(future_skew_seconds),
        "max_future_skew_seconds": float(max_future_skew_seconds),
        "reason": (
            "replay_governance_artifact_missing_payload_ts"
            if ts is None
            else ("replay_governance_artifact_future_dated" if future_dated else None)
        ),
        "max_age_hours": float(max_age_hours),
        "rows": _as_int(payload.get("rows"), 0),
        "orders_submitted": _as_int(payload.get("orders_submitted"), 0),
        "fill_events": _as_int(payload.get("fill_events"), 0),
        "violations_count": int(max(0, violations_count)),
        "violations_by_code": violations_by_code,
        "cap_adjustments_count": int(max(0, cap_adjustments_count)),
        "counterfactual_passed": bool(counterfactual_passed),
        "schema_version": schema_version,
        "policy_hash": policy_hash,
        "counterfactual_available": "passed" in counterfactual,
        "counterfactual": counterfactual,
        "counterfactual_candidate_samples": _as_int(
            counterfactual_candidate.get("sample_count"),
            0,
        ),
        "counterfactual_required_samples": _as_int(
            counterfactual_required.get("min_samples"),
            0,
        ),
        "counterfactual_candidate_net_edge_bps": _as_float(
            counterfactual_candidate.get("net_edge_bps"),
            0.0,
        ),
        "counterfactual_required_net_edge_bps": _as_float(
            counterfactual_required.get("min_net_edge_bps"),
            0.0,
        ),
        "source_data_fresh": source_data.get("fresh") is True,
        "source_data_available": bool(source_data),
        "source_data": source_data,
    }


def summarize_replay_live_parity_gate(
    *,
    replay_governance: Mapping[str, Any] | None = None,
    oms_lifecycle_parity: Mapping[str, Any] | None = None,
    expected_policy_hash: str | None = None,
) -> dict[str, Any]:
    enabled = bool(
        get_env("AI_TRADING_REPLAY_LIVE_PARITY_GATE_ENABLED", True, cast=bool)
    )
    if not enabled:
        return {"enabled": False, "available": False, "ok": True, "status": "disabled"}

    replay_snapshot = (
        dict(replay_governance)
        if isinstance(replay_governance, Mapping)
        else _load_latest_replay_governance_snapshot()
    )
    lifecycle_snapshot = (
        dict(oms_lifecycle_parity)
        if isinstance(oms_lifecycle_parity, Mapping)
        else {}
    )
    require_fresh_replay = bool(
        get_env(
            "AI_TRADING_REPLAY_LIVE_PARITY_REQUIRE_FRESH_REPLAY",
            True,
            cast=bool,
        )
    )
    require_fresh_source = bool(
        get_env(
            "AI_TRADING_REPLAY_LIVE_PARITY_REQUIRE_FRESH_SOURCE",
            True,
            cast=bool,
        )
    )
    require_counterfactual_passed = bool(
        get_env(
            "AI_TRADING_REPLAY_LIVE_PARITY_REQUIRE_COUNTERFACTUAL_PASSED",
            True,
            cast=bool,
        )
    )
    max_replay_violations = max(
        0,
        _as_int(
            get_env(
                "AI_TRADING_REPLAY_LIVE_PARITY_MAX_REPLAY_VIOLATIONS",
                0,
                cast=int,
            ),
            0,
        ),
    )
    require_oms_lifecycle_parity = bool(
        get_env(
            "AI_TRADING_REPLAY_LIVE_PARITY_REQUIRE_OMS_LIFECYCLE_PARITY",
            _default_fail_closed_outside_tests(),
            cast=bool,
        )
    )
    max_oms_lifecycle_parity_violations = max(
        0,
        _as_int(
            get_env(
                "AI_TRADING_REPLAY_LIVE_PARITY_MAX_OMS_LIFECYCLE_PARITY_VIOLATIONS",
                0,
                cast=int,
            ),
            0,
        ),
    )

    replay_available = bool(replay_snapshot.get("available"))
    replay_fresh = bool(replay_snapshot.get("fresh"))
    replay_future_dated = bool(replay_snapshot.get("future_dated", False))
    source_data_raw = replay_snapshot.get("source_data")
    source_data = (
        dict(source_data_raw)
        if isinstance(source_data_raw, Mapping)
        else {}
    )
    source_data_available = bool(
        replay_snapshot.get("source_data_available", bool(source_data))
    )
    source_data_fresh = (
        replay_snapshot.get("source_data_fresh", source_data.get("fresh")) is True
    )
    replay_violations_count = max(
        0,
        _as_int(replay_snapshot.get("violations_count"), 0),
    )
    counterfactual_raw = replay_snapshot.get("counterfactual_passed")
    replay_counterfactual_available = replay_snapshot.get("counterfactual_available")
    if counterfactual_raw is None and isinstance(replay_snapshot.get("counterfactual"), Mapping):
        counterfactual_payload = dict(replay_snapshot.get("counterfactual", {}))
        counterfactual_raw = counterfactual_payload.get("passed")
        replay_counterfactual_available = "passed" in counterfactual_payload
    else:
        counterfactual_payload = (
            dict(replay_snapshot.get("counterfactual", {}))
            if isinstance(replay_snapshot.get("counterfactual"), Mapping)
            else {}
        )
    candidate_payload = (
        dict(counterfactual_payload.get("candidate", {}))
        if isinstance(counterfactual_payload.get("candidate"), Mapping)
        else {}
    )
    required_payload = (
        dict(counterfactual_payload.get("required", {}))
        if isinstance(counterfactual_payload.get("required"), Mapping)
        else {}
    )
    replay_counterfactual_passed = counterfactual_raw is True
    replay_counterfactual_available = bool(replay_counterfactual_available)

    replay_schema_version = str(replay_snapshot.get("schema_version") or "").strip()
    replay_policy_hash = str(replay_snapshot.get("policy_hash") or "").strip()
    resolved_expected_policy_hash = str(
        expected_policy_hash
        or get_env("AI_TRADING_EFFECTIVE_POLICY_HASH", "", cast=str)
        or ""
    ).strip()
    replay_contract_ok = bool(
        replay_schema_version == REPLAY_GOVERNANCE_SCHEMA_VERSION
        and replay_policy_hash
        and (
            not resolved_expected_policy_hash
            or replay_policy_hash == resolved_expected_policy_hash
        )
    )

    lifecycle_enabled = bool(
        lifecycle_snapshot.get("enabled", bool(lifecycle_snapshot))
    )
    lifecycle_available = bool(
        lifecycle_snapshot.get("available", lifecycle_enabled)
    )
    lifecycle_total_violations = max(
        0,
        _as_int(lifecycle_snapshot.get("total_violations"), 0),
    )
    lifecycle_ok_raw = lifecycle_snapshot.get("ok")
    lifecycle_ok = (
        bool(lifecycle_ok_raw)
        if lifecycle_ok_raw is not None
        else bool(
            lifecycle_available
            and lifecycle_total_violations <= int(max_oms_lifecycle_parity_violations)
        )
    )

    replay_available_ok = bool(replay_available)
    replay_fresh_ok = bool(
        (replay_fresh or (not require_fresh_replay)) and not replay_future_dated
    )
    replay_source_fresh_ok = bool(
        (source_data_available and source_data_fresh)
        or (not require_fresh_source)
    )
    replay_violations_ok = bool(
        replay_violations_count <= int(max_replay_violations)
    )
    replay_counterfactual_ok = bool(
        replay_counterfactual_passed or (not require_counterfactual_passed)
    )
    lifecycle_available_ok = bool(
        lifecycle_available if require_oms_lifecycle_parity else True
    )
    lifecycle_consistent_ok = bool(
        (
            lifecycle_ok
            and lifecycle_total_violations <= int(max_oms_lifecycle_parity_violations)
        )
        if (require_oms_lifecycle_parity and lifecycle_available)
        else (not require_oms_lifecycle_parity)
    )

    checks = {
        "replay_available": replay_available_ok,
        "replay_contract": replay_contract_ok,
        "replay_fresh": replay_fresh_ok,
        "replay_source_fresh": replay_source_fresh_ok,
        "replay_violations": replay_violations_ok,
        "replay_counterfactual": replay_counterfactual_ok,
        "oms_lifecycle_parity_available": lifecycle_available_ok,
        "oms_lifecycle_parity_consistent": lifecycle_consistent_ok,
    }
    failed_checks = [name for name, passed in checks.items() if not bool(passed)]
    reason = "ok"
    if failed_checks:
        reason = str(failed_checks[0])

    return {
        "enabled": True,
        "available": bool(replay_available or lifecycle_available),
        "ok": not failed_checks,
        "status": "pass" if not failed_checks else "fail",
        "reason": reason,
        "failed_checks": failed_checks,
        "checks": checks,
        "thresholds": {
            "require_fresh_replay": bool(require_fresh_replay),
            "require_fresh_source": bool(require_fresh_source),
            "required_schema_version": REPLAY_GOVERNANCE_SCHEMA_VERSION,
            "expected_policy_hash": resolved_expected_policy_hash or None,
            "max_replay_age_hours": float(
                replay_snapshot.get(
                    "max_age_hours",
                    get_env(
                        "AI_TRADING_REPLAY_LIVE_PARITY_MAX_REPLAY_AGE_HOURS",
                        24.0,
                        cast=float,
                    ),
                )
            ),
            "max_replay_future_skew_seconds": float(
                replay_snapshot.get(
                    "max_future_skew_seconds",
                    get_env(
                        "AI_TRADING_REPLAY_LIVE_PARITY_MAX_FUTURE_SKEW_SECONDS",
                        300.0,
                        cast=float,
                    ),
                )
            ),
            "require_counterfactual_passed": bool(require_counterfactual_passed),
            "max_replay_violations": int(max_replay_violations),
            "require_oms_lifecycle_parity": bool(require_oms_lifecycle_parity),
            "max_oms_lifecycle_parity_violations": int(
                max_oms_lifecycle_parity_violations
            ),
        },
        "observed": {
            "replay_available": bool(replay_available),
            "replay_schema_version": replay_schema_version or None,
            "replay_policy_hash": replay_policy_hash or None,
            "replay_schema_valid": replay_schema_version == REPLAY_GOVERNANCE_SCHEMA_VERSION,
            "replay_policy_hash_matches": replay_contract_ok,
            "replay_fresh": bool(replay_fresh),
            "replay_age_hours": replay_snapshot.get("age_hours"),
            "replay_age_basis": replay_snapshot.get("age_basis"),
            "replay_session_age_hours": replay_snapshot.get("session_age_hours"),
            "replay_non_trading_hours_excluded": replay_snapshot.get(
                "non_trading_hours_excluded"
            ),
            "sessions_since_replay": replay_snapshot.get("sessions_since_replay"),
            "replay_covered_session": replay_snapshot.get("covered_session"),
            "replay_freshness_calendar": replay_snapshot.get("freshness_calendar"),
            "replay_freshness_rule": (
                "fresh when wall-clock age <= max_replay_age_hours, or when "
                "exchange-session age (weekend/holiday hours excluded) <= "
                "max_replay_age_hours and sessions_since_replay <= 1"
            ),
            "replay_future_dated": bool(replay_future_dated),
            "replay_future_skew_seconds": replay_snapshot.get("future_skew_seconds"),
            "replay_violations_count": int(replay_violations_count),
            "replay_source_data_available": bool(source_data_available),
            "replay_source_data_fresh": bool(source_data_fresh),
            "replay_source_age_hours": source_data.get("age_hours"),
            "replay_source_max_age_hours": source_data.get("max_age_hours"),
            "replay_source_max_ts": source_data.get("max_ts"),
            "replay_cap_adjustments_count": int(
                max(0, _as_int(replay_snapshot.get("cap_adjustments_count"), 0))
            ),
            "replay_counterfactual_passed": bool(replay_counterfactual_passed),
            "replay_counterfactual_available": bool(replay_counterfactual_available),
            "replay_counterfactual_waived": bool(
                not require_counterfactual_passed and not replay_counterfactual_passed
            ),
            "replay_counterfactual_candidate_samples": _as_int(
                replay_snapshot.get(
                    "counterfactual_candidate_samples",
                    candidate_payload.get("sample_count"),
                ),
                0,
            ),
            "replay_counterfactual_required_samples": _as_int(
                replay_snapshot.get(
                    "counterfactual_required_samples",
                    required_payload.get("min_samples"),
                ),
                0,
            ),
            "replay_counterfactual_candidate_net_edge_bps": _as_float(
                replay_snapshot.get(
                    "counterfactual_candidate_net_edge_bps",
                    candidate_payload.get("net_edge_bps"),
                ),
                0.0,
            ),
            "replay_counterfactual_required_net_edge_bps": _as_float(
                replay_snapshot.get(
                    "counterfactual_required_net_edge_bps",
                    required_payload.get("min_net_edge_bps"),
                ),
                0.0,
            ),
            "oms_lifecycle_parity_enabled": bool(lifecycle_enabled),
            "oms_lifecycle_parity_available": bool(lifecycle_available),
            "oms_lifecycle_parity_ok": bool(lifecycle_ok),
            "oms_lifecycle_parity_total_violations": int(
                lifecycle_total_violations
            ),
        },
        "replay_governance": replay_snapshot,
        "oms_lifecycle_parity": lifecycle_snapshot,
    }


__all__ = [
    "summarize_replay_live_parity_gate",
]
