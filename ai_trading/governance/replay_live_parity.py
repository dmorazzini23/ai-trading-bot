from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


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
                    96.0,
                    cast=float,
                ),
                96.0,
            ),
            24.0 * 14.0,
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
    for path in candidates:
        try:
            mtime_ts = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        except OSError:
            mtime_ts = datetime.fromtimestamp(0.0, UTC)
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            ranked_candidates.append(
                (mtime_ts, str(path), path, None, "replay_governance_artifact_invalid")
            )
            continue
        if not isinstance(raw_payload, Mapping):
            ranked_candidates.append(
                (mtime_ts, str(path), path, None, "replay_governance_artifact_invalid")
            )
            continue
        payload = dict(raw_payload)
        payload_ts = _parse_ts(payload.get("ts"))
        ranked_candidates.append(
            (
                payload_ts if payload_ts is not None else mtime_ts,
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
    if ts is None:
        try:
            ts = datetime.fromtimestamp(candidate.stat().st_mtime, UTC)
        except OSError:
            ts = None
    age_hours: float | None = None
    if ts is not None:
        age_hours = max((datetime.now(UTC) - ts).total_seconds(), 0.0) / 3600.0
    fresh = bool(age_hours is not None and age_hours <= float(max_age_hours))
    violations_raw = payload.get("violations")
    violations_count = (
        len(violations_raw)
        if isinstance(violations_raw, list)
        else _as_int(payload.get("violations_count"), 0)
    )
    counterfactual_raw = payload.get("counterfactual")
    counterfactual = (
        dict(counterfactual_raw)
        if isinstance(counterfactual_raw, Mapping)
        else {}
    )
    counterfactual_passed = bool(counterfactual.get("passed", True))

    return {
        "enabled": True,
        "available": True,
        "fresh": bool(fresh),
        "path": str(candidate),
        "ts": ts.isoformat() if ts is not None else None,
        "ts_source": "payload" if _parse_ts(payload.get("ts")) is not None else "mtime",
        "age_hours": float(age_hours) if age_hours is not None else None,
        "max_age_hours": float(max_age_hours),
        "rows": _as_int(payload.get("rows"), 0),
        "orders_submitted": _as_int(payload.get("orders_submitted"), 0),
        "fill_events": _as_int(payload.get("fill_events"), 0),
        "violations_count": int(max(0, violations_count)),
        "violations_by_code": dict(payload.get("violations_by_code", {}) or {}),
        "counterfactual_passed": bool(counterfactual_passed),
        "counterfactual": counterfactual,
    }


def summarize_replay_live_parity_gate(
    *,
    replay_governance: Mapping[str, Any] | None = None,
    oms_lifecycle_parity: Mapping[str, Any] | None = None,
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
    replay_violations_count = max(
        0,
        _as_int(replay_snapshot.get("violations_count"), 0),
    )
    replay_counterfactual_passed = bool(
        replay_snapshot.get("counterfactual_passed", True)
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
    replay_fresh_ok = bool(replay_fresh or (not require_fresh_replay))
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
        "replay_fresh": replay_fresh_ok,
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
            "max_replay_age_hours": float(
                replay_snapshot.get(
                    "max_age_hours",
                    get_env(
                        "AI_TRADING_REPLAY_LIVE_PARITY_MAX_REPLAY_AGE_HOURS",
                        96.0,
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
            "replay_fresh": bool(replay_fresh),
            "replay_age_hours": replay_snapshot.get("age_hours"),
            "replay_violations_count": int(replay_violations_count),
            "replay_counterfactual_passed": bool(replay_counterfactual_passed),
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
