"""Build a unified runtime decay-control artifact from live health signals."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


_ACTION_RANK = {
    "normal": 0,
    "reduce_size": 1,
    "shadow_only": 2,
    "disable_new_entries": 3,
    "safe_mode": 4,
}


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _read_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _max_action(current: str, candidate: str) -> str:
    return (
        candidate
        if _ACTION_RANK.get(candidate, 0) > _ACTION_RANK.get(current, 0)
        else current
    )


def _status_map(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    status = payload.get("status")
    return status if isinstance(status, Mapping) else {}


def _observed_map(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    observed = payload.get("observed")
    return observed if isinstance(observed, Mapping) else {}


def _symbol_summary(payload: Mapping[str, Any]) -> dict[str, int]:
    rows = payload.get("symbols")
    if not isinstance(rows, list):
        return {"disabled": 0, "shadow_only": 0}
    disabled = 0
    shadow_only = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        mode = str(row.get("effective_mode") or "").strip().lower()
        if mode == "disabled":
            disabled += 1
        elif mode == "shadow_only":
            shadow_only += 1
    return {"disabled": disabled, "shadow_only": shadow_only}


def build_runtime_decay_controls(
    *,
    runtime_gonogo: Mapping[str, Any] | None = None,
    live_cost_model: Mapping[str, Any] | None = None,
    execution_quality_governor: Mapping[str, Any] | None = None,
    symbol_universe_scorecard: Mapping[str, Any] | None = None,
    provider_state: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Return a single reversible throttle recommendation artifact."""

    now = generated_at.astimezone(UTC) if generated_at is not None else datetime.now(UTC)
    runtime_gonogo = runtime_gonogo or {}
    live_cost_model = live_cost_model or {}
    execution_quality_governor = execution_quality_governor or {}
    symbol_universe_scorecard = symbol_universe_scorecard or {}
    provider_state = provider_state or {}

    action = "normal"
    size_scale = 1.0
    reasons: list[str] = []
    observed: dict[str, Any] = {}

    provider_status = str(provider_state.get("status") or "").strip().lower()
    provider_safe_mode = bool(provider_state.get("safe_mode"))
    observed["provider_status"] = provider_status or None
    observed["provider_safe_mode"] = provider_safe_mode
    if provider_safe_mode:
        action = _max_action(action, "safe_mode")
        size_scale = 0.0
        reasons.append("provider_safe_mode")
    elif provider_status in {"degraded", "unhealthy", "down", "error"}:
        action = _max_action(action, "disable_new_entries")
        size_scale = min(size_scale, 0.0)
        reasons.append(f"provider_{provider_status}")

    if runtime_gonogo:
        gate_passed = bool(runtime_gonogo.get("gate_passed"))
        failed_checks = runtime_gonogo.get("failed_checks")
        observed["runtime_gonogo_passed"] = gate_passed
        observed["runtime_gonogo_failed_checks"] = (
            list(failed_checks) if isinstance(failed_checks, list) else []
        )
        if not gate_passed:
            action = _max_action(action, "reduce_size")
            size_scale = min(size_scale, 0.5)
            reasons.append("runtime_gonogo_failed")

    eq_status = _status_map(execution_quality_governor)
    eq_actions = execution_quality_governor.get("actions")
    eq_actions = eq_actions if isinstance(eq_actions, Mapping) else {}
    pause_active = bool(eq_status.get("pause_active") or eq_actions.get("pause_active"))
    observed["execution_quality_pause_active"] = pause_active
    if pause_active:
        action = _max_action(action, "disable_new_entries")
        size_scale = 0.0
        reasons.append("execution_quality_pause")

    live_status = _status_map(live_cost_model)
    live_alerts = live_cost_model.get("alerts")
    live_alerts = live_alerts if isinstance(live_alerts, Mapping) else {}
    breach_count = int(_as_float(live_status.get("breach_count")) or 0.0)
    if breach_count <= 0:
        breaches = live_alerts.get("cost_threshold_breaches")
        breach_count = len(breaches) if isinstance(breaches, list) else 0
    observed["live_cost_breach_count"] = breach_count
    if breach_count > 0:
        action = _max_action(action, "reduce_size")
        size_scale = min(size_scale, 0.5)
        reasons.append("live_cost_breach")

    symbol_summary = _symbol_summary(symbol_universe_scorecard)
    observed["disabled_symbols"] = symbol_summary["disabled"]
    observed["shadow_only_symbols"] = symbol_summary["shadow_only"]
    if symbol_summary["disabled"] > 0:
        action = _max_action(action, "reduce_size")
        size_scale = min(size_scale, 0.75)
        reasons.append("symbol_universe_disabled")
    elif symbol_summary["shadow_only"] > 0:
        action = _max_action(action, "reduce_size")
        size_scale = min(size_scale, 0.85)
        reasons.append("symbol_universe_shadow_only")

    entries_allowed = action not in {"disable_new_entries", "safe_mode"}
    status = "ready" if reasons else "normal"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "runtime_decay_controls",
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "status": {
            "available": True,
            "status": status,
            "mode": "enforce" if not entries_allowed else "observe",
            "reason": ",".join(reasons) if reasons else "ok",
        },
        "observed": observed,
        "actions": {
            "max_action": action,
            "entries_allowed": bool(entries_allowed),
            "size_scale": float(max(0.0, min(size_scale, 1.0))),
            "reasons": reasons,
        },
        "recovery": {
            "criteria": [
                "provider healthy",
                "runtime go/no-go passes",
                "execution quality pause cleared",
                "live cost breaches cleared",
                "symbol universe scorecard no longer deteriorating",
            ],
            "reversible": True,
        },
    }


def _default_path(env_key: str, default_relative: str) -> Path:
    configured = str(get_env(env_key, default_relative, cast=str, resolve_aliases=False) or default_relative)
    return resolve_runtime_artifact_path(configured, default_relative=default_relative)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-gonogo-json", type=Path, default=None)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--execution-quality-governor-json", type=Path, default=None)
    parser.add_argument("--symbol-universe-scorecard-json", type=Path, default=None)
    parser.add_argument("--provider-state-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    output_path = args.output_json or _default_path(
        "AI_TRADING_RUNTIME_DECAY_CONTROLS_PATH",
        "runtime/runtime_decay_controls_latest.json",
    )
    artifact = build_runtime_decay_controls(
        runtime_gonogo=_read_json_mapping(args.runtime_gonogo_json),
        live_cost_model=_read_json_mapping(
            args.live_cost_model_json
            or _default_path("AI_TRADING_LIVE_COST_MODEL_PATH", "runtime/live_cost_model_latest.json")
        ),
        execution_quality_governor=_read_json_mapping(
            args.execution_quality_governor_json
            or _default_path(
                "AI_TRADING_EXECUTION_QUALITY_GOVERNOR_REPORT_PATH",
                "runtime/execution_quality_governor_latest.json",
            )
        ),
        symbol_universe_scorecard=_read_json_mapping(
            args.symbol_universe_scorecard_json
            or _default_path(
                "AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_PATH",
                "runtime/symbol_universe_scorecard_latest.json",
            )
        ),
        provider_state=_read_json_mapping(args.provider_state_json),
    )
    artifact["paths"] = {"report": str(output_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_path), "status": artifact["status"]}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
