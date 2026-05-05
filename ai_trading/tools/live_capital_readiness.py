"""Build the live-capital readiness gate artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.request import urlopen

from ai_trading.config.launch_profiles import launch_profile_payload, resolve_launch_profile
from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _health_from_endpoint(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=5.0) as response:  # nosec B310 - local operator health endpoint
            parsed = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _nested(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = get_env(name, None, cast=str, resolve_aliases=False)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str = "") -> str:
    return str(get_env(name, default, cast=str, resolve_aliases=False) or default).strip()


def _daily_loss_configured() -> bool:
    for name in (
        "AI_TRADING_LIVE_MAX_DAILY_LOSS",
        "DOLLAR_RISK_LIMIT",
        "AI_TRADING_DAILY_LOSS_LIMIT",
    ):
        raw = _env_text(name)
        if raw:
            try:
                return float(raw) > 0.0
            except ValueError:
                continue
    return False


def build_live_capital_readiness(
    *,
    health: Mapping[str, Any],
    live_cost_model: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
    validation: Mapping[str, Any] | None = None,
    canary_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validation = validation or {}
    canary_plan = canary_plan or {}
    profile = resolve_launch_profile()
    profile_payload = launch_profile_payload(profile)
    reasons: list[str] = []
    actions: list[str] = []

    if not bool(validation.get("full_validation_green", False)):
        reasons.append("full_validation_green_artifact_missing")
        actions.append("run full validation and save a green validation artifact")
    if not bool(health.get("ok")) or str(health.get("status") or "").lower() not in {"healthy", "ready"}:
        reasons.append("health_not_healthy")
    broker = _nested(health, "broker")
    if not bool(broker.get("connected")):
        reasons.append("broker_not_connected")
    if not bool(_nested(health, "database").get("ok")):
        reasons.append("database_not_ok")
    for gate_name in ("oms_invariants", "oms_lifecycle_parity"):
        gate = _nested(health, gate_name)
        if gate and not bool(gate.get("ok", True)):
            reasons.append(f"{gate_name}_not_ok")
    replay_gate = _nested(health, "replay_live_parity_gate")
    if replay_gate and not bool(replay_gate.get("ok", True)):
        reasons.append("replay_governance_not_ok")
    if profile.promotion_required and not bool(promotion_report.get("promotion_ready")):
        reasons.append("promotion_report_not_ready")
    live_status = _nested(live_cost_model, "status")
    if not bool(live_status.get("available", bool(live_cost_model))):
        reasons.append("live_cost_model_unavailable")
    if int(live_status.get("breach_count") or 0) > 0:
        reasons.append("live_cost_breaches_present")
    provider = _nested(health, "data_provider")
    if str(provider.get("status") or "").lower() in {"degraded", "unhealthy", "down", "error"}:
        reasons.append("provider_degraded")
    if not _daily_loss_configured() and profile.max_daily_loss is None:
        reasons.append("daily_max_loss_not_configured")
    if profile.name not in {"live_canary", "live_restricted", "live_normal"}:
        reasons.append("live_capital_profile_not_selected")
    if not _env_bool("AI_TRADING_LIVE_ACCOUNT_CONFIRMED", False):
        reasons.append("live_account_not_explicitly_confirmed")
        actions.append("set AI_TRADING_LIVE_ACCOUNT_CONFIRMED=1 only after account review")
    if not canary_plan and profile.name == "live_canary":
        reasons.append("paper_vs_live_canary_plan_missing")
    if profile.name == "live_canary" and (
        profile.max_notional_per_order is None or float(profile.max_notional_per_order) > 250.0
    ):
        reasons.append("live_canary_notional_cap_not_tiny")
    if profile.name == "live_canary" and profile.max_order_count > 3:
        reasons.append("live_canary_order_count_too_high")

    if reasons:
        status = "paper_only" if reasons == ["live_capital_profile_not_selected"] else "blocked"
    elif profile.name == "live_canary":
        status = "live_canary_allowed"
    else:
        status = "live_allowed"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "live_capital_readiness",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "reasons": reasons,
        "required_operator_actions": actions,
        "launch_profile": profile_payload,
        "gates": {
            "full_validation_green": bool(validation.get("full_validation_green", False)),
            "health_ok": bool(health.get("ok")),
            "broker_connected": bool(broker.get("connected")),
            "database_ok": bool(_nested(health, "database").get("ok")),
            "promotion_ready": bool(promotion_report.get("promotion_ready")),
            "live_cost_ready": bool(live_status.get("available", bool(live_cost_model))),
            "daily_loss_configured": _daily_loss_configured() or profile.max_daily_loss is not None,
            "live_account_confirmed": _env_bool("AI_TRADING_LIVE_ACCOUNT_CONFIRMED", False),
        },
    }


def _default_output() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/live_capital_readiness_latest.json",
        default_relative="runtime/live_capital_readiness_latest.json",
        for_write=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-json", type=Path, default=None)
    parser.add_argument("--health-url", default="http://127.0.0.1:9001/healthz")
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--promotion-report-json", type=Path, default=None)
    parser.add_argument("--validation-json", type=Path, default=None)
    parser.add_argument("--canary-plan-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--success-on-blocked", action="store_true")
    args = parser.parse_args(argv)
    health = _read_json(args.health_json) if args.health_json else _health_from_endpoint(str(args.health_url))
    report = build_live_capital_readiness(
        health=health,
        live_cost_model=_read_json(
            args.live_cost_model_json
            or resolve_runtime_artifact_path(
                "runtime/live_cost_model_latest.json",
                default_relative="runtime/live_cost_model_latest.json",
            )
        ),
        promotion_report=_read_json(args.promotion_report_json),
        validation=_read_json(args.validation_json),
        canary_plan=_read_json(args.canary_plan_json),
    )
    output = args.output_json or _default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output), "status": report["status"]}) + "\n")
    if args.success_on_blocked:
        return 0
    return 0 if report["status"] in {"live_canary_allowed", "live_allowed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
