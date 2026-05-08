"""Build the live-capital readiness gate artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
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


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _freshness(payload: Mapping[str, Any], *, max_age_hours: float, now: datetime | None = None) -> dict[str, Any]:
    generated = _parse_utc_timestamp(
        payload.get("generated_at")
        or payload.get("created_at")
        or payload.get("timestamp")
        or payload.get("as_of")
    )
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    max_age = timedelta(hours=max(0.0, float(max_age_hours)))
    fresh = bool(generated is not None and generated <= now_utc and now_utc - generated <= max_age)
    age_hours = None
    if generated is not None:
        age_hours = max(0.0, (now_utc - generated).total_seconds() / 3600.0)
    return {
        "fresh": fresh,
        "generated_at": generated.isoformat().replace("+00:00", "Z") if generated else None,
        "age_hours": age_hours,
        "max_age_hours": float(max_age_hours),
    }


def _freshness_limit(name: str, default: float) -> float:
    raw = _env_text(name, str(default))
    try:
        parsed = float(raw)
    except ValueError:
        return float(default)
    return max(0.0, parsed)


def _summary_status(payload: Mapping[str, Any], default: str = "missing") -> str:
    raw = payload.get("status")
    if isinstance(raw, Mapping):
        return str(raw.get("status") or default)
    return str(raw or default)


def build_live_capital_readiness(
    *,
    health: Mapping[str, Any],
    live_cost_model: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
    validation: Mapping[str, Any] | None = None,
    canary_plan: Mapping[str, Any] | None = None,
    edge_calibration: Mapping[str, Any] | None = None,
    execution_capture: Mapping[str, Any] | None = None,
    portfolio_edge: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validation = validation or {}
    canary_plan = canary_plan or {}
    edge_calibration = edge_calibration or {}
    execution_capture = execution_capture or {}
    portfolio_edge = portfolio_edge or {}
    profile = resolve_launch_profile()
    profile_payload = launch_profile_payload(profile)
    reasons: list[str] = []
    actions: list[str] = []
    freshness = {
        "validation": _freshness(
            validation,
            max_age_hours=_freshness_limit("AI_TRADING_LIVE_VALIDATION_MAX_AGE_HOURS", 72.0),
        ),
        "live_cost_model": _freshness(
            live_cost_model,
            max_age_hours=_freshness_limit("AI_TRADING_LIVE_COST_MAX_AGE_HOURS", 24.0),
        ),
        "promotion_report": _freshness(
            promotion_report,
            max_age_hours=_freshness_limit("AI_TRADING_LIVE_PROMOTION_MAX_AGE_HOURS", 72.0),
        ),
        "canary_plan": _freshness(
            canary_plan,
            max_age_hours=_freshness_limit("AI_TRADING_LIVE_CANARY_PLAN_MAX_AGE_HOURS", 72.0),
        ),
        "edge_calibration": _freshness(
            edge_calibration,
            max_age_hours=_freshness_limit("AI_TRADING_LIVE_EDGE_CALIBRATION_MAX_AGE_HOURS", 24.0),
        ),
        "execution_capture": _freshness(
            execution_capture,
            max_age_hours=_freshness_limit("AI_TRADING_LIVE_EXECUTION_CAPTURE_MAX_AGE_HOURS", 24.0),
        ),
        "portfolio_edge": _freshness(
            portfolio_edge,
            max_age_hours=_freshness_limit("AI_TRADING_LIVE_PORTFOLIO_EDGE_MAX_AGE_HOURS", 24.0),
        ),
    }

    if not bool(validation.get("full_validation_green", False)):
        reasons.append("full_validation_green_artifact_missing")
        actions.append("run full validation and save a green validation artifact")
    elif not freshness["validation"]["fresh"]:
        reasons.append("full_validation_green_artifact_stale")
        actions.append("rerun full validation and save a fresh green artifact")
    if not bool(health.get("ok")) or str(health.get("status") or "").lower() not in {"healthy", "ready"}:
        reasons.append("health_not_healthy")
    broker = _nested(health, "broker")
    if not bool(broker.get("connected")):
        reasons.append("broker_not_connected")
    if not bool(_nested(health, "database").get("ok")):
        reasons.append("database_not_ok")
    if "market_closed_non_flat_positions" in {
        str(flag) for flag in health.get("attention_flags", []) if flag is not None
    }:
        reasons.append("surprise_non_flat_positions")
    for gate_name in ("oms_invariants", "oms_lifecycle_parity"):
        gate = _nested(health, gate_name)
        if gate and not bool(gate.get("ok", True)):
            reasons.append(f"{gate_name}_not_ok")
    provider_authority = _nested(health, "provider_authority")
    if provider_authority and provider_authority.get("ok") is False:
        reasons.append("provider_authority_not_ok")
    replay_gate = _nested(health, "replay_live_parity_gate")
    if replay_gate and not bool(replay_gate.get("ok", True)):
        reasons.append("replay_governance_not_ok")
    if profile.promotion_required and not bool(promotion_report.get("promotion_ready")):
        reasons.append("promotion_report_not_ready")
    if profile.promotion_required and promotion_report and not freshness["promotion_report"]["fresh"]:
        reasons.append("promotion_report_stale")
    live_status = _nested(live_cost_model, "status")
    if not bool(live_status.get("available", bool(live_cost_model))):
        reasons.append("live_cost_model_unavailable")
    elif not freshness["live_cost_model"]["fresh"]:
        reasons.append("live_cost_model_stale")
    elif str(live_status.get("status") or "").lower() not in {"ready", "ok"}:
        reasons.append("live_cost_model_not_ready")
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
    elif profile.name == "live_canary" and not freshness["canary_plan"]["fresh"]:
        reasons.append("paper_vs_live_canary_plan_stale")
    if profile.name == "live_canary" and canary_plan.get("trade_allowed") is False:
        reasons.append("daily_research_trade_not_allowed")
    runtime_gonogo = _nested(canary_plan, "runtime_gonogo")
    if profile.name == "live_canary" and runtime_gonogo.get("gate_passed") is False:
        reasons.append("runtime_gonogo_failed")
    if profile.name == "live_canary" and (
        profile.max_notional_per_order is None or float(profile.max_notional_per_order) > 250.0
    ):
        reasons.append("live_canary_notional_cap_not_tiny")
    if profile.name == "live_canary" and profile.max_order_count > 3:
        reasons.append("live_canary_order_count_too_high")
    if profile.name.startswith("live_"):
        edge_status = str(edge_calibration.get("status") or "").lower()
        if not edge_calibration:
            reasons.append("edge_calibration_missing")
        elif not freshness["edge_calibration"]["fresh"]:
            reasons.append("edge_calibration_stale")
        elif edge_status in {"inverted", "overestimated"}:
            reasons.append(f"edge_calibration_{edge_status}")
        capture_status = str(execution_capture.get("status") or "").lower()
        if not execution_capture:
            reasons.append("execution_capture_missing")
        elif not freshness["execution_capture"]["fresh"]:
            reasons.append("execution_capture_stale")
        elif capture_status in {"needs_review", "degraded"}:
            reasons.append("execution_capture_not_acceptable")
        portfolio_status = str(portfolio_edge.get("status") or portfolio_edge.get("output") or "").lower()
        if not portfolio_edge:
            reasons.append("portfolio_edge_missing")
        elif not freshness["portfolio_edge"]["fresh"]:
            reasons.append("portfolio_edge_stale")
        elif portfolio_status in {"control_breach", "no_new_entries"}:
            reasons.append("portfolio_edge_control_breach")

    if reasons:
        status = "paper_only" if reasons == ["live_capital_profile_not_selected"] else "blocked"
    elif profile.name == "live_canary":
        status = "live_canary_allowed"
    else:
        status = "live_allowed"
    report = {
        "schema_version": "1.0.0",
        "artifact_type": "live_capital_readiness",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "reasons": reasons,
        "required_operator_actions": actions,
        "launch_profile": profile_payload,
        "freshness": freshness,
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
    report["canary_evidence"] = {
        "daily_research_trade_allowed": canary_plan.get("trade_allowed"),
        "daily_research_mode": canary_plan.get("recommended_next_session_mode"),
        "daily_research_blocked_reasons": list(canary_plan.get("blocked_reasons", []))
        if isinstance(canary_plan.get("blocked_reasons"), list)
        else [],
        "runtime_gonogo": _nested(canary_plan, "runtime_gonogo"),
        "health_report_summary": _nested(canary_plan, "health_report_summary"),
        "next_session_limits": _nested(canary_plan, "next_session_limits"),
        "edge_calibration": {
            "status": edge_calibration.get("status"),
            "fresh": freshness["edge_calibration"]["fresh"],
            "summary": _nested(edge_calibration, "summary"),
        },
        "execution_capture": {
            "status": execution_capture.get("status"),
            "fresh": freshness["execution_capture"]["fresh"],
            "summary": _nested(execution_capture, "summary"),
        },
        "portfolio_edge": {
            "status": portfolio_edge.get("status") or portfolio_edge.get("output"),
            "fresh": freshness["portfolio_edge"]["fresh"],
            "summary": _nested(portfolio_edge, "summary"),
        },
    }
    report["health_report_summary"] = {
        "status": status,
        "health_status": _summary_status(health),
        "health_ok": bool(health.get("ok")),
        "broker_connected": bool(broker.get("connected")),
        "database_ok": bool(_nested(health, "database").get("ok")),
        "provider_status": _summary_status(_nested(health, "data_provider")),
        "provider_authority_ok": _nested(health, "provider_authority").get("ok"),
        "promotion_ready": bool(promotion_report.get("promotion_ready")),
        "live_cost_status": _summary_status(live_status),
        "validation_fresh": bool(freshness["validation"]["fresh"]),
        "live_cost_fresh": bool(freshness["live_cost_model"]["fresh"]),
        "canary_plan_fresh": bool(freshness["canary_plan"]["fresh"]),
        "reasons": reasons,
    }
    report["openclaw_summary"] = {
        "service": "ai-trading-live-capital",
        "severity": "info" if status in {"live_canary_allowed", "live_allowed"} else "warning",
        "summary": f"live_capital_readiness status={status} profile={profile.name}",
        "suggested_action": (
            "manual review required before enabling live capital"
            if status in {"live_canary_allowed", "live_allowed"}
            else "resolve live-capital readiness blockers before live cutover"
        ),
        "blocked_reasons": reasons,
        "details": {
            "health_report_summary": report["health_report_summary"],
            "canary_evidence": report["canary_evidence"],
        },
    }
    return report


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
    parser.add_argument("--edge-calibration-json", type=Path, default=None)
    parser.add_argument("--execution-capture-json", type=Path, default=None)
    parser.add_argument("--portfolio-edge-json", type=Path, default=None)
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
        edge_calibration=_read_json(args.edge_calibration_json),
        execution_capture=_read_json(args.execution_capture_json),
        portfolio_edge=_read_json(args.portfolio_edge_json),
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
