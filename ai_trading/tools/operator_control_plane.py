"""Build a read-only operator control-plane aggregation artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.request import urlopen

from ai_trading.config.launch_profiles import launch_profile_payload, resolve_launch_profile
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _health_from_endpoint(url: str | None) -> dict[str, Any]:
    target = str(url or "").strip()
    if not target:
        return {}
    try:
        with urlopen(target, timeout=3.0) as response:  # nosec B310 - operator read-only health endpoint
            parsed = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _artifact_status(payload: Mapping[str, Any]) -> str:
    status = payload.get("status")
    if isinstance(status, Mapping):
        return str(status.get("status") or status.get("state") or "present")
    if status not in (None, ""):
        return str(status)
    if "ok" in payload:
        return "ok" if bool(payload.get("ok")) else "not_ok"
    if payload:
        return "present"
    return "missing"


def _go_no_go_payload(readiness: Mapping[str, Any], runtime_gonogo: Mapping[str, Any]) -> dict[str, Any]:
    readiness_summary = _mapping(readiness.get("health_report_summary"))
    runtime_failed = runtime_gonogo.get("failed_checks")
    if not isinstance(runtime_failed, list):
        runtime_failed = []
    return {
        "readiness_status": readiness.get("status"),
        "readiness_reasons": list(readiness.get("reasons", []))
        if isinstance(readiness.get("reasons"), list)
        else [],
        "runtime_gate_passed": runtime_gonogo.get("gate_passed"),
        "runtime_failed_checks": runtime_failed,
        "runtime_status": _artifact_status(runtime_gonogo),
        "health_report_summary": readiness_summary,
    }


def _orders_positions_oms(
    health: Mapping[str, Any],
    runtime_performance: Mapping[str, Any],
    oms: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _mapping(_mapping(runtime_performance.get("go_no_go")).get("observed"))
    return {
        "broker": {
            "open_orders_count": _mapping(health.get("broker")).get("open_orders_count"),
            "positions_count": _mapping(health.get("broker")).get("positions_count"),
            "connected": _mapping(health.get("broker")).get("connected"),
        },
        "positions": {
            "reconciliation_available": observed.get("open_position_reconciliation_available"),
            "reconciliation_consistent": observed.get("open_position_reconciliation_consistent"),
            "mismatch_count": observed.get("open_position_reconciliation_mismatch_count"),
            "source": runtime_performance.get("source"),
        },
        "orders": {
            "available": runtime_performance.get("available"),
            "source": runtime_performance.get("source"),
        },
        "oms": {
            "artifact_status": _artifact_status(oms),
            "invariants": _mapping(health.get("oms_invariants")),
            "lifecycle_parity": _mapping(health.get("oms_lifecycle_parity")),
            "replay_live_parity_gate": _mapping(health.get("replay_live_parity_gate")),
            "details": dict(oms),
        },
    }


def _operator_actions(payload: Mapping[str, Any]) -> dict[str, Any]:
    actions = payload.get("actions")
    action_list = actions if isinstance(actions, list) else []
    approvals = payload.get("recent_promotion_approvals")
    approval_list = approvals if isinstance(approvals, list) else []
    return {
        "artifact_status": _artifact_status(payload),
        "pending_actions": action_list,
        "latest_promotion_approval": payload.get("latest_promotion_approval"),
        "recent_promotion_approvals": approval_list[-5:],
        "manual_overrides": _mapping(payload.get("manual_overrides")),
    }


def _section(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "artifact_status": _artifact_status(payload),
        "payload": dict(payload),
    }


def build_operator_control_plane(
    *,
    health: Mapping[str, Any] | None = None,
    readiness: Mapping[str, Any] | None = None,
    runtime_gonogo: Mapping[str, Any] | None = None,
    runtime_performance: Mapping[str, Any] | None = None,
    oms: Mapping[str, Any] | None = None,
    model_registry: Mapping[str, Any] | None = None,
    latest_research: Mapping[str, Any] | None = None,
    weekend_research: Mapping[str, Any] | None = None,
    drift: Mapping[str, Any] | None = None,
    surveillance: Mapping[str, Any] | None = None,
    risk_verifier: Mapping[str, Any] | None = None,
    paper_sampling: Mapping[str, Any] | None = None,
    operator_actions: Mapping[str, Any] | None = None,
    huggingface_research: Mapping[str, Any] | None = None,
    upward_trajectory: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Return an operator snapshot without invoking mutating runtime actions."""

    health_payload = _mapping(health)
    readiness_payload = _mapping(readiness)
    runtime_gonogo_payload = _mapping(runtime_gonogo)
    runtime_performance_payload = _mapping(runtime_performance)
    oms_payload = _mapping(oms)
    model_registry_payload = _mapping(model_registry)
    latest_research_payload = _mapping(latest_research)
    weekend_research_payload = _mapping(weekend_research)
    drift_payload = _mapping(drift)
    surveillance_payload = _mapping(surveillance)
    risk_verifier_payload = _mapping(risk_verifier)
    paper_sampling_payload = _mapping(paper_sampling)
    operator_actions_payload = _mapping(operator_actions)
    huggingface_research_payload = _mapping(huggingface_research)
    upward_trajectory_payload = _mapping(upward_trajectory)
    generated = generated_at.astimezone(UTC) if generated_at else datetime.now(UTC)
    launch_profile = launch_profile_payload(resolve_launch_profile())
    attention_flags = [
        str(flag)
        for flag in health_payload.get("attention_flags", [])
        if flag not in (None, "")
    ] if isinstance(health_payload.get("attention_flags"), list) else []
    missing_sections = [
        name
        for name, payload in (
            ("health", health_payload),
            ("readiness", readiness_payload),
            ("runtime_gonogo", runtime_gonogo_payload),
            ("runtime_performance", runtime_performance_payload),
            ("oms", oms_payload),
            ("model_registry", model_registry_payload),
            ("latest_research", latest_research_payload),
            ("weekend_research", weekend_research_payload),
            ("drift", drift_payload),
            ("surveillance", surveillance_payload),
            ("risk_verifier", risk_verifier_payload),
            ("paper_sampling", paper_sampling_payload),
            ("operator_actions", operator_actions_payload),
            ("huggingface_research", huggingface_research_payload),
            ("upward_trajectory", upward_trajectory_payload),
        )
        if not payload
    ]
    return {
        "schema_version": "1.0.0",
        "artifact_type": "operator_control_plane",
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "status": "complete" if not missing_sections else "partial",
        "read_only": True,
        "safety_contract": {
            "places_orders": False,
            "edits_environment": False,
            "restarts_service": False,
            "patches_code": False,
            "writes": ["output_artifact_only"],
        },
        "missing_sections": missing_sections,
        "health": {
            "artifact_status": _artifact_status(health_payload),
            "ok": health_payload.get("ok"),
            "status": health_payload.get("status"),
            "reason": health_payload.get("reason"),
            "attention_flags": attention_flags,
            "broker": _mapping(health_payload.get("broker")),
            "database": _mapping(health_payload.get("database")),
            "data_provider": _mapping(health_payload.get("data_provider")),
        },
        "launch_profile": launch_profile,
        "readiness": _section(readiness_payload, label="live_capital_readiness"),
        "go_no_go": _go_no_go_payload(readiness_payload, runtime_gonogo_payload),
        "orders_positions_oms": _orders_positions_oms(
            health_payload,
            runtime_performance_payload,
            oms_payload,
        ),
        "model_registry": _section(model_registry_payload, label="model_registry"),
        "latest_research": _section(latest_research_payload, label="latest_research"),
        "weekend_research": {
            **_section(weekend_research_payload, label="weekend_research"),
            "research_only": True,
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "manual_approval_required": True,
        },
        "drift": _section(drift_payload, label="drift"),
        "surveillance": _section(surveillance_payload, label="surveillance"),
        "risk_verifier": _section(risk_verifier_payload, label="risk_verifier"),
        "paper_sampling": _section(paper_sampling_payload, label="paper_sampling"),
        "operator_actions": _operator_actions(operator_actions_payload),
        "huggingface_research": {
            **_section(huggingface_research_payload, label="huggingface_research"),
            "research_only": True,
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
        "upward_trajectory": {
            **_section(upward_trajectory_payload, label="upward_trajectory"),
            "research_only": True,
            "paper_only_diagnostics": True,
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "manual_approval_required": True,
        },
    }


def _default_path(path_value: str) -> Path:
    return resolve_runtime_artifact_path(path_value, default_relative=path_value)


def _default_output() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/operator_control_plane_latest.json",
        default_relative="runtime/operator_control_plane_latest.json",
        for_write=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-json", type=Path, default=None)
    parser.add_argument("--health-url", default="")
    parser.add_argument("--readiness-json", type=Path, default=None)
    parser.add_argument("--runtime-gonogo-json", type=Path, default=None)
    parser.add_argument("--runtime-performance-json", type=Path, default=None)
    parser.add_argument("--oms-json", type=Path, default=None)
    parser.add_argument("--model-registry-json", type=Path, default=None)
    parser.add_argument("--latest-research-json", type=Path, default=None)
    parser.add_argument("--weekend-research-json", type=Path, default=None)
    parser.add_argument("--drift-json", type=Path, default=None)
    parser.add_argument("--surveillance-json", type=Path, default=None)
    parser.add_argument("--risk-verifier-json", type=Path, default=None)
    parser.add_argument("--paper-sampling-json", type=Path, default=None)
    parser.add_argument("--operator-actions-json", type=Path, default=None)
    parser.add_argument("--huggingface-research-json", type=Path, default=None)
    parser.add_argument("--upward-trajectory-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    health = (
        _read_json_mapping(args.health_json)
        if args.health_json
        else _health_from_endpoint(args.health_url)
    )
    report = build_operator_control_plane(
        health=health,
        readiness=_read_json_mapping(
            args.readiness_json or _default_path("runtime/live_capital_readiness_latest.json")
        ),
        runtime_gonogo=_read_json_mapping(
            args.runtime_gonogo_json or _default_path("runtime/runtime_gonogo_status_latest.json")
        ),
        runtime_performance=_read_json_mapping(
            args.runtime_performance_json
            or _default_path("runtime/runtime_performance_report_latest.json")
        ),
        oms=_read_json_mapping(args.oms_json or _default_path("runtime/oms_lifecycle_parity_latest.json")),
        model_registry=_read_json_mapping(
            args.model_registry_json or _default_path("models/registry_index.json")
        ),
        latest_research=_read_json_mapping(
            args.latest_research_json
            or _default_path("runtime/research_reports/latest/daily_readiness_latest.json")
        ),
        weekend_research=_read_json_mapping(
            args.weekend_research_json
            or _default_path("runtime/research_reports/latest/weekend_research_latest.json")
        ),
        drift=_read_json_mapping(args.drift_json or _default_path("runtime/model_data_drift_monitor_latest.json")),
        surveillance=_read_json_mapping(
            args.surveillance_json or _default_path("runtime/reports/post_trade_surveillance_latest.json")
        ),
        risk_verifier=_read_json_mapping(
            args.risk_verifier_json
            or _default_path("runtime/reports/pretrade_risk_control_verification_latest.json")
        ),
        paper_sampling=_read_json_mapping(
            args.paper_sampling_json or _default_path("runtime/paper_sampling_state_latest.json")
        ),
        operator_actions=_read_json_mapping(
            args.operator_actions_json or _default_path("runtime/operator_actions_latest.json")
        ),
        huggingface_research=_read_json_mapping(
            args.huggingface_research_json
            or _default_path("runtime/research_reports/latest/hf_discovery_latest.json")
        ),
        upward_trajectory=_read_json_mapping(
            args.upward_trajectory_json
            or _default_path("runtime/reports/upward_trajectory_latest.json")
        ),
    )
    output = args.output_json or _default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
