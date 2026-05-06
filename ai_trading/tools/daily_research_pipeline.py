"""Build the operator-facing daily research report bundle."""

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


def _default_path(relative: str) -> Path:
    return resolve_runtime_artifact_path(relative, default_relative=relative)


def _default_output(report_date: str) -> tuple[Path, Path, Path]:
    root = resolve_runtime_artifact_path(
        "runtime/research_reports",
        default_relative="runtime/research_reports",
        for_write=True,
    )
    return (
        root / f"daily_research_{report_date.replace('-', '')}.json",
        root / "daily_research_latest.json",
        root / f"daily_research_{report_date.replace('-', '')}.md",
    )


def _status(payload: Mapping[str, Any], default: str = "missing") -> str:
    raw = payload.get("status")
    if isinstance(raw, Mapping):
        return str(raw.get("status") or default)
    return str(raw or default)


def _promotion_ok(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("promotion_ready") or payload.get("status") == "ready_for_approval")


def _trade_allowed(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    runtime_health = _nested(report, "runtime_health")
    if not bool(runtime_health.get("ok")):
        reasons.append("runtime_health_not_ok")
    if str(_nested(report, "data_provider_health").get("status") or "").lower() not in {
        "healthy",
        "ready",
        "warming_up",
    }:
        reasons.append("data_provider_not_healthy")
    if _nested(report, "live_cost_status").get("available") is False:
        reasons.append("live_cost_unavailable")
    if int(_nested(report, "live_cost_status").get("breach_count") or 0) > 0:
        reasons.append("live_cost_breach")
    replay_status = _nested(report, "replay_status")
    if replay_status and replay_status.get("ok") is False:
        reasons.append("replay_governance_failed")
    launch_profile = _nested(report, "launch_profile")
    if str(launch_profile.get("name") or "").startswith("live_"):
        promotion_status = _nested(report, "promotion_status")
        if not bool(promotion_status.get("promotion_ready")):
            reasons.append("promotion_not_ready")
    if str(_nested(report, "memory_status").get("status") or "").lower() == "critical":
        reasons.append("memory_status_critical")
    return not reasons, reasons


def build_daily_research_report(
    *,
    report_date: str,
    health: Mapping[str, Any] | None = None,
    live_cost_model: Mapping[str, Any] | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    replay_governance: Mapping[str, Any] | None = None,
    symbol_scorecard: Mapping[str, Any] | None = None,
    promotion_report: Mapping[str, Any] | None = None,
    runtime_gonogo: Mapping[str, Any] | None = None,
    memory_audit: Mapping[str, Any] | None = None,
    artifact_retention: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    health = health or {}
    live_cost_model = live_cost_model or {}
    shadow_report = shadow_report or {}
    replay_governance = replay_governance or {}
    symbol_scorecard = symbol_scorecard or {}
    promotion_report = promotion_report or {}
    runtime_gonogo = runtime_gonogo or {}
    memory_audit = memory_audit or {}
    artifact_retention = artifact_retention or {}
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_type": "daily_research_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "runtime_health": {
            "ok": bool(health.get("ok")),
            "status": health.get("status"),
            "reason": health.get("reason"),
            "broker": _nested(health, "broker"),
            "database": _nested(health, "database"),
            "oms_invariants": _nested(health, "oms_invariants"),
            "oms_lifecycle_parity": _nested(health, "oms_lifecycle_parity"),
        },
        "data_provider_health": _nested(health, "data_provider"),
        "provider_authority": _nested(health, "provider_authority"),
        "launch_profile": launch_profile_payload(resolve_launch_profile()),
        "live_cost_status": _nested(live_cost_model, "status"),
        "shadow_status": {
            "status": _status(_nested(shadow_report, "sample_gate")),
            "decision_summary": _nested(shadow_report, "decision_summary"),
            "sample_gate": _nested(shadow_report, "sample_gate"),
        },
        "replay_status": _nested(replay_governance, "replay_live_parity_gate")
        or _nested(replay_governance, "status"),
        "promotion_status": {
            "status": promotion_report.get("status", "missing"),
            "promotion_ready": _promotion_ok(promotion_report),
            "gates": _nested(promotion_report, "gates"),
        },
        "runtime_gonogo": {
            "gate_passed": bool(runtime_gonogo.get("gate_passed")),
            "failed_checks": list(runtime_gonogo.get("failed_checks", []))
            if isinstance(runtime_gonogo.get("failed_checks"), list)
            else [],
        },
        "memory_status": {
            "status": memory_audit.get("status", "missing"),
            "service_memory": _nested(memory_audit, "service_memory"),
            "recent_memory_samples": _nested(memory_audit, "recent_memory_samples"),
            "observations": list(memory_audit.get("observations", []))
            if isinstance(memory_audit.get("observations"), list)
            else [],
        },
        "artifact_retention": {
            "status": artifact_retention.get("status", "missing"),
            "apply": bool(artifact_retention.get("apply", False)),
            "total_reclaimable_mb": artifact_retention.get("total_reclaimable_mb"),
        },
        "symbol_actions": {
            "summary": _nested(symbol_scorecard, "summary"),
            "policy": _nested(symbol_scorecard, "policy"),
            "shadow_promotion": _nested(symbol_scorecard, "shadow_promotion"),
            "symbols": symbol_scorecard.get("symbols", []),
        },
    }
    allowed, reasons = _trade_allowed(report)
    profile_name = str(_nested(report, "launch_profile").get("name") or "paper_observe")
    report["trade_allowed"] = bool(allowed)
    report["blocked_reasons"] = reasons
    report["recommended_next_session_mode"] = (
        profile_name if allowed else ("paper_only" if profile_name.startswith("live_") else "observe")
    )
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    reasons = report.get("blocked_reasons") if isinstance(report.get("blocked_reasons"), list) else []
    shadow_suggestions = _nested(report, "symbol_actions", "shadow_promotion").get(
        "suggestions",
        [],
    )
    suggestion_text = "none"
    if isinstance(shadow_suggestions, list) and shadow_suggestions:
        suggestion_text = ", ".join(
            str(item.get("symbol"))
            for item in shadow_suggestions
            if isinstance(item, Mapping) and item.get("symbol")
        ) or "none"
    return "\n".join(
        [
            f"# Daily Research {report.get('report_date')}",
            "",
            f"- Trade allowed: `{str(report.get('trade_allowed')).lower()}`",
            f"- Recommended next session mode: `{report.get('recommended_next_session_mode')}`",
            f"- Blockers: {', '.join(str(item) for item in reasons) if reasons else 'none'}",
            f"- Runtime health: `{_nested(report, 'runtime_health').get('status')}`",
            f"- Data provider: `{_nested(report, 'data_provider_health').get('status')}`",
            f"- Live cost: `{_nested(report, 'live_cost_status').get('status', 'missing')}`",
            f"- Memory: `{_nested(report, 'memory_status').get('status', 'missing')}`",
            f"- Promotion: `{_nested(report, 'promotion_status').get('status', 'missing')}`",
            f"- Shadow promotion candidates: {suggestion_text}",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--health-json", type=Path, default=None)
    parser.add_argument("--health-url", default="http://127.0.0.1:9001/healthz")
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--shadow-report-json", type=Path, default=None)
    parser.add_argument("--replay-governance-json", type=Path, default=None)
    parser.add_argument("--symbol-scorecard-json", type=Path, default=None)
    parser.add_argument("--promotion-report-json", type=Path, default=None)
    parser.add_argument("--runtime-gonogo-json", type=Path, default=None)
    parser.add_argument("--memory-audit-json", type=Path, default=None)
    parser.add_argument("--artifact-retention-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args(argv)
    default_output, default_latest, default_md = _default_output(args.report_date)
    output_json = args.output_json or default_output
    latest_json = args.latest_json or default_latest
    output_md = args.output_md or default_md
    report = build_daily_research_report(
        report_date=str(args.report_date),
        health=(
            _read_json(args.health_json)
            if args.health_json is not None
            else _health_from_endpoint(str(args.health_url))
            or _read_json(_default_path("runtime/health_latest.json"))
        ),
        live_cost_model=_read_json(
            args.live_cost_model_json or _default_path("runtime/live_cost_model_latest.json")
        ),
        shadow_report=_read_json(
            args.shadow_report_json
            or _default_path("runtime/ml_shadow_report_latest.json")
        ),
        replay_governance=_read_json(
            args.replay_governance_json
            or _default_path("runtime/replay_governance_refresh_latest.json")
        ),
        symbol_scorecard=_read_json(
            args.symbol_scorecard_json or _default_path("runtime/symbol_universe_scorecard_latest.json")
        ),
        promotion_report=_read_json(
            args.promotion_report_json or _default_path("runtime/promotion_report_latest.json")
        ),
        runtime_gonogo=_read_json(args.runtime_gonogo_json),
        memory_audit=_read_json(
            args.memory_audit_json or _default_path("runtime/memory_hotspot_audit_latest.json")
        ),
        artifact_retention=_read_json(
            args.artifact_retention_json
            or _default_path("runtime/runtime_artifact_retention_latest.json")
        ),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_json.parent.mkdir(parents=True, exist_ok=True)
    latest_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(report), encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "trade_allowed": report["trade_allowed"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
