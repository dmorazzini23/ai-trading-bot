"""Build a gated model-promotion report artifact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.config.management import get_env
from ai_trading.models.artifacts import default_manifest_path, load_artifact_manifest, verify_artifact
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


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


def _parse_ts(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    token = str(value).strip()
    if not token:
        return None
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(token)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _freshness_gate(
    payload: Mapping[str, Any],
    *,
    label: str,
    now: datetime,
    max_age_hours: float,
) -> dict[str, Any]:
    for key in ("generated_at", "as_of", "ts", "timestamp", "created_at"):
        parsed = _parse_ts(payload.get(key))
        if parsed is None:
            continue
        age_hours = max(0.0, (now - parsed).total_seconds() / 3600.0)
        future_dated = parsed > now
        ok = bool(age_hours <= max_age_hours and not future_dated)
        return {
            "ok": ok,
            "label": label,
            "timestamp": parsed.isoformat().replace("+00:00", "Z"),
            "age_hours": age_hours,
            "max_age_hours": max_age_hours,
            "reason": "ok" if ok else ("evidence_future_dated" if future_dated else "evidence_stale"),
        }
    for nested_key in ("replay", "source", "source_artifact", "evidence", "payload"):
        nested_replay = payload.get(nested_key)
        if not isinstance(nested_replay, Mapping):
            continue
        return _freshness_gate(
            nested_replay,
            label=label,
            now=now,
            max_age_hours=max_age_hours,
        )
    return {
        "ok": False,
        "label": label,
        "timestamp": None,
        "age_hours": None,
        "max_age_hours": max_age_hours,
        "reason": "evidence_timestamp_missing",
    }


def _replay_gate(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not payload:
        return {"ok": False, "label": label, "reason": "replay_evidence_missing"}
    artifact_type = str(payload.get("artifact_type") or "").strip()
    if artifact_type and artifact_type != "offline_replay_summary":
        return {
            "ok": False,
            "label": label,
            "reason": "unsupported_replay_schema",
            "artifact_type": artifact_type,
        }
    if "replay" in payload and "aggregate" not in payload:
        return {
            "ok": False,
            "label": label,
            "reason": "unsupported_replay_schema",
            "artifact_type": artifact_type or "replay_governance_summary",
        }
    aggregate_raw = payload.get("aggregate")
    aggregate = aggregate_raw if isinstance(aggregate_raw, Mapping) else payload
    net_edge = _as_float(
        aggregate.get("expectancy_bps")
        if aggregate.get("expectancy_bps") is not None
        else aggregate.get("net_edge_bps")
    )
    violations = int(_as_float(aggregate.get("violation_count")) or 0.0)
    if violations <= 0:
        violations_raw = aggregate.get("violations_count")
        violations = int(_as_float(violations_raw) or 0.0)
    total_trades = int(
        _as_float(aggregate.get("total_trades") or aggregate.get("fill_events") or 0.0)
        or 0.0
    )
    ok = bool(net_edge is not None and net_edge > 0.0 and violations == 0 and total_trades > 0)
    return {
        "ok": ok,
        "label": label,
        "net_edge_bps": net_edge,
        "total_trades": total_trades,
        "violations": violations,
        "reason": "ok" if ok else "replay_gate_failed",
    }


def _evidence_authority_gate(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    authority_raw = payload.get("authority")
    authority = authority_raw if isinstance(authority_raw, Mapping) else {}
    if not authority:
        return {"ok": False, "label": label, "reason": "authority_metadata_missing"}
    source_providers_raw = authority.get("source_providers", ())
    if isinstance(source_providers_raw, str):
        source_providers = {source_providers_raw.strip().lower()} if source_providers_raw.strip() else set()
    else:
        try:
            source_providers = {
                str(provider).strip().lower()
                for provider in source_providers_raw
                if str(provider).strip()
            }
        except TypeError:
            source_providers = {str(source_providers_raw).strip().lower()}
    source_providers.discard("")
    timestamp_authoritative = bool(authority.get("timestamp_authoritative", True))
    research_synthetic = bool(authority.get("research_synthetic", False))
    yahoo_sourced = bool(source_providers.intersection({"yahoo", "yfinance", "yf"}))
    ok = bool(timestamp_authoritative and not research_synthetic and not yahoo_sourced)
    reasons: list[str] = []
    if not timestamp_authoritative:
        reasons.append("timestamp_authority_missing")
    if research_synthetic:
        reasons.append("research_synthetic_data")
    if yahoo_sourced:
        reasons.append("yahoo_research_boundary")
    return {
        "ok": ok,
        "label": label,
        "reason": "ok" if ok else ";".join(reasons),
        "timestamp_authoritative": timestamp_authoritative,
        "research_synthetic": research_synthetic,
        "source_providers": sorted(source_providers),
    }


def _shadow_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"ok": False, "reason": "shadow_report_missing"}
    sample_gate_raw = payload.get("sample_gate")
    sample_gate = sample_gate_raw if isinstance(sample_gate_raw, Mapping) else {}
    markout_raw = payload.get("markout_summary")
    markout = markout_raw if isinstance(markout_raw, Mapping) else {}
    decision_raw = payload.get("decision_summary")
    decision = decision_raw if isinstance(decision_raw, Mapping) else {}
    challenger_mean = _as_float(markout.get("challenger_mean_net_markout_bps"))
    skew_rate = _as_float(decision.get("skew_breach_rate")) or 0.0
    status = str(sample_gate.get("status") or "").strip().lower()
    rows = int(_as_float(decision.get("rows")) or _as_float(payload.get("filtered_rows")) or 0.0)
    ok = bool(
        status in {"review_eligible", "sufficient", "ready"}
        and challenger_mean is not None
        and challenger_mean > 0.0
        and skew_rate <= 0.01
    )
    return {
        "ok": ok,
        "reason": "ok" if ok else "shadow_gate_failed",
        "sample_gate_status": status,
        "rows": rows,
        "challenger_mean_net_markout_bps": challenger_mean,
        "skew_breach_rate": skew_rate,
    }


def _live_cost_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"ok": False, "reason": "live_cost_model_missing"}
    status_raw = payload.get("status")
    status = status_raw if isinstance(status_raw, Mapping) else {}
    breach_count = int(_as_float(status.get("breach_count")) or 0.0)
    available = bool(status.get("available", bool(payload)))
    status_token = str(status.get("status") or payload.get("status") or "").strip().lower()
    status_ready = status_token in {"ready", "ok"}
    ok = bool(available and status_ready and breach_count == 0)
    return {
        "ok": ok,
        "reason": "ok" if ok else "live_cost_gate_failed",
        "status": status.get("status"),
        "breach_count": breach_count,
        "available": available,
        "status_ready": status_ready,
    }


def _decay_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"ok": False, "reason": "runtime_decay_controls_missing"}
    actions_raw = payload.get("actions")
    actions = actions_raw if isinstance(actions_raw, Mapping) else {}
    entries_allowed = bool(actions.get("entries_allowed", True))
    max_action = str(actions.get("max_action") or "normal")
    ok = bool(entries_allowed and max_action in {"normal", "reduce_size"})
    return {
        "ok": ok,
        "reason": "ok" if ok else "runtime_decay_gate_failed",
        "max_action": max_action,
        "entries_allowed": entries_allowed,
        "size_scale": _as_float(actions.get("size_scale")),
        "reasons": list(actions.get("reasons")) if isinstance(actions.get("reasons"), list) else [],
    }


def _manifest_payload(model_path: Path, manifest_path: Path) -> dict[str, Any]:
    ok, reason = verify_artifact(model_path=model_path, manifest_path=manifest_path)
    payload: dict[str, Any] = {
        "ok": ok,
        "reason": reason,
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
    }
    if ok:
        manifest = load_artifact_manifest(manifest_path)
        payload.update(
            {
                "model_version": manifest.model_version,
                "checksum_sha256": manifest.checksum_sha256,
                "created_ts": manifest.created_ts,
                "metadata": dict(manifest.metadata or {}),
            }
        )
    return payload


def build_promotion_report(
    *,
    model_path: Path,
    manifest_path: Path | None = None,
    strategy: str = "replay_aligned",
    full_replay: Mapping[str, Any] | None = None,
    tail_replay: Mapping[str, Any] | None = None,
    recent_replay: Mapping[str, Any] | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    live_cost_model: Mapping[str, Any] | None = None,
    runtime_decay_controls: Mapping[str, Any] | None = None,
    current_champion_path: str | None = None,
    rollback_command: str | None = None,
    generated_at: datetime | None = None,
    max_evidence_age_hours: float | None = None,
) -> dict[str, Any]:
    """Return a promotion report with hard gates and rollback notes."""

    model_path = model_path.expanduser()
    resolved_manifest = manifest_path.expanduser() if manifest_path else default_manifest_path(model_path)
    generated = generated_at.astimezone(UTC) if generated_at else datetime.now(UTC)
    max_age_hours = (
        float(max_evidence_age_hours)
        if max_evidence_age_hours is not None
        else float(
            get_env(
                "AI_TRADING_PROMOTION_MAX_EVIDENCE_AGE_HOURS",
                "96",
                cast=float,
                resolve_aliases=False,
            )
            or 96.0
        )
    )
    manifest_gate = _manifest_payload(model_path, resolved_manifest)
    replay_gates = {
        "full": _replay_gate(full_replay or {}, label="full"),
        "tail": _replay_gate(tail_replay or full_replay or {}, label="tail"),
        "recent": _replay_gate(recent_replay or tail_replay or full_replay or {}, label="recent"),
    }
    authority_gates = {
        "full_replay": _evidence_authority_gate(full_replay or {}, label="full_replay"),
        "tail_replay": _evidence_authority_gate(
            tail_replay or full_replay or {},
            label="tail_replay",
        ),
        "recent_replay": _evidence_authority_gate(
            recent_replay or tail_replay or full_replay or {},
            label="recent_replay",
        ),
    }
    freshness_gates = {
        "full_replay": _freshness_gate(
            full_replay or {},
            label="full_replay",
            now=generated,
            max_age_hours=max_age_hours,
        ),
        "tail_replay": _freshness_gate(
            tail_replay or full_replay or {},
            label="tail_replay",
            now=generated,
            max_age_hours=max_age_hours,
        ),
        "recent_replay": _freshness_gate(
            recent_replay or tail_replay or full_replay or {},
            label="recent_replay",
            now=generated,
            max_age_hours=max_age_hours,
        ),
        "shadow_report": _freshness_gate(
            shadow_report or {},
            label="shadow_report",
            now=generated,
            max_age_hours=max_age_hours,
        ),
        "live_cost_model": _freshness_gate(
            live_cost_model or {},
            label="live_cost_model",
            now=generated,
            max_age_hours=max_age_hours,
        ),
        "runtime_decay_controls": _freshness_gate(
            runtime_decay_controls or {},
            label="runtime_decay_controls",
            now=generated,
            max_age_hours=max_age_hours,
        ),
    }
    shadow_gate = _shadow_gate(shadow_report or {})
    live_cost_gate = _live_cost_gate(live_cost_model or {})
    decay_gate = _decay_gate(runtime_decay_controls or {})
    gates = {
        "manifest_verified": bool(manifest_gate.get("ok")),
        "full_replay_positive": bool(replay_gates["full"].get("ok")),
        "tail_replay_positive": bool(replay_gates["tail"].get("ok")),
        "recent_replay_positive": bool(replay_gates["recent"].get("ok")),
        "shadow_telemetry_acceptable": bool(shadow_gate.get("ok")),
        "live_cost_model_acceptable": bool(live_cost_gate.get("ok")),
        "runtime_decay_controls_acceptable": bool(decay_gate.get("ok")),
        "evidence_fresh": all(bool(gate.get("ok")) for gate in freshness_gates.values()),
        "evidence_authority_acceptable": all(
            bool(gate.get("ok")) for gate in authority_gates.values()
        ),
    }
    promotion_ready = all(gates.values())
    rollback = rollback_command or (
        f"AI_TRADING_MODEL_PATH={current_champion_path}"
        if current_champion_path
        else "restore previous AI_TRADING_MODEL_PATH and restart ai-trading.service"
    )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "model_promotion_report",
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "authority": {
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "manual_approval_required": True,
        },
        "strategy": str(strategy or "replay_aligned"),
        "model": {
            "path": str(model_path),
            "manifest_path": str(resolved_manifest),
            "current_champion_path": current_champion_path,
        },
        "manifest": manifest_gate,
        "replay": replay_gates,
        "evidence_authority": authority_gates,
        "evidence_freshness": freshness_gates,
        "shadow": shadow_gate,
        "live_cost_model": live_cost_gate,
        "runtime_decay_controls": decay_gate,
        "gates": gates,
        "promotion_ready": bool(promotion_ready),
        "status": "ready_for_approval" if promotion_ready else "blocked",
        "risk_notes": [
            "promotion report is advisory; it does not mutate runtime model paths",
            "keep the current champion available until the candidate proves itself live",
            "rollback command must be operator-reviewed before production use",
        ],
        "rollback": {
            "command": rollback,
            "current_champion_path": current_champion_path,
        },
    }


def _default_runtime_path(env_key: str, default_relative: str) -> Path:
    configured = str(get_env(env_key, default_relative, cast=str, resolve_aliases=False) or default_relative)
    return resolve_runtime_artifact_path(configured, default_relative=default_relative)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--strategy", type=str, default="replay_aligned")
    parser.add_argument("--full-replay-json", type=Path, default=None)
    parser.add_argument("--tail-replay-json", type=Path, default=None)
    parser.add_argument("--recent-replay-json", type=Path, default=None)
    parser.add_argument("--shadow-report-json", type=Path, default=None)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--runtime-decay-controls-json", type=Path, default=None)
    parser.add_argument("--current-champion-path", type=str, default="")
    parser.add_argument("--rollback-command", type=str, default="")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    output_json = (
        args.output_json
        if args.output_json.is_absolute()
        else resolve_runtime_artifact_path(
            args.output_json,
            default_relative=str(args.output_json),
            for_write=True,
        )
    )

    report = build_promotion_report(
        model_path=args.model_path,
        manifest_path=args.manifest_path,
        strategy=args.strategy,
        full_replay=_read_json_mapping(args.full_replay_json),
        tail_replay=_read_json_mapping(args.tail_replay_json),
        recent_replay=_read_json_mapping(args.recent_replay_json),
        shadow_report=_read_json_mapping(args.shadow_report_json),
        live_cost_model=_read_json_mapping(
            args.live_cost_model_json
            or _default_runtime_path("AI_TRADING_LIVE_COST_MODEL_PATH", "runtime/live_cost_model_latest.json")
        ),
        runtime_decay_controls=_read_json_mapping(
            args.runtime_decay_controls_json
            or _default_runtime_path(
                "AI_TRADING_RUNTIME_DECAY_CONTROLS_PATH",
                "runtime/runtime_decay_controls_latest.json",
            )
        ),
        current_champion_path=str(args.current_champion_path or "").strip() or None,
        rollback_command=str(args.rollback_command or "").strip() or None,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        json.dumps(
            {
                "path": str(output_json),
                "promotion_ready": report["promotion_ready"],
                "status": report["status"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0 if bool(report["promotion_ready"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
