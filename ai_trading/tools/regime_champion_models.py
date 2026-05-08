"""Build regime-scoped champion model selection artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


_AUTHORITY_RANKS = {
    "off": 0,
    "observe": 1,
    "shadow": 2,
    "paper": 3,
    "paper_trade": 3,
    "canary": 4,
    "live_canary": 4,
    "restricted": 5,
    "live_restricted": 5,
    "live": 6,
    "production": 6,
}


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_int(value: Any) -> int | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "approved", "pass"}
    return bool(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _candidate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("regimes", "candidates", "models"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, Mapping)]
    if payload:
        return [dict(payload)]
    return []


def _authority_rank(value: Any) -> int:
    return _AUTHORITY_RANKS.get(str(value or "observe").strip().lower(), 0)


def _authority_increase(current_authority: Any, requested_authority: Any) -> bool:
    return _authority_rank(requested_authority) > _authority_rank(current_authority)


def _cost_adjusted_expectancy_bps(candidate: Mapping[str, Any], evidence: Mapping[str, Any]) -> float | None:
    for key in ("cost_adjusted_expectancy_bps", "net_expectancy_bps", "expectancy_bps"):
        parsed = _as_float(candidate.get(key))
        if parsed is not None:
            return parsed
        parsed = _as_float(evidence.get(key))
        if parsed is not None:
            return parsed
    gross = _as_float(candidate.get("gross_expectancy_bps"))
    if gross is None:
        gross = _as_float(evidence.get("gross_expectancy_bps"))
    cost = _as_float(candidate.get("avg_cost_bps"))
    if cost is None:
        cost = _as_float(candidate.get("cost_bps"))
    if cost is None:
        cost = _as_float(evidence.get("avg_cost_bps"))
    if cost is None:
        cost = _as_float(evidence.get("cost_bps"))
    if gross is None:
        return None
    return float(gross - (cost or 0.0))


def _sample_count(candidate: Mapping[str, Any], evidence: Mapping[str, Any]) -> int:
    for key in ("sample_count", "samples", "n", "rows", "total_trades", "trade_count"):
        parsed = _as_int(candidate.get(key))
        if parsed is not None:
            return parsed
        parsed = _as_int(evidence.get(key))
        if parsed is not None:
            return parsed
    replay = _mapping(evidence.get("replay") or candidate.get("replay"))
    aggregate = _mapping(replay.get("aggregate"))
    for key in ("total_trades", "samples", "rows"):
        parsed = _as_int(aggregate.get(key))
        if parsed is not None:
            return parsed
    shadow = _mapping(evidence.get("shadow") or candidate.get("shadow"))
    for key in ("filtered_rows", "rows", "sample_count"):
        parsed = _as_int(shadow.get(key))
        if parsed is not None:
            return parsed
    return 0


def _evidence_payload(candidate: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    evidence = _mapping(candidate.get("evidence"))
    for key in keys:
        payload = _mapping(candidate.get(key))
        if payload:
            return payload
        payload = _mapping(evidence.get(key))
        if payload:
            return payload
    return {}


def _replay_gate(payload: Mapping[str, Any], *, min_expectancy_bps: float) -> dict[str, Any]:
    if not payload:
        return {"ok": False, "reason": "replay_evidence_missing"}
    aggregate = _mapping(payload.get("aggregate")) or _mapping(payload.get("summary")) or dict(payload)
    status = str(payload.get("status") or aggregate.get("status") or "").strip().lower()
    explicit = payload.get("ok", payload.get("gate_passed", payload.get("passed")))
    expectancy = _as_float(
        aggregate.get("cost_adjusted_expectancy_bps")
        if aggregate.get("cost_adjusted_expectancy_bps") is not None
        else aggregate.get("net_expectancy_bps")
    )
    if expectancy is None:
        expectancy = _as_float(aggregate.get("expectancy_bps"))
    samples = _as_int(aggregate.get("total_trades") or aggregate.get("samples") or aggregate.get("rows")) or 0
    violations = _as_int(aggregate.get("violation_count") or aggregate.get("violations_count")) or 0
    ok = bool(
        status not in {"fail", "failed", "blocked", "error"}
        and violations == 0
        and (
            _as_bool(explicit)
            or (expectancy is not None and expectancy >= min_expectancy_bps and samples > 0)
        )
    )
    return {
        "ok": ok,
        "reason": "ok" if ok else "replay_gate_failed",
        "status": status or None,
        "samples": samples,
        "cost_adjusted_expectancy_bps": expectancy,
        "violations": violations,
    }


def _shadow_gate(payload: Mapping[str, Any], *, min_samples: int) -> dict[str, Any]:
    if not payload:
        return {"ok": False, "reason": "shadow_evidence_missing"}
    sample_gate = _mapping(payload.get("sample_gate"))
    decision = _mapping(payload.get("decision_summary"))
    markout = _mapping(payload.get("markout_summary"))
    status = str(payload.get("status") or sample_gate.get("status") or "").strip().lower()
    explicit = payload.get("ok", payload.get("gate_passed", payload.get("passed")))
    rows = (
        _as_int(payload.get("filtered_rows"))
        or _as_int(payload.get("rows"))
        or _as_int(decision.get("rows"))
        or 0
    )
    challenger_markout = _as_float(
        markout.get("challenger_mean_net_markout_bps")
        if markout.get("challenger_mean_net_markout_bps") is not None
        else payload.get("mean_net_markout_bps")
    )
    status_ok = status in {"", "ok", "pass", "passed", "ready", "sufficient", "review_eligible"}
    markout_ok = challenger_markout is None or challenger_markout > 0.0
    ok = bool(
        status_ok
        and rows >= min_samples
        and markout_ok
        and (_as_bool(explicit) or status in {"ready", "sufficient", "review_eligible", "ok", "pass", "passed"})
    )
    return {
        "ok": ok,
        "reason": "ok" if ok else "shadow_gate_failed",
        "status": status or None,
        "samples": rows,
        "mean_net_markout_bps": challenger_markout,
    }


def _approval_for_regime(approval: Mapping[str, Any], regime: str, model_id: str) -> dict[str, Any]:
    if not approval:
        return {"approved": False, "reason": "manual_approval_missing"}
    approved_regimes = approval.get("approved_regimes")
    if isinstance(approved_regimes, list) and regime in {str(item) for item in approved_regimes}:
        return {"approved": True, "reason": "approved_regime"}
    approved_models = approval.get("approved_models")
    if isinstance(approved_models, list) and model_id in {str(item) for item in approved_models}:
        return {"approved": True, "reason": "approved_model"}
    authority_increases = _mapping(approval.get("authority_increases"))
    if _as_bool(authority_increases.get(regime)) or _as_bool(authority_increases.get(model_id)):
        return {"approved": True, "reason": "approved_authority_increase"}
    approvals = approval.get("approvals")
    if isinstance(approvals, list):
        for row in approvals:
            if not isinstance(row, Mapping):
                continue
            row_regime = str(row.get("regime") or "").strip()
            row_model = str(row.get("model_id") or "").strip()
            if row_regime not in {"", regime}:
                continue
            if row_model not in {"", model_id}:
                continue
            if _as_bool(row.get("approved")):
                return {"approved": True, "reason": str(row.get("reason") or "approved_event")}
    if _as_bool(approval.get("approved")) and str(approval.get("regime") or regime) == regime:
        return {"approved": True, "reason": str(approval.get("reason") or "approved")}
    return {"approved": False, "reason": "manual_approval_missing"}


def _current_for_regime(registry: Mapping[str, Any], regime: str) -> dict[str, Any]:
    regimes = _mapping(registry.get("regimes"))
    current = _mapping(regimes.get(regime))
    if current:
        return current
    champions = _mapping(registry.get("champions"))
    current = _mapping(champions.get(regime))
    if current:
        return current
    return {}


def build_regime_champion_report(
    *,
    candidates: Mapping[str, Any],
    current_registry: Mapping[str, Any] | None = None,
    approval: Mapping[str, Any] | None = None,
    min_samples: int = 30,
    min_cost_adjusted_expectancy_bps: float = 0.0,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Return a conservative regime champion decision artifact."""

    registry = current_registry or {}
    approval_payload = approval or {}
    generated = generated_at.astimezone(UTC) if generated_at else datetime.now(UTC)
    fallback_model_id = str(
        candidates.get("conservative_fallback_model_id")
        or candidates.get("fallback_model_id")
        or "conservative_fallback"
    )
    decisions: list[dict[str, Any]] = []
    for candidate in _candidate_rows(candidates):
        regime = str(candidate.get("regime") or candidate.get("name") or "default").strip() or "default"
        model_id = str(candidate.get("model_id") or candidate.get("candidate_model_id") or "").strip()
        evidence = _mapping(candidate.get("evidence"))
        current = _current_for_regime(registry, regime)
        current_model_id = str(
            candidate.get("current_model_id")
            or current.get("model_id")
            or current.get("champion_model_id")
            or ""
        ).strip()
        current_authority = str(
            candidate.get("current_authority")
            or current.get("authority")
            or current.get("authority_level")
            or "shadow"
        )
        requested_authority = str(
            candidate.get("requested_authority")
            or candidate.get("proposed_authority")
            or candidate.get("authority")
            or "shadow"
        )
        samples = _sample_count(candidate, evidence)
        expectancy = _cost_adjusted_expectancy_bps(candidate, evidence)
        replay = _replay_gate(
            _evidence_payload(candidate, "replay", "replay_evidence"),
            min_expectancy_bps=min_cost_adjusted_expectancy_bps,
        )
        shadow = _shadow_gate(
            _evidence_payload(candidate, "shadow", "shadow_evidence", "shadow_report"),
            min_samples=min_samples,
        )
        authority_increase = _authority_increase(current_authority, requested_authority)
        manual_approval = (
            _approval_for_regime(approval_payload, regime, model_id)
            if authority_increase
            else {"approved": True, "reason": "authority_not_increased"}
        )
        reasons: list[str] = []
        if not model_id:
            reasons.append("candidate_model_id_missing")
        if samples < min_samples:
            reasons.append("insufficient_samples")
        if expectancy is None or expectancy < min_cost_adjusted_expectancy_bps:
            reasons.append("cost_adjusted_expectancy_too_low")
        if not bool(replay.get("ok")):
            reasons.append(str(replay.get("reason") or "replay_gate_failed"))
        if not bool(shadow.get("ok")):
            reasons.append(str(shadow.get("reason") or "shadow_gate_failed"))
        if authority_increase and not bool(manual_approval.get("approved")):
            reasons.append("manual_approval_required_for_authority_increase")

        approved = not reasons
        selected_model_id = model_id if approved else (current_model_id or fallback_model_id)
        decisions.append(
            {
                "regime": regime,
                "candidate_model_id": model_id or None,
                "selected_model_id": selected_model_id,
                "status": "champion_selected" if approved else "conservative_fallback",
                "reasons": reasons,
                "samples": samples,
                "cost_adjusted_expectancy_bps": expectancy,
                "current_model_id": current_model_id or None,
                "current_authority": current_authority,
                "requested_authority": requested_authority,
                "authority_increase": authority_increase,
                "effective_authority": requested_authority if approved else current_authority,
                "manual_approval": manual_approval,
                "replay_evidence": replay,
                "shadow_evidence": shadow,
                "fallback": {
                    "mode": "candidate" if approved else "conservative",
                    "model_id": selected_model_id,
                    "preferred_current_champion": bool(not approved and current_model_id),
                },
            }
        )
    blocked = [row for row in decisions if row["status"] != "champion_selected"]
    return {
        "schema_version": "1.0.0",
        "artifact_type": "regime_champion_models",
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "status": "ready" if decisions and not blocked else "blocked",
        "thresholds": {
            "min_samples": int(min_samples),
            "min_cost_adjusted_expectancy_bps": float(min_cost_adjusted_expectancy_bps),
        },
        "summary": {
            "regime_count": len(decisions),
            "selected_count": len(decisions) - len(blocked),
            "fallback_count": len(blocked),
            "manual_authority_increases_requested": sum(
                1 for row in decisions if bool(row["authority_increase"])
            ),
        },
        "decisions": decisions,
        "blocked_regimes": [str(row["regime"]) for row in blocked],
    }


def _default_output() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/regime_champion_models_latest.json",
        default_relative="runtime/regime_champion_models_latest.json",
        for_write=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-json", type=Path, required=True)
    parser.add_argument("--current-registry-json", type=Path, default=None)
    parser.add_argument("--approval-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-cost-adjusted-expectancy-bps", type=float, default=0.0)
    parser.add_argument("--success-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    report = build_regime_champion_report(
        candidates=_read_json_mapping(args.candidates_json),
        current_registry=_read_json_mapping(args.current_registry_json),
        approval=_read_json_mapping(args.approval_json),
        min_samples=max(0, int(args.min_samples)),
        min_cost_adjusted_expectancy_bps=float(args.min_cost_adjusted_expectancy_bps),
    )
    output = args.output_json or _default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output), "status": report["status"]}) + "\n")
    if args.success_on_blocked:
        return 0
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
