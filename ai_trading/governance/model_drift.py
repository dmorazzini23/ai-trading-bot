"""Canonical fail-closed interpretation of model-data drift evidence."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def evaluate_model_drift_gate(
    report: Mapping[str, Any] | None,
    *,
    required: bool,
    now: datetime | None = None,
    max_age_hours: float = 48.0,
) -> dict[str, Any]:
    """Normalize a drift monitor artifact without granting trading authority."""

    source = _mapping(report)
    evaluated_at = (now or datetime.now(UTC)).astimezone(UTC)
    bounded_max_age = max(
        0.0,
        float(max_age_hours) if math.isfinite(float(max_age_hours)) else 48.0,
    )
    status = str(source.get("status") or "missing").strip().lower() or "missing"
    reasons: list[str] = []
    if not source:
        reasons.append("model_data_drift_missing")
    elif str(source.get("artifact_type") or "") != "model_data_drift_monitor":
        reasons.append("model_data_drift_artifact_type_invalid")

    generated_at = _parse_timestamp(source.get("generated_at"))
    age_hours: float | None = None
    artifact_fresh = False
    if source:
        if generated_at is None:
            reasons.append("model_data_drift_generated_at_invalid")
        else:
            age_hours = (evaluated_at - generated_at).total_seconds() / 3600.0
            if age_hours < 0.0:
                reasons.append("model_data_drift_from_future")
            elif age_hours > bounded_max_age:
                reasons.append("model_data_drift_stale")
            else:
                artifact_fresh = True

    declared_reasons = [
        str(reason).strip()
        for reason in source.get("reasons", ())
        if str(reason).strip()
    ] if isinstance(source.get("reasons"), (list, tuple)) else []
    reasons.extend(reason for reason in declared_reasons if reason not in reasons)

    freshness = _mapping(source.get("freshness"))
    baseline_freshness = _mapping(freshness.get("baseline"))
    current_freshness = _mapping(freshness.get("current"))
    if source and not bool(baseline_freshness.get("fresh")):
        reasons.append("baseline_not_fresh")
    if source and not bool(current_freshness.get("fresh")):
        reasons.append("current_not_fresh")

    contract = _mapping(source.get("contract"))
    for coverage_name in ("baseline_coverage", "current_coverage"):
        coverage = _mapping(contract.get(coverage_name))
        if source and not bool(coverage.get("complete")):
            reasons.append(f"{coverage_name}_incomplete")
    baseline_model_id = str(contract.get("baseline_model_id") or "").strip() or None
    current_model_id = str(contract.get("current_model_id") or "").strip() or None
    baseline_model_hash = str(contract.get("baseline_model_hash") or "").strip() or None
    current_model_hash = str(contract.get("current_model_hash") or "").strip() or None
    if baseline_model_id and current_model_id and baseline_model_id != current_model_id:
        reasons.append("model_id_mismatch")
    if baseline_model_hash and current_model_hash and baseline_model_hash != current_model_hash:
        reasons.append("model_hash_mismatch")

    summary = dict(_mapping(source.get("summary")))
    drift_categories = [
        str(category)
        for category in summary.get("drift_categories", ())
        if str(category).strip()
    ] if isinstance(summary.get("drift_categories"), (list, tuple)) else []
    if status == "drift_detected" or drift_categories:
        reasons.append("model_data_drift_detected")
    if status not in {"ok", "drift_detected"} and source:
        reasons.append("model_data_drift_blocked")

    unique_reasons = list(dict.fromkeys(reasons))
    evidence_clean = bool(source) and status == "ok" and artifact_fresh and not unique_reasons
    gate_passed = bool(evidence_clean or not required)
    normalized_status = status if source else ("blocked" if required else "not_required")
    return {
        "required": bool(required),
        "gate_passed": gate_passed,
        "evidence_clean": evidence_clean,
        "status": normalized_status,
        "reasons": unique_reasons,
        "freshness": {
            "artifact_fresh": artifact_fresh,
            "generated_at": generated_at.isoformat() if generated_at is not None else None,
            "age_hours": age_hours,
            "max_age_hours": bounded_max_age,
            "baseline": dict(baseline_freshness),
            "current": dict(current_freshness),
        },
        "identity": {
            "baseline_model_id": baseline_model_id,
            "current_model_id": current_model_id,
            "baseline_model_hash": baseline_model_hash,
            "current_model_hash": current_model_hash,
        },
        "summary": summary,
        "authority_effect": "restrict_only",
    }


__all__ = ["evaluate_model_drift_gate"]
