from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.governance.model_drift import evaluate_model_drift_gate


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


def _clean_report() -> dict[str, object]:
    coverage = {"complete": True, "missing_categories": []}
    return {
        "artifact_type": "model_data_drift_monitor",
        "generated_at": (NOW - timedelta(hours=1)).isoformat(),
        "status": "ok",
        "reasons": [],
        "freshness": {
            "baseline": {"fresh": True},
            "current": {"fresh": True},
        },
        "contract": {
            "baseline_model_id": "model-1",
            "current_model_id": "model-1",
            "baseline_model_hash": "hash-1",
            "current_model_hash": "hash-1",
            "baseline_coverage": coverage,
            "current_coverage": coverage,
        },
        "summary": {
            "drift_categories": [],
            "drift_count": 0,
        },
    }


def test_required_clean_drift_evidence_passes_with_identity() -> None:
    gate = evaluate_model_drift_gate(_clean_report(), required=True, now=NOW)

    assert gate["gate_passed"] is True
    assert gate["evidence_clean"] is True
    assert gate["reasons"] == []
    assert gate["identity"]["current_model_id"] == "model-1"
    assert gate["authority_effect"] == "restrict_only"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("stale", "model_data_drift_stale"),
        ("drift", "model_data_drift_detected"),
        ("mismatch", "model_id_mismatch"),
        ("incomplete", "current_coverage_incomplete"),
        ("unapproved", "baseline_unapproved"),
        ("blocked", "model_data_drift_blocked"),
    ],
)
def test_required_drift_evidence_fails_closed(
    mutation: str,
    expected_reason: str,
) -> None:
    report = deepcopy(_clean_report())
    if mutation == "stale":
        report["generated_at"] = (NOW - timedelta(hours=49)).isoformat()
    elif mutation == "drift":
        report["status"] = "drift_detected"
        report["summary"]["drift_categories"] = ["feature"]  # type: ignore[index]
    elif mutation == "mismatch":
        report["contract"]["current_model_id"] = "model-2"  # type: ignore[index]
    elif mutation == "incomplete":
        report["contract"]["current_coverage"] = {"complete": False}  # type: ignore[index]
    elif mutation == "unapproved":
        report["status"] = "blocked"
        report["reasons"] = ["baseline_unapproved"]
    elif mutation == "blocked":
        report["status"] = "blocked"

    gate = evaluate_model_drift_gate(report, required=True, now=NOW)

    assert gate["gate_passed"] is False
    assert gate["evidence_clean"] is False
    assert expected_reason in gate["reasons"]


def test_missing_optional_evidence_preserves_compatibility_without_claiming_clean() -> None:
    gate = evaluate_model_drift_gate(None, required=False, now=NOW)

    assert gate["gate_passed"] is True
    assert gate["evidence_clean"] is False
    assert gate["status"] == "not_required"
    assert gate["reasons"] == ["model_data_drift_missing"]


def test_missing_required_evidence_fails_closed() -> None:
    gate = evaluate_model_drift_gate({}, required=True, now=NOW)

    assert gate["gate_passed"] is False
    assert gate["status"] == "blocked"
    assert gate["reasons"] == ["model_data_drift_missing"]
