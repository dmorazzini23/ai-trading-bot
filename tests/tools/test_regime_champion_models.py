from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.tools import regime_champion_models


def _clean_drift_report(
    generated_at: datetime = datetime(2026, 5, 5, 19, 0, tzinfo=UTC),
) -> dict[str, object]:
    coverage = {"complete": True}
    return {
        "artifact_type": "model_data_drift_monitor",
        "generated_at": generated_at.isoformat(),
        "status": "ok",
        "reasons": [],
        "freshness": {
            "baseline": {"fresh": True},
            "current": {"fresh": True},
        },
        "contract": {
            "baseline_model_id": "trend-v2",
            "current_model_id": "trend-v2",
            "baseline_model_hash": "hash-v2",
            "current_model_hash": "hash-v2",
            "baseline_coverage": coverage,
            "current_coverage": coverage,
        },
        "summary": {"drift_categories": [], "drift_count": 0},
    }


def _candidate_payload() -> dict[str, object]:
    return {
        "regimes": [
            {
                "regime": "trend_up",
                "model_id": "trend-v2",
                "sample_count": 80,
                "gross_expectancy_bps": 9.5,
                "avg_cost_bps": 2.0,
                "current_authority": "shadow",
                "requested_authority": "live_canary",
                "replay": {
                    "aggregate": {
                        "expectancy_bps": 7.5,
                        "total_trades": 44,
                        "violation_count": 0,
                    }
                },
                "shadow": {
                    "sample_gate": {"status": "review_eligible"},
                    "decision_summary": {"rows": 80},
                    "markout_summary": {"challenger_mean_net_markout_bps": 3.1},
                },
            }
        ],
        "fallback_model_id": "global-safe",
    }


def test_regime_champion_selects_candidate_with_approval_and_evidence() -> None:
    report = regime_champion_models.build_regime_champion_report(
        candidates=_candidate_payload(),
        current_registry={"regimes": {"trend_up": {"model_id": "trend-v1", "authority": "shadow"}}},
        approval={"approved_regimes": ["trend_up"]},
        drift_report=_clean_drift_report(),
        min_samples=50,
        min_cost_adjusted_expectancy_bps=1.0,
        generated_at=datetime(2026, 5, 5, 20, 0, tzinfo=UTC),
    )

    assert report["status"] == "ready"
    decision = report["decisions"][0]
    assert decision["status"] == "champion_selected"
    assert decision["selected_model_id"] == "trend-v2"
    assert decision["cost_adjusted_expectancy_bps"] == 7.5
    assert decision["manual_approval"]["approved"] is True
    assert decision["effective_authority"] == "live_canary"
    assert decision["model_data_drift"]["gate_passed"] is True
    assert decision["model_data_drift"]["identity"]["current_model_id"] == "trend-v2"


def test_regime_champion_blocks_authority_increase_and_falls_back_to_current() -> None:
    payload = _candidate_payload()
    payload["regimes"][0]["sample_count"] = 10  # type: ignore[index]
    payload["regimes"][0]["gross_expectancy_bps"] = 1.0  # type: ignore[index]

    report = regime_champion_models.build_regime_champion_report(
        candidates=payload,
        current_registry={"regimes": {"trend_up": {"model_id": "trend-v1", "authority": "shadow"}}},
        approval={},
        min_samples=50,
        min_cost_adjusted_expectancy_bps=1.0,
    )

    decision = report["decisions"][0]
    assert report["status"] == "blocked"
    assert decision["status"] == "conservative_fallback"
    assert decision["selected_model_id"] == "trend-v1"
    assert "insufficient_samples" in decision["reasons"]
    assert "cost_adjusted_expectancy_too_low" in decision["reasons"]
    assert "manual_approval_required_for_authority_increase" in decision["reasons"]
    assert "model_data_drift_gate_failed" in decision["reasons"]
    assert decision["fallback"]["preferred_current_champion"] is True


def test_regime_champion_retains_current_when_required_drift_is_blocked() -> None:
    report = regime_champion_models.build_regime_champion_report(
        candidates=_candidate_payload(),
        current_registry={
            "regimes": {
                "trend_up": {"model_id": "trend-v1", "authority": "shadow"}
            }
        },
        approval={"approved_regimes": ["trend_up"]},
        drift_report={
            **_clean_drift_report(),
            "status": "drift_detected",
            "summary": {"drift_categories": ["regime"], "drift_count": 1},
        },
        min_samples=50,
        min_cost_adjusted_expectancy_bps=1.0,
        generated_at=datetime(2026, 5, 5, 20, 0, tzinfo=UTC),
    )

    decision = report["decisions"][0]
    assert report["status"] == "blocked"
    assert decision["selected_model_id"] == "trend-v1"
    assert decision["effective_authority"] == "shadow"
    assert decision["model_data_drift"]["gate_passed"] is False
    assert "model_data_drift_detected" in decision["model_data_drift"]["reasons"]
    assert "model_data_drift_gate_failed" in decision["reasons"]
    assert report["summary"]["drift_blocked_authority_increases"] == 1


def test_shadow_observation_continues_without_drift_authority() -> None:
    payload = _candidate_payload()
    candidate = payload["regimes"][0]  # type: ignore[index]
    candidate["requested_authority"] = "shadow"  # type: ignore[index]

    report = regime_champion_models.build_regime_champion_report(
        candidates=payload,
        current_registry={
            "regimes": {
                "trend_up": {"model_id": "trend-v1", "authority": "shadow"}
            }
        },
        approval={},
        drift_report=None,
        min_samples=50,
        min_cost_adjusted_expectancy_bps=1.0,
    )

    decision = report["decisions"][0]
    assert decision["status"] == "champion_selected"
    assert decision["effective_authority"] == "shadow"
    assert decision["model_data_drift"]["required"] is False
    assert decision["model_data_drift"]["evidence_clean"] is False
    assert "model_data_drift_missing" in decision["model_data_drift"]["reasons"]


def test_regime_champion_cli_writes_artifact_and_returns_blocked(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.json"
    output = tmp_path / "regime_champions.json"
    payload = _candidate_payload()
    payload["regimes"][0]["shadow"] = {}  # type: ignore[index]
    candidates.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = regime_champion_models.main(
        [
            "--candidates-json",
            str(candidates),
            "--output-json",
            str(output),
            "--min-samples",
            "50",
        ]
    )

    assert exit_code == 2
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "regime_champion_models"
    assert artifact["status"] == "blocked"
    assert artifact["decisions"][0]["selected_model_id"] == "global-safe"
