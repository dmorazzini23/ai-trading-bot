from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from ai_trading.governance import promotion as promotion_mod
from ai_trading.governance.promotion import ModelPromotion, PromotionCriteria
from ai_trading.model_registry import ModelRegistry


def _register_model(registry: ModelRegistry, *, strategy: str, marker: str) -> str:
    return cast(
        str,
        registry.register_model(
            model={"marker": marker},
            strategy=strategy,
            model_type="dict",
            metadata={
                "marker": marker,
                "dataset_hash": f"dataset-{marker}",
                "feature_version": "features-v1",
                "model_artifact_hash": f"artifact-{marker}",
                "policy_hash": "policy-v1",
                "config_snapshot_hash": "config-v1",
            },
        ),
    )


def _promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    criteria: PromotionCriteria | None = None,
) -> tuple[ModelPromotion, ModelRegistry]:
    monkeypatch.setenv("AI_TRADING_GOVERNANCE_EVENT_STORE_ENABLED", "0")
    registry = ModelRegistry(tmp_path / "registry")
    return (
        ModelPromotion(
            model_registry=registry,
            criteria=criteria,
            base_path=str(tmp_path / "governance"),
        ),
        registry,
    )


def test_jsonl_tail_and_latest_approval_ignore_malformed_or_mismatched_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, _registry = _promotion(tmp_path, monkeypatch)
    approvals = tmp_path / "governance" / "promotion_approvals.jsonl"
    approvals.write_text(
        "\n".join(
            [
                "",
                "{not-json",
                json.dumps(["not", "a", "dict"]),
                json.dumps(
                    {
                        "strategy": "other",
                        "model_id": "model-a",
                        "decision": "approved",
                    }
                ),
                json.dumps(
                    {
                        "strategy": "edge",
                        "model_id": "wrong-model",
                        "decision": "approved",
                    }
                ),
                json.dumps(
                    {
                        "approval_id": "fresh",
                        "strategy": "edge",
                        "model_id": "model-a",
                        "decision": "approved",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    recent = promotion.list_recent_promotion_approvals(limit=10)
    latest = promotion._latest_promotion_approval(strategy="edge", model_id="model-a")

    assert [row.get("decision") for row in recent] == ["approved", "approved", "approved"]
    assert latest is not None
    assert latest["approval_id"] == "fresh"
    assert promotion._latest_promotion_approval(strategy="missing", model_id="model-a") is None


def test_governance_audit_event_uses_related_lineage_and_handles_store_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, _registry = _promotion(tmp_path, monkeypatch)
    monkeypatch.setenv("AI_TRADING_GOVERNANCE_EVENT_STORE_ENABLED", "1")
    appended: list[dict[str, Any]] = []

    class Store:
        def append_oms_event_payload(self, **kwargs: Any) -> bool:
            appended.append(kwargs)
            return True

    promotion._event_store = Store()
    assert promotion._append_governance_audit_event(
        event_type="model_promoted",
        payload={"ts": "2026-04-27T00:00:00+00:00", "x": datetime(2026, 4, 27, tzinfo=UTC)},
        related_lineages={
            "empty": {},
            "champion": {"model_hash": "hash-a", "policy_hash": "policy-a"},
        },
    )

    assert appended[0]["event_type"] == "MODEL_PROMOTED"
    assert appended[0]["policy_hash"] == "policy-a"
    assert appended[0]["model_hash"] == "hash-a"
    assert "champion" in appended[0]["payload"]["related_lineages"]
    assert "empty" not in appended[0]["payload"]["related_lineages"]

    class FailingStore:
        def append_oms_event_payload(self, **_kwargs: Any) -> bool:
            raise ValueError("event store unavailable")

    promotion._event_store = FailingStore()
    assert promotion._append_governance_audit_event(
        event_type="",
        payload={"ts": "2026-04-27T00:00:00+00:00"},
        primary_lineage={"model_artifact_hash": "artifact-only"},
    ) is False


def test_live_kpi_state_malformed_inputs_and_relative_path_are_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, _registry = _promotion(tmp_path, monkeypatch)
    monkeypatch.setenv("AI_TRADING_PROMOTION_LIVE_KPI_BREACH_STATE_PATH", "nested/state.json")
    monkeypatch.setenv("AI_TRADING_PROMOTION_LIVE_KPI_BREACH_CONSECUTIVE_REQUIRED", "999")
    state_path = tmp_path / "governance" / "nested" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{bad-json", encoding="utf-8")

    assert promotion._live_kpi_breach_state_path() == state_path
    assert promotion._load_live_kpi_breach_state() == {"version": 1, "strategies": {}}
    assert promotion._required_live_kpi_breach_count() == 30

    state_path.write_text(json.dumps({"strategies": ["bad"]}), encoding="utf-8")
    loaded = promotion._load_live_kpi_breach_state()
    assert loaded == {"strategies": {}, "version": 1}


def test_live_kpi_pending_dry_run_demotion_and_successful_rollback_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    criteria = PromotionCriteria(min_shadow_days=0)
    promotion, registry = _promotion(tmp_path, monkeypatch, criteria=criteria)
    monkeypatch.setenv("AI_TRADING_PROMOTION_LIVE_KPI_BREACH_CONSECUTIVE_REQUIRED", "2")
    champion = _register_model(registry, strategy="kpi_edges", marker="champion")
    challenger = _register_model(registry, strategy="kpi_edges", marker="challenger")
    registry.update_governance_status(champion, "production")
    assert promotion.promote_to_production(challenger, force=True) is True

    breached = {"max_drawdown": 0.10}
    first = promotion.evaluate_live_kpis_and_maybe_rollback(
        strategy="kpi_edges",
        live_kpis=breached,
    )
    assert first["status"] == "pending"
    assert first["consecutive_breach_count"] == 1

    dry_run = promotion.evaluate_live_kpis_and_maybe_rollback(
        strategy="kpi_edges",
        live_kpis=breached,
    )
    assert dry_run["status"] == "dry_run_disabled"
    assert dry_run["triggered"] is False

    monkeypatch.setenv("AI_TRADING_PROMOTION_AUTO_ROLLBACK_ON_CONTROL_BAND", "1")
    rolled_back = promotion.evaluate_live_kpis_and_maybe_rollback(
        strategy="kpi_edges",
        live_kpis=breached,
    )
    assert rolled_back["status"] == "rolled_back"
    assert rolled_back["triggered"] is True
    assert registry.get_production_model("kpi_edges")[0] == champion

    no_target = _register_model(registry, strategy="kpi_no_target", marker="solo")
    registry.update_governance_status(no_target, "production")
    monkeypatch.setenv("AI_TRADING_PROMOTION_LIVE_KPI_BREACH_CONSECUTIVE_REQUIRED", "1")
    demoted = promotion.evaluate_live_kpis_and_maybe_rollback(
        strategy="kpi_no_target",
        live_kpis=breached,
    )
    assert demoted["status"] == "demoted_no_rollback_target"
    assert demoted["triggered"] is True
    assert registry.model_index[no_target]["governance"]["status"] == "challenger"


def test_rollback_skip_branches_record_audit_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, registry = _promotion(tmp_path, monkeypatch)
    solo = _register_model(registry, strategy="rollback_solo", marker="solo")
    missing_previous = _register_model(registry, strategy="rollback_missing", marker="current")
    registry.update_governance_status(solo, "production")
    registry.update_governance_status(
        missing_previous,
        "production",
        {"previous_production_model_id": "missing-model"},
    )

    assert promotion.rollback_to_previous_production(strategy="none", reason="test") is False
    assert promotion.rollback_to_previous_production(
        strategy="rollback_solo",
        reason="test",
    ) is False
    assert promotion.rollback_to_previous_production(
        strategy="rollback_missing",
        reason="test",
    ) is False

    statuses = [row["status"] for row in promotion.list_recent_rollback_audit(limit=5)]
    assert statuses == [
        "skipped_no_production_model",
        "skipped_no_previous_model",
        "skipped_previous_model_missing",
    ]


def test_challenger_evaluation_scorecards_cover_returns_and_summary_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, registry = _promotion(tmp_path, monkeypatch)
    champion = _register_model(registry, strategy="eval_edges", marker="champion")
    challenger = _register_model(registry, strategy="eval_edges", marker="challenger")

    path = promotion.record_challenger_evaluation(
        strategy="eval_edges",
        champion_model_id=champion,
        challenger_model_id=challenger,
        metrics={
            "challenger_returns": [0.002] * 25,
            "champion_returns": [0.001] * 25,
            "champion_net_expectancy_bps": 1.0,
            "challenger_net_expectancy_bps": 3.5,
            "champion_reject_rate": 0.01,
            "challenger_reject_rate": 0.015,
            "champion_execution_drift_bps": 5.0,
            "challenger_execution_drift_bps": 7.5,
        },
    )
    assert path is not None

    summary_path = promotion.record_challenger_evaluation(
        strategy="eval_edges",
        champion_model_id=champion,
        challenger_model_id=challenger,
        metrics={"challenger_uplift_bps": 2.0, "challenger_p_value": 0.25},
    )
    assert summary_path == path

    scorecards = promotion.list_recent_champion_challenger_scorecards(limit=2)
    assert scorecards[0]["session_pass"] is True
    assert scorecards[0]["delta_net_expectancy_bps"] == 2.5
    assert scorecards[1]["uplift_bps"] == 2.0
    assert scorecards[1]["p_value"] == 0.25


def test_shadow_metric_update_handles_returns_costs_calibration_and_challenger_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    criteria = PromotionCriteria(
        challenger_sequential_min_samples=3,
        challenger_sequential_required_passes=1,
        min_challenger_uplift_bps=1.0,
    )
    promotion, registry = _promotion(tmp_path, monkeypatch, criteria=criteria)
    monkeypatch.setenv("AI_TRADING_MODEL_GOVERNANCE_AUTO_VALIDATION_ENABLED", "0")
    model_id = _register_model(registry, strategy="shadow_update", marker="candidate")
    assert promotion.start_shadow_testing(model_id) is True

    promotion.update_shadow_metrics(
        model_id,
        {
            "trade_count": 3,
            "turnover_ratio": 1.0,
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.01,
            "drift_psi": 0.02,
            "avg_latency_ms": 12.0,
            "error_rate": 0.0,
            "returns": [0.003, 0.002, 0.001, -0.0005],
            "avg_cost_bps": math.inf,
            "sortino_ratio": 1.2,
            "calmar_ratio": 0.8,
            "tail_loss_95": 0.002,
            "risk_of_ruin": 0.01,
            "purged_walk_forward_pass_ratio": 0.8,
            "regime_pass_ratio": 0.9,
            "monte_carlo_p05_bps": -5.0,
            "tca_gate_passed": False,
            "reject_rate": 0.02,
            "execution_drift_bps": 3.0,
            "live_calibration_ece": 0.05,
            "live_calibration_brier": 0.08,
            "calibration_samples": 100,
            "challenger_returns": [0.01, 0.01, 0.01],
            "champion_returns": [0.001, 0.001, 0.001],
        },
    )

    metrics = promotion._load_shadow_metrics(model_id)
    assert metrics is not None
    assert metrics.sessions_completed == 1
    assert metrics.total_trades == 3
    assert metrics.avg_cost_bps == 0.0
    assert metrics.net_expectancy_bps > 0.0
    assert metrics.tca_gate_passed is False
    assert metrics.challenger_eval_samples == 3
    assert metrics.challenger_sequential_passes == 1


def test_institutional_validation_derives_frame_and_calibration_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, _registry = _promotion(tmp_path, monkeypatch)
    monkeypatch.setattr(
        promotion_mod,
        "run_purged_walk_forward_validation",
        lambda *_args, **_kwargs: {"pass_ratio": 0.75, "folds": []},
    )
    monkeypatch.setattr(
        promotion_mod,
        "run_regime_split_validation",
        lambda *_args, **_kwargs: {"pass_ratio": 0.5, "regimes": []},
    )
    monkeypatch.setenv("AI_TRADING_MODEL_GOVERNANCE_PWF_SPLITS", "2")
    monkeypatch.setenv("AI_TRADING_MODEL_GOVERNANCE_REGIME_MIN_SAMPLES", "5")

    derived = promotion._derive_institutional_validation_metrics(
        {
            "returns": [0.01, "bad", 0.02, float("nan"), -0.01],
            "regimes": ["bull", "", "bear"],
            "validation_frame": [
                {"prob": 1.2, "label": 1, "return": 0.01},
                {"prob": -0.2, "label": 0, "return": 0.02},
                {"prob": "bad", "label": 1, "return": -0.01},
            ],
        }
    )

    assert derived["purged_walk_forward_pass_ratio"] == 0.75
    assert derived["regime_pass_ratio"] == 0.5
    assert derived["calibration_samples"] == 2
    assert derived["live_calibration_brier"] == 0.0
    assert derived["live_calibration_ece"] == 0.0

    assert promotion._derive_institutional_validation_metrics({"returns": []}) == {}
    assert ModelPromotion._coerce_returns("not-a-sequence") == []
