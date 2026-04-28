from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from ai_trading.governance.promotion import (
    ModelPromotion,
    PromotionCriteria,
    PromotionMetrics,
)
from ai_trading.model_registry import ModelRegistry


def _register_model(
    registry: ModelRegistry,
    *,
    strategy: str,
    marker: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    return cast(
        str,
        registry.register_model(
            model={"marker": marker},
            strategy=strategy,
            model_type="dict",
            metadata={"marker": marker, **dict(metadata or {})},
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


def test_promotion_eligibility_accepts_exact_threshold_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_POLICY_PROMOTION_MIN_OOS_SAMPLES", "20")
    monkeypatch.setenv("AI_TRADING_POLICY_PROMOTION_MIN_OOS_NET_BPS", "0")
    criteria = PromotionCriteria(min_shadow_days=0)
    promotion, registry = _promotion(tmp_path, monkeypatch, criteria=criteria)
    model_id = _register_model(registry, strategy="boundary", marker="candidate")
    registry.update_governance_status(
        model_id,
        "shadow",
        {"shadow_start_time": datetime.now(UTC).isoformat()},
    )
    promotion._save_shadow_metrics(
        model_id,
        PromotionMetrics(
            sessions_completed=criteria.min_shadow_sessions,
            total_trades=criteria.min_trade_count,
            turnover_ratio=criteria.max_turnover_ratio,
            live_sharpe_ratio=criteria.min_live_sharpe,
            max_drawdown=criteria.max_drawdown_threshold,
            drift_psi=criteria.max_drift_psi,
            live_sortino_ratio=criteria.min_live_sortino,
            live_calmar_ratio=criteria.min_live_calmar,
            tail_loss_95=criteria.max_tail_loss_95,
            risk_of_ruin=criteria.max_risk_of_ruin,
            purged_walk_forward_pass_ratio=criteria.min_purged_walk_forward_pass_ratio,
            monte_carlo_p05_bps=criteria.min_monte_carlo_p05_bps,
            regime_pass_ratio=criteria.min_regime_pass_ratio,
            tca_gate_passed=True,
            reject_rate=criteria.max_reject_rate,
            execution_drift_bps=criteria.max_execution_drift_bps,
            challenger_uplift_bps=criteria.min_challenger_uplift_bps,
            challenger_p_value=criteria.challenger_significance_alpha,
            gross_expectancy_bps=0.0,
            avg_cost_bps=0.0,
            net_expectancy_bps=criteria.min_net_expectancy_bps,
            live_calibration_ece=criteria.max_live_calibration_ece,
            live_calibration_brier=criteria.max_live_calibration_brier,
            calibration_samples=criteria.min_calibration_samples,
            challenger_eval_samples=20,
            challenger_sequential_passes=criteria.challenger_sequential_required_passes,
        ),
    )

    eligible, details = promotion.check_promotion_eligibility(model_id)

    assert eligible is True
    assert all(details["checks"].values())
    assert details["metrics"]["policy_min_oos_samples"] == 20
    assert details["criteria"]["min_sessions"] == criteria.min_shadow_sessions


def test_malformed_shadow_metrics_artifact_blocks_eligibility_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, registry = _promotion(tmp_path, monkeypatch)
    model_id = _register_model(registry, strategy="malformed_metrics", marker="candidate")
    (tmp_path / "governance" / f"{model_id}_shadow_metrics.json").write_text(
        "{not-json",
        encoding="utf-8",
    )

    assert promotion._load_shadow_metrics(model_id) is None
    eligible, details = promotion.check_promotion_eligibility(model_id)

    assert eligible is False
    assert details == {"error": "No shadow metrics found"}


def test_helper_env_clamps_and_disabled_event_store_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, _registry = _promotion(tmp_path, monkeypatch)

    monkeypatch.setenv("AI_TRADING_PROMOTION_APPROVAL_MAX_AGE_HOURS", "nan")
    assert promotion._promotion_approval_max_age_hours() == 168.0

    monkeypatch.setenv("AI_TRADING_PROMOTION_APPROVAL_MAX_AGE_HOURS", "0")
    assert promotion._promotion_approval_max_age_hours() == 1.0

    monkeypatch.setenv("AI_TRADING_PROMOTION_APPROVAL_MAX_AGE_HOURS", "100000")
    assert promotion._promotion_approval_max_age_hours() == 24.0 * 365.0

    monkeypatch.setenv("AI_TRADING_GOVERNANCE_EVENT_STORE_ENABLED", "0")
    assert promotion._governance_event_store_enabled() is False
    assert promotion._resolve_event_store() is None
    assert promotion._append_governance_audit_event(
        event_type="noop",
        payload={"ts": datetime.now(UTC)},
        primary_lineage={"model_hash": "hash"},
    ) is False
    assert ModelPromotion._json_default(object()).startswith("<object object")
    assert ModelPromotion._lineage_text("  policy-hash  ") == "policy-hash"
    assert ModelPromotion._lineage_text("  ") is None


def test_approval_with_unparseable_timestamp_blocks_approved_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PROMOTION_REQUIRE_APPROVAL", "1")
    promotion, registry = _promotion(tmp_path, monkeypatch)
    strategy = "approval_bad_ts"
    champion = _register_model(registry, strategy=strategy, marker="champion")
    challenger = _register_model(registry, strategy=strategy, marker="challenger")
    registry.update_governance_status(champion, "production")
    registry.update_governance_status(challenger, "shadow")
    monkeypatch.setattr(
        promotion,
        "check_promotion_eligibility",
        lambda _model_id: (True, {"eligible": True}),
    )
    (tmp_path / "governance" / "promotion_approvals.jsonl").write_text(
        json.dumps(
            {
                "approval_id": "bad-ts",
                "ts": "not-a-date",
                "strategy": strategy,
                "model_id": challenger,
                "approver": "ops@example.com",
                "decision": "approved",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert promotion.promote_to_production(challenger, force=False) is False
    assert registry.get_production_model(strategy)[0] == champion


def test_approval_with_missing_timestamp_blocks_approved_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PROMOTION_REQUIRE_APPROVAL", "1")
    promotion, registry = _promotion(tmp_path, monkeypatch)
    strategy = "approval_missing_ts"
    champion = _register_model(registry, strategy=strategy, marker="champion")
    challenger = _register_model(registry, strategy=strategy, marker="challenger")
    registry.update_governance_status(champion, "production")
    registry.update_governance_status(challenger, "shadow")
    monkeypatch.setattr(
        promotion,
        "check_promotion_eligibility",
        lambda _model_id: (True, {"eligible": True}),
    )
    (tmp_path / "governance" / "promotion_approvals.jsonl").write_text(
        json.dumps(
            {
                "approval_id": "missing-ts",
                "strategy": strategy,
                "model_id": challenger,
                "approver": "ops@example.com",
                "decision": "approved",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert promotion.promote_to_production(challenger, force=False) is False
    assert registry.get_production_model(strategy)[0] == champion


def test_active_model_path_and_shadow_listing_cover_helper_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    criteria = PromotionCriteria(min_shadow_days=0)
    promotion, registry = _promotion(tmp_path, monkeypatch, criteria=criteria)
    strategy = "helpers"
    model_id = _register_model(registry, strategy=strategy, marker="candidate")

    assert promotion.start_shadow_testing("missing-model") is False
    assert promotion.start_shadow_testing(model_id) is True
    shadow_models = promotion.list_shadow_models()

    assert len(shadow_models) == 1
    assert shadow_models[0]["model_id"] == model_id
    assert shadow_models[0]["strategy"] == strategy
    assert shadow_models[0]["promotion_eligible"] is False
    assert isinstance(shadow_models[0]["metrics"], PromotionMetrics)

    assert promotion.promote_to_production(model_id, force=True) is True
    active_path = promotion.get_active_model_path(strategy)
    assert active_path is not None
    assert Path(active_path).name == model_id

    promotion._remove_active_symlink(strategy)
    assert promotion.get_active_model_path(strategy) is None


def test_challenger_significance_filters_non_numeric_and_zero_variance_cases() -> None:
    insufficient = ModelPromotion.evaluate_challenger_significance(
        [0.01, "bad"],  # type: ignore[list-item]
        [0.0, None],  # type: ignore[list-item]
    )
    positive_zero_variance = ModelPromotion.evaluate_challenger_significance(
        [0.02, 0.02, 0.02],
        [0.01, 0.01, 0.01],
    )
    flat_zero_variance = ModelPromotion.evaluate_challenger_significance(
        [0.01, 0.01, 0.01],
        [0.01, 0.01, 0.01],
    )

    assert insufficient == {"uplift_bps": 0.0, "p_value": 1.0, "significant": False}
    assert positive_zero_variance["uplift_bps"] == pytest.approx(100.0)
    assert positive_zero_variance["p_value"] == 0.0
    assert positive_zero_variance["significant"] is True
    assert flat_zero_variance["uplift_bps"] == 0.0
    assert flat_zero_variance["p_value"] == 1.0
    assert flat_zero_variance["significant"] is False


def test_update_shadow_metrics_handles_invalid_session_payload_without_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, registry = _promotion(tmp_path, monkeypatch)
    monkeypatch.setenv("AI_TRADING_MODEL_GOVERNANCE_AUTO_VALIDATION_ENABLED", "0")
    model_id = _register_model(registry, strategy="bad_session", marker="candidate")

    promotion.update_shadow_metrics(
        model_id,
        {
            "trade_count": "bad",
            "returns": [0.01, 0.02],
            "avg_cost_bps": math.inf,
        },
    )

    assert promotion._load_shadow_metrics(model_id) is None


def test_jsonl_write_failure_and_scorecard_write_failure_return_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, registry = _promotion(tmp_path, monkeypatch)
    champion = _register_model(registry, strategy="write_fail", marker="champion")
    challenger = _register_model(registry, strategy="write_fail", marker="challenger")

    def fail_append_jsonl(
        *,
        filename: str,
        payload: dict[str, Any],
        error_event: str,
    ) -> str | None:
        assert filename == "promotion_approvals.jsonl"
        assert payload["decision"] == "approved"
        assert error_event == "PROMOTION_APPROVAL_WRITE_FAILED"
        return None

    monkeypatch.setattr(promotion, "_append_jsonl_event", fail_append_jsonl)
    assert promotion.record_promotion_approval(
        strategy="write_fail",
        model_id=challenger,
        approver="ops@example.com",
    ) is None

    blocker = tmp_path / "governance" / "challenger_evaluations.jsonl"
    blocker.mkdir(parents=True)
    assert promotion.record_challenger_evaluation(
        strategy="write_fail",
        champion_model_id=champion,
        challenger_model_id=challenger,
        metrics={"challenger_uplift_bps": 1.0},
    ) is None
