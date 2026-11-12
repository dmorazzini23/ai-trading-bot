from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from ai_trading.governance.promotion import (
    ModelPromotion,
    PromotionCriteria,
    PromotionMetrics,
)
from ai_trading.model_registry import ModelRegistry


def _register_test_model(
    registry: ModelRegistry,
    *,
    strategy: str,
    marker: str,
) -> str:
    return cast(
        str,
        registry.register_model(
            model={"marker": marker},
            strategy=strategy,
            model_type="dict",
            metadata={"marker": marker},
        ),
    )


def _promotion_with_challenger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    strategy: str,
) -> tuple[ModelPromotion, ModelRegistry, str, str, list[dict[str, Any]]]:
    monkeypatch.setenv("AI_TRADING_GOVERNANCE_EVENT_STORE_ENABLED", "0")
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(
        model_registry=registry,
        base_path=str(tmp_path / "governance"),
    )
    champion = _register_test_model(registry, strategy=strategy, marker="champion")
    challenger = _register_test_model(registry, strategy=strategy, marker="challenger")
    registry.update_governance_status(champion, "production")
    registry.update_governance_status(challenger, "shadow")
    monkeypatch.setattr(
        promotion,
        "check_promotion_eligibility",
        lambda _model_id: (True, {"eligible": True}),
    )
    audit_events: list[dict[str, Any]] = []

    def record_audit(**kwargs: Any) -> bool:
        audit_events.append(kwargs)
        return True

    monkeypatch.setattr(promotion, "_append_governance_audit_event", record_audit)
    return promotion, registry, champion, challenger, audit_events


def test_promotion_approval_gate_classifies_rejected_and_invalid_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PROMOTION_REQUIRE_APPROVAL", "1")
    promotion, registry, champion, challenger, audit_events = _promotion_with_challenger(
        tmp_path,
        monkeypatch,
        strategy="approval_rejected",
    )

    with pytest.raises(ValueError, match="decision must be"):
        promotion.record_promotion_approval(
            strategy="approval_rejected",
            model_id=challenger,
            approver="ops@example.com",
            decision="maybe",
        )

    approval_path = promotion.record_promotion_approval(
        strategy="approval_rejected",
        model_id=challenger,
        approver="ops@example.com",
        decision="rejected",
        ticket="GOV-REJECT",
    )

    assert approval_path is not None
    assert promotion.promote_to_production(challenger, force=False) is False
    assert registry.get_production_model("approval_rejected")[0] == champion
    blocked = audit_events[-1]["payload"]
    assert blocked["reason"] == "approval_rejected"
    assert blocked["approval"]["decision"] == "rejected"


def test_promote_rolls_back_governance_when_active_symlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = "symlink_failure"
    promotion, registry, champion, challenger, audit_events = _promotion_with_challenger(
        tmp_path,
        monkeypatch,
        strategy=strategy,
    )

    def fail_symlink(_strategy: str, _model_id: str) -> None:
        raise OSError("cannot create active link")

    monkeypatch.setattr(promotion, "_create_active_symlink", fail_symlink)

    assert promotion.promote_to_production(challenger, force=True) is False

    production = registry.get_production_model(strategy)
    assert production is not None
    assert production[0] == champion
    assert registry.model_index[champion]["governance"]["status"] == "production"
    assert registry.model_index[challenger]["governance"]["status"] == "shadow"
    assert audit_events[-1]["event_type"] == "MODEL_PROMOTION_BLOCKED"
    assert audit_events[-1]["payload"]["reason"] == "active_symlink_failed"


def test_promotion_approval_gate_classifies_stale_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PROMOTION_REQUIRE_APPROVAL", "1")
    monkeypatch.setenv("AI_TRADING_PROMOTION_APPROVAL_MAX_AGE_HOURS", "1")
    promotion, registry, champion, challenger, audit_events = _promotion_with_challenger(
        tmp_path,
        monkeypatch,
        strategy="approval_stale",
    )
    stale_ts = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    approval = {
        "approval_id": "stale-approval",
        "ts": stale_ts,
        "strategy": "approval_stale",
        "model_id": challenger,
        "approver": "ops@example.com",
        "decision": "approved",
    }
    (tmp_path / "governance" / "promotion_approvals.jsonl").write_text(
        json.dumps(approval) + "\n",
        encoding="utf-8",
    )

    assert promotion.promote_to_production(challenger, force=False) is False
    assert registry.get_production_model("approval_stale")[0] == champion
    blocked = audit_events[-1]["payload"]
    assert blocked["reason"] == "approval_stale"
    assert blocked["approval"]["approval_id"] == "stale-approval"
    assert blocked["age_hours"] > blocked["max_age_hours"]


def test_promotion_approval_gate_blocks_future_checkpoint_beyond_skew(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PROMOTION_REQUIRE_APPROVAL", "1")
    monkeypatch.setenv("AI_TRADING_PROMOTION_APPROVAL_FUTURE_SKEW_SECONDS", "30")
    promotion, registry, champion, challenger, audit_events = _promotion_with_challenger(
        tmp_path,
        monkeypatch,
        strategy="approval_future",
    )
    approval = {
        "approval_id": "future-approval",
        "ts": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        "strategy": "approval_future",
        "model_id": challenger,
        "approver": "ops@example.com",
        "decision": "approved",
    }
    (tmp_path / "governance" / "promotion_approvals.jsonl").write_text(
        json.dumps(approval) + "\n",
        encoding="utf-8",
    )

    assert promotion.promote_to_production(challenger, force=False) is False
    assert registry.get_production_model("approval_future")[0] == champion
    blocked = audit_events[-1]["payload"]
    assert blocked["reason"] == "approval_timestamp_future"
    assert blocked["future_seconds"] > blocked["future_skew_seconds"]


def test_promotion_approval_gate_blocks_missing_then_promotes_fresh_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PROMOTION_REQUIRE_APPROVAL", "1")
    promotion, registry, champion, challenger, audit_events = _promotion_with_challenger(
        tmp_path,
        monkeypatch,
        strategy="approval_fresh",
    )

    assert promotion.promote_to_production(challenger, force=False) is False
    assert audit_events[-1]["payload"]["reason"] == "approval_missing"
    assert registry.get_production_model("approval_fresh")[0] == champion

    approval_path = promotion.record_promotion_approval(
        strategy="approval_fresh",
        model_id=challenger,
        approver="ops@example.com",
        decision="approved",
        note="fresh review",
    )

    assert approval_path is not None
    assert promotion.promote_to_production(challenger, force=False) is True
    governance = registry.model_index[challenger]["governance"]
    assert governance["status"] == "production"
    assert governance["promotion_approved_by"] == "ops@example.com"
    assert governance["promotion_approval_ts"]
    assert registry.model_index[champion]["governance"]["status"] == "challenger"


def test_promote_restores_active_pointer_when_registry_status_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = "registry_failure"
    promotion, registry, champion, challenger, audit_events = _promotion_with_challenger(
        tmp_path,
        monkeypatch,
        strategy=strategy,
    )
    promotion._create_active_symlink(strategy, champion)
    champion_active_path = promotion.get_active_model_path(strategy)
    assert champion_active_path is not None
    real_update = registry.update_governance_status

    def fail_challenger_production(
        model_id: str,
        status: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if model_id == challenger and status == "production":
            raise OSError("registry unavailable")
        real_update(model_id, status, extra)

    monkeypatch.setattr(registry, "update_governance_status", fail_challenger_production)

    assert promotion.promote_to_production(challenger, force=True) is False
    assert promotion.get_active_model_path(strategy) == champion_active_path
    assert registry.model_index[champion]["governance"]["status"] == "production"
    assert registry.model_index[challenger]["governance"]["status"] == "shadow"
    assert audit_events[-1]["payload"]["reason"] == "registry_status_failed"


def test_promotion_eligibility_reports_threshold_gate_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_GOVERNANCE_EVENT_STORE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_POLICY_PROMOTION_MIN_OOS_SAMPLES", "20")
    monkeypatch.setenv("AI_TRADING_POLICY_PROMOTION_MIN_OOS_NET_BPS", "0.0")
    registry = ModelRegistry(tmp_path / "registry")
    criteria = PromotionCriteria(min_shadow_days=0)
    promotion = ModelPromotion(
        model_registry=registry,
        criteria=criteria,
        base_path=str(tmp_path / "governance"),
    )
    model_id = _register_test_model(registry, strategy="thresholds", marker="candidate")
    registry.update_governance_status(
        model_id,
        "shadow",
        {"shadow_start_time": (datetime.now(UTC) - timedelta(days=5)).isoformat()},
    )
    promotion._save_shadow_metrics(
        model_id,
        PromotionMetrics(
            sessions_completed=5,
            total_trades=10,
            turnover_ratio=1.0,
            live_sharpe_ratio=0.6,
            max_drawdown=0.01,
            drift_psi=0.1,
            live_sortino_ratio=0.6,
            live_calmar_ratio=0.5,
            tail_loss_95=0.01,
            risk_of_ruin=0.1,
            purged_walk_forward_pass_ratio=0.7,
            monte_carlo_p05_bps=-10.0,
            regime_pass_ratio=0.7,
            tca_gate_passed=True,
            reject_rate=0.031,
            execution_drift_bps=10.0,
            challenger_uplift_bps=1.0,
            challenger_p_value=0.01,
            net_expectancy_bps=1.0,
            live_calibration_ece=0.05,
            live_calibration_brier=0.31,
            calibration_samples=50,
            challenger_eval_samples=20,
            challenger_sequential_passes=3,
            last_updated=datetime.now(UTC),
        ),
    )

    eligible, details = promotion.check_promotion_eligibility(model_id)

    assert eligible is False
    assert details["checks"]["reject_rate_check"] is False
    assert details["checks"]["calibration_ece_check"] is True
    assert details["checks"]["calibration_brier_check"] is False
    assert details["metrics"]["reject_rate"] == 0.031
    assert details["criteria"]["max_live_calibration_brier"] == 0.3


def test_live_kpi_rollout_classifies_all_control_band_breaches_and_pending_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, registry, _champion, challenger, _audit_events = _promotion_with_challenger(
        tmp_path,
        monkeypatch,
        strategy="rollout_pending",
    )
    assert promotion.promote_to_production(challenger, force=True) is True

    result = promotion.evaluate_live_kpis_and_maybe_rollback(
        strategy="rollout_pending",
        live_kpis={
            "max_drawdown": 0.081,
            "reject_rate": 0.051,
            "execution_drift_bps": 35.1,
            "drift_psi": 0.301,
            "live_calibration_ece": 0.151,
            "live_calibration_brier": 0.351,
        },
        allow_rollback=False,
    )

    assert result["status"] == "pending"
    assert result["triggered"] is False
    assert result["model_id"] == challenger
    assert result["failed_kpis"] == [
        "drift_psi",
        "execution_drift_bps",
        "live_calibration_brier",
        "live_calibration_ece",
        "max_drawdown",
        "reject_rate",
    ]
    assert registry.get_production_model("rollout_pending")[0] == challenger


def test_live_kpi_rollout_clears_prior_window_when_state_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion, _registry, champion, _challenger, _audit_events = _promotion_with_challenger(
        tmp_path,
        monkeypatch,
        strategy="rollout_recovered",
    )
    state_path = tmp_path / "governance" / "live_kpi_breach_state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "strategies": {
                    "rollout_recovered": {
                        "model_id": champion,
                        "consecutive_breach_count": 2,
                        "last_status": "pending",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = promotion.evaluate_live_kpis_and_maybe_rollback(
        strategy="rollout_recovered",
        live_kpis={
            "max_drawdown": 0.01,
            "reject_rate": 0.01,
            "execution_drift_bps": 5.0,
            "drift_psi": 0.1,
            "live_calibration_ece": 0.03,
            "live_calibration_brier": 0.1,
        },
    )

    assert result["breached"] is False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["strategies"]["rollout_recovered"]["consecutive_breach_count"] == 0
    assert state["strategies"]["rollout_recovered"]["observation_hashes"]








