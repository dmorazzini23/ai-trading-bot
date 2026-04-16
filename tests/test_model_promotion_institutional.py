from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from ai_trading.governance.promotion import ModelPromotion, PromotionCriteria
from ai_trading.model_registry import ModelRegistry


def _register_test_model(registry: ModelRegistry, *, strategy: str, marker: str) -> str:
    return cast(
        str,
        registry.register_model(
            model={"marker": marker},
            strategy=strategy,
            model_type="dict",
            metadata={"marker": marker},
        ),
    )


def test_promote_marks_previous_production_as_challenger(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))
    strategy = "momentum"

    model_a = _register_test_model(registry, strategy=strategy, marker="a")
    model_b = _register_test_model(registry, strategy=strategy, marker="b")
    registry.update_governance_status(model_a, "production")

    promoted = promotion.promote_to_production(model_b, force=True)

    assert promoted is True
    production = registry.get_production_model(strategy)
    assert production is not None
    prod_id, prod_meta = production
    assert prod_id == model_b
    assert prod_meta.get("governance", {}).get("previous_production_model_id") == model_a
    assert registry.model_index[model_a]["governance"]["status"] == "challenger"


def test_rollback_to_previous_production(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))
    strategy = "mean_reversion"

    champion = _register_test_model(registry, strategy=strategy, marker="champion")
    challenger = _register_test_model(registry, strategy=strategy, marker="challenger")
    registry.update_governance_status(champion, "production")
    assert promotion.promote_to_production(challenger, force=True) is True

    rolled_back = promotion.rollback_to_previous_production(
        strategy=strategy,
        reason="live_degradation",
    )

    assert rolled_back is True
    production = registry.get_production_model(strategy)
    assert production is not None
    prod_id, _prod_meta = production
    assert prod_id == champion
    assert registry.model_index[challenger]["governance"]["status"] == "challenger"
    assert (
        registry.model_index[challenger]["governance"]["rolled_back_to_model_id"]
        == champion
    )


def test_record_challenger_evaluation_writes_jsonl(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))

    eval_path = promotion.record_challenger_evaluation(
        strategy="swing",
        champion_model_id="champ-1",
        challenger_model_id="challenger-2",
        metrics={"is_bps": 4.2, "reject_rate": 0.01},
    )

    assert eval_path is not None
    lines = (tmp_path / "governance" / "challenger_evaluations.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["strategy"] == "swing"
    assert payload["champion_model_id"] == "champ-1"
    assert payload["challenger_model_id"] == "challenger-2"


def test_record_challenger_evaluation_adds_significance(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))

    eval_path = promotion.record_challenger_evaluation(
        strategy="swing",
        champion_model_id="champ",
        challenger_model_id="chall",
        metrics={
            "challenger_returns": [0.003, 0.002, 0.004, 0.003],
            "champion_returns": [0.001, 0.001, 0.001, 0.001],
        },
    )

    assert eval_path is not None
    payload = json.loads((tmp_path / "governance" / "challenger_evaluations.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "significance" in payload
    assert float(payload["significance"]["uplift_bps"]) > 0.0


def test_live_kpi_control_band_triggers_rollback(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))
    strategy = "momentum"

    champion = _register_test_model(registry, strategy=strategy, marker="champion")
    challenger = _register_test_model(registry, strategy=strategy, marker="challenger")
    registry.update_governance_status(champion, "production")
    assert promotion.promote_to_production(challenger, force=True) is True

    result = promotion.evaluate_live_kpis_and_maybe_rollback(
        strategy=strategy,
        live_kpis={"max_drawdown": 0.20, "reject_rate": 0.01, "execution_drift_bps": 10.0},
    )

    assert result["breached"] is True
    assert result["triggered"] is True
    production = registry.get_production_model(strategy)
    assert production is not None
    assert production[0] == champion


def test_live_kpi_control_band_pending_when_rollback_not_allowed(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))
    strategy = "momentum"

    champion = _register_test_model(registry, strategy=strategy, marker="champion")
    challenger = _register_test_model(registry, strategy=strategy, marker="challenger")
    registry.update_governance_status(champion, "production")
    assert promotion.promote_to_production(challenger, force=True) is True

    result = promotion.evaluate_live_kpis_and_maybe_rollback(
        strategy=strategy,
        live_kpis={"max_drawdown": 0.20, "reject_rate": 0.01, "execution_drift_bps": 10.0},
        allow_rollback=False,
    )

    assert result["breached"] is True
    assert result["triggered"] is False
    assert result["status"] == "pending"
    production = registry.get_production_model(strategy)
    assert production is not None
    assert production[0] == challenger


def test_update_shadow_metrics_autoderives_validation_ratios(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))
    strategy = "ml_edge"
    model_id = _register_test_model(registry, strategy=strategy, marker="candidate")
    assert promotion.start_shadow_testing(model_id) is True

    returns = [0.0025 if idx % 3 else -0.001 for idx in range(240)]
    regimes = ["trend", "chop", "high_vol"] * 80
    promotion.update_shadow_metrics(
        model_id,
        {
            "trade_count": 240,
            "turnover_ratio": 0.8,
            "returns": returns,
            "regimes": regimes,
        },
    )

    metrics = promotion._load_shadow_metrics(model_id)  # noqa: SLF001 - test-only inspection
    assert metrics is not None
    assert metrics.purged_walk_forward_pass_ratio > 0.0
    assert metrics.regime_pass_ratio > 0.0


def test_challenger_sequential_gate_requires_consecutive_passes(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    criteria = PromotionCriteria(
        min_shadow_sessions=1,
        min_shadow_days=0,
        min_trade_count=1,
        max_turnover_ratio=10.0,
        min_live_sharpe=-10.0,
        max_drift_psi=1.0,
        max_drawdown_threshold=1.0,
        min_live_sortino=-10.0,
        min_live_calmar=-10.0,
        max_tail_loss_95=1.0,
        max_risk_of_ruin=1.0,
        min_purged_walk_forward_pass_ratio=0.0,
        min_monte_carlo_p05_bps=-10_000.0,
        min_regime_pass_ratio=0.0,
        require_tca_gate=False,
        max_reject_rate=1.0,
        max_execution_drift_bps=10_000.0,
        min_net_expectancy_bps=-10_000.0,
        max_live_calibration_ece=1.0,
        max_live_calibration_brier=1.0,
        min_calibration_samples=1,
        challenger_sequential_min_samples=4,
        challenger_sequential_required_passes=2,
        min_challenger_uplift_bps=-10_000.0,
    )
    promotion = ModelPromotion(
        model_registry=registry,
        criteria=criteria,
        base_path=str(tmp_path / "governance"),
    )
    model_id = _register_test_model(registry, strategy="ml_edge", marker="candidate")
    assert promotion.start_shadow_testing(model_id) is True

    payload = {
        "trade_count": 10,
        "turnover_ratio": 0.8,
        "returns": [0.002, 0.001, -0.0005, 0.0015, 0.0025],
        "live_calibration_ece": 0.04,
        "live_calibration_brier": 0.18,
        "calibration_samples": 10,
        "challenger_returns": [0.004, 0.003, 0.004, 0.005],
        "champion_returns": [0.001, 0.001, 0.001, 0.001],
    }

    promotion.update_shadow_metrics(model_id, payload)
    eligible_1, details_1 = promotion.check_promotion_eligibility(model_id)
    assert eligible_1 is False
    assert details_1["checks"]["challenger_sequential_check"] is False

    promotion.update_shadow_metrics(model_id, payload)
    eligible_2, details_2 = promotion.check_promotion_eligibility(model_id)
    assert details_2["checks"]["challenger_sequential_check"] is True
    assert "calibration_ece_check" in details_2["checks"]
    assert "calibration_brier_check" in details_2["checks"]


def test_promote_to_production_requires_explicit_approval_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PROMOTION_REQUIRE_APPROVAL", "1")
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))
    strategy = "approval_gate"

    champion = _register_test_model(registry, strategy=strategy, marker="champion")
    challenger = _register_test_model(registry, strategy=strategy, marker="challenger")
    registry.update_governance_status(champion, "production")
    registry.update_governance_status(challenger, "shadow")
    monkeypatch.setattr(
        promotion,
        "check_promotion_eligibility",
        lambda _model_id: (True, {"eligible": True}),
    )

    assert promotion.promote_to_production(challenger, force=False) is False

    approval_path = promotion.record_promotion_approval(
        strategy=strategy,
        model_id=challenger,
        approver="ops@example.com",
        decision="approved",
        note="weekly promotion review",
    )
    assert approval_path is not None

    promoted = promotion.promote_to_production(challenger, force=False)
    assert promoted is True
    governance = registry.model_index[challenger]["governance"]
    assert governance["status"] == "production"
    assert governance["promotion_approved_by"] == "ops@example.com"
    assert governance["promotion_approval_id"]


def test_record_challenger_evaluation_writes_scorecard(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))

    promotion.record_challenger_evaluation(
        strategy="swing",
        champion_model_id="champ",
        challenger_model_id="chall",
        metrics={
            "challenger_returns": [0.004, 0.003, 0.004, 0.005],
            "champion_returns": [0.001, 0.001, 0.001, 0.001],
            "champion_net_expectancy_bps": 4.0,
            "challenger_net_expectancy_bps": 8.0,
        },
    )

    scorecard_lines = (
        tmp_path / "governance" / "champion_challenger_scorecards.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    assert len(scorecard_lines) == 1
    scorecard = json.loads(scorecard_lines[0])
    assert scorecard["champion_model_id"] == "champ"
    assert scorecard["challenger_model_id"] == "chall"
    assert float(scorecard["delta_net_expectancy_bps"]) == 4.0


def test_rollback_to_previous_production_writes_audit_log(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path / "governance"))
    strategy = "audit_lineage"

    champion = _register_test_model(registry, strategy=strategy, marker="champion")
    challenger = _register_test_model(registry, strategy=strategy, marker="challenger")
    registry.update_governance_status(champion, "production")
    assert promotion.promote_to_production(challenger, force=True) is True
    assert promotion.rollback_to_previous_production(strategy=strategy, reason="drawdown") is True

    lines = (tmp_path / "governance" / "rollback_audit.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert lines
    payload = json.loads(lines[-1])
    assert payload["status"] == "rolled_back"
    assert payload["from_model_id"] == challenger
    assert payload["to_model_id"] == champion
