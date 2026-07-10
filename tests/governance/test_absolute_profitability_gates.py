from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ai_trading.governance.promotion import (
    ModelPromotion,
    PromotionCriteria,
    PromotionMetrics,
)
from ai_trading.model_registry import ModelRegistry
from ai_trading.policy.compiler import (
    compile_effective_policy,
    evaluate_counterfactual_non_regression,
)


@pytest.mark.parametrize(
    ("candidate_net_edge_bps", "minimum_net_edge_bps", "expected"),
    [
        (-1.0, 0.0, False),
        (0.0, 0.0, False),
        (float("nan"), 0.0, False),
        (float("inf"), 0.0, False),
        (0.1, 0.0, True),
        (0.5, 0.5, False),
        (0.6, 0.5, True),
        (0.1, -5.0, True),
    ],
)
def test_replay_requires_finite_edge_strictly_above_nonnegative_floor(
    candidate_net_edge_bps: float,
    minimum_net_edge_bps: float,
    expected: bool,
) -> None:
    passed, details = evaluate_counterfactual_non_regression(
        baseline={"sample_count": 100, "net_edge_bps": -10.0, "max_drawdown_pct": 0.02},
        candidate={
            "sample_count": 100,
            "net_edge_bps": candidate_net_edge_bps,
            "max_drawdown_pct": 0.02,
        },
        min_samples=100,
        min_net_edge_bps=minimum_net_edge_bps,
        net_tolerance_bps=20.0,
        drawdown_tolerance_pct=0.0,
    )

    assert passed is expected
    assert details["checks"]["candidate_net_edge_positive"] is expected
    assert details["required"]["min_net_edge_bps"] == max(
        0.0, minimum_net_edge_bps
    )


def test_compiled_replay_floor_cannot_be_negative() -> None:
    policy = compile_effective_policy(
        SimpleNamespace(trading_mode="balanced"),
        {
            "AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE": "0",
            "AI_TRADING_POLICY_REPLAY_MIN_NET_EDGE_BPS": "-12.5",
        },
    )

    assert policy.governance.replay_min_net_edge_bps == 0.0


def _register_shadow_model(registry: ModelRegistry) -> str:
    model_id = cast(
        str,
        registry.register_model(
            model={"marker": "candidate"},
            strategy="absolute_profitability",
            model_type="dict",
        ),
    )
    registry.update_governance_status(
        model_id,
        "shadow",
        {"shadow_start_time": (datetime.now(UTC) - timedelta(days=5)).isoformat()},
    )
    return model_id


def _eligible_metrics(*, net_expectancy_bps: float) -> PromotionMetrics:
    return PromotionMetrics(
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
        reject_rate=0.01,
        execution_drift_bps=10.0,
        challenger_uplift_bps=1.0,
        challenger_p_value=0.01,
        net_expectancy_bps=net_expectancy_bps,
        live_calibration_ece=0.05,
        live_calibration_brier=0.20,
        calibration_samples=50,
        challenger_eval_samples=20,
        challenger_sequential_passes=3,
        last_updated=datetime.now(UTC),
    )


@pytest.mark.parametrize(
    ("candidate_net_expectancy_bps", "configured_floor_bps", "expected"),
    [
        (-1.0, 0.0, False),
        (0.0, 0.0, False),
        (float("nan"), 0.0, False),
        (float("inf"), 0.0, False),
        (0.1, 0.0, True),
        (0.5, 0.5, False),
        (0.6, 0.5, True),
        (0.1, -5.0, True),
    ],
)
def test_shadow_promotion_requires_finite_expectancy_above_nonnegative_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_net_expectancy_bps: float,
    configured_floor_bps: float,
    expected: bool,
) -> None:
    monkeypatch.setenv("AI_TRADING_GOVERNANCE_EVENT_STORE_ENABLED", "0")
    monkeypatch.setenv(
        "AI_TRADING_POLICY_PROMOTION_MIN_OOS_NET_BPS",
        str(configured_floor_bps),
    )
    registry = ModelRegistry(tmp_path / "registry")
    promotion = ModelPromotion(
        model_registry=registry,
        criteria=PromotionCriteria(min_shadow_days=0),
        base_path=str(tmp_path / "governance"),
    )
    model_id = _register_shadow_model(registry)
    promotion._save_shadow_metrics(
        model_id,
        _eligible_metrics(net_expectancy_bps=candidate_net_expectancy_bps),
    )

    eligible, details = promotion.check_promotion_eligibility(model_id)

    assert eligible is expected
    assert details["checks"]["absolute_positive_net_expectancy_check"] is expected
    assert details["criteria"]["absolute_min_net_expectancy_bps"] == max(
        0.0, configured_floor_bps
    )
