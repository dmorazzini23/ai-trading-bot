from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_trading.registry.manifest import (
    MARKET_REGIME_CLASSIFIER_ID,
    ManifestValidationError,
    derive_market_regime_policy,
    evaluate_market_regime_policy,
    validate_market_regime_policy,
)


def _walk_forward(*, candidate_qualified: bool = True) -> dict[str, Any]:
    def row(
        *,
        support: int,
        edge: float,
        ranking: float,
        profitable_fold_ratio: float,
        qualified: bool,
    ) -> dict[str, Any]:
        return {
            "support": support,
            "mean_post_cost_net_edge_bps": edge,
            "profitable_fold_ratio": profitable_fold_ratio,
            "ranking_separation": {
                "high_minus_low_net_edge_bps": ranking,
            },
            "evidence_qualified": qualified,
        }

    return {
        "market_regime_classifier": MARKET_REGIME_CLASSIFIER_ID,
        "aggregate": {"evidence_qualified": candidate_qualified},
        "config": {
            "min_trades": 75,
            "min_mean_net_edge_bps": 0.0,
            "min_ranking_separation_bps": 0.0,
            "min_profitable_fold_ratio": 0.60,
        },
        "by_market_regime": {
            "sideways": row(
                support=40,
                edge=6.21,
                ranking=2.5,
                profitable_fold_ratio=0.80,
                qualified=True,
            ),
            "downtrend": row(
                support=30,
                edge=-4.0,
                ranking=1.0,
                profitable_fold_ratio=0.40,
                qualified=False,
            ),
            "volatile": row(
                support=0,
                edge=-8.0,
                ranking=-1.0,
                profitable_fold_ratio=0.0,
                qualified=False,
            ),
        },
    }


def test_policy_derivation_observes_only_supported_positive_stable_regime() -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)

    policy = derive_market_regime_policy(_walk_forward(), generated_at=now)

    assert policy["allowed_regimes"] == ["sideways"]
    assert policy["abstained_regimes"] == ["downtrend", "volatile"]
    assert policy["governance_status"] == "shadow"
    assert policy["promotion_authority"] is False
    assert policy["regimes"]["sideways"]["support"] == 40
    assert policy["regimes"]["sideways"]["mean_post_cost_net_edge_bps"] == pytest.approx(
        6.21
    )
    assert evaluate_market_regime_policy(
        policy,
        market_regime="sideways",
        now=now,
    ).allowed
    assert not evaluate_market_regime_policy(
        policy,
        market_regime="downtrend",
        now=now,
    ).allowed
    assert not evaluate_market_regime_policy(
        policy,
        market_regime="volatile",
        now=now,
    ).allowed


def test_unqualified_candidate_derives_valid_explicit_all_abstain_policy() -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    walk_forward = _walk_forward(candidate_qualified=False)
    zero_support = walk_forward["by_market_regime"]["volatile"]
    zero_support["mean_post_cost_net_edge_bps"] = None
    zero_support["ranking_separation"]["high_minus_low_net_edge_bps"] = None

    policy = derive_market_regime_policy(
        walk_forward,
        generated_at=now,
    )

    assert policy["allowed_regimes"] == []
    assert policy["abstained_regimes"] == ["downtrend", "sideways", "volatile"]
    assert validate_market_regime_policy(policy) == policy
    assert not evaluate_market_regime_policy(
        policy,
        market_regime="sideways",
        now=now,
    ).allowed


def test_policy_derivation_rejects_classifier_mismatch() -> None:
    walk_forward = _walk_forward()
    walk_forward["market_regime_classifier"] = "fold_local_atr_thresholds"

    with pytest.raises(ManifestValidationError, match="incompatible with serving"):
        derive_market_regime_policy(
            walk_forward,
            generated_at=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
        )


def test_declared_policy_fails_closed_when_stale_unknown_or_malformed() -> None:
    generated_at = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    policy = derive_market_regime_policy(
        _walk_forward(),
        generated_at=generated_at,
        max_age_days=14,
    )

    stale = evaluate_market_regime_policy(
        policy,
        market_regime="sideways",
        now=generated_at + timedelta(days=15),
    )
    unknown = evaluate_market_regime_policy(
        policy,
        market_regime="pandemic",
        now=generated_at,
    )
    malformed_policy = deepcopy(policy)
    malformed_policy["allowed_regimes"] = ["sideways", "volatile"]
    malformed = evaluate_market_regime_policy(
        malformed_policy,
        market_regime="sideways",
        now=generated_at,
    )

    assert (stale.allowed, stale.reason) == (False, "policy_stale")
    assert (unknown.allowed, unknown.reason) == (False, "market_regime_unknown")
    assert (malformed.allowed, malformed.reason) == (False, "policy_invalid")


def test_absent_policy_preserves_legacy_behavior_without_declaring_authority() -> None:
    decision = evaluate_market_regime_policy(
        None,
        market_regime="sideways",
        now=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
    )

    assert decision.allowed is True
    assert decision.declared is False
    assert decision.reason == "legacy_policy_absent"
    assert decision.evidence_identity is None
    assert decision.evidence_sha256 is None
