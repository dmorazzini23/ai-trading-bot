from __future__ import annotations

from types import SimpleNamespace

from ai_trading.policy.compiler import (
    ExecutionCandidate,
    SafetyTier,
    approve_execution_candidate,
    compile_effective_policy,
)


def _base_candidate(*, reject_rate_pct: float) -> ExecutionCandidate:
    return ExecutionCandidate(
        symbol="AAPL",
        side="buy",
        proposed_delta_shares=10,
        current_shares=0,
        price=100.0,
        expected_edge_bps=40.0,
        expected_cost_bps=2.0,
        confidence=0.8,
        spread_bps=1.0,
        rolling_volume=100000.0,
        pending_oldest_age_sec=0.0,
        pacing_headroom=5,
        stale_orders_present=False,
        calibration_ok=True,
        portfolio_post_gross_dollars=1000.0,
        sleeve_post_notional_dollars=1000.0,
        factor_post_ratio=0.05,
        reject_rate_pct=reject_rate_pct,
        safety_tier=SafetyTier.ATTACK,
    )


def test_attack_scale_uses_relaxed_multiplier_when_reject_rate_low() -> None:
    policy = compile_effective_policy(
        SimpleNamespace(trading_mode="balanced"),
        env={
            "AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE": "0",
            "AI_TRADING_POLICY_ATTACK_SIZE_MULTIPLIER": "1.40",
            "AI_TRADING_POLICY_ATTACK_RELAX_SIZE_MULTIPLIER": "1.10",
            "AI_TRADING_POLICY_ATTACK_RELAX_REJECT_RATE_PCT": "2.0",
        },
    )

    approval = approve_execution_candidate(policy, _base_candidate(reject_rate_pct=1.0))

    assert approval.allowed is True
    assert approval.adjusted_delta_shares == 11
    assert "SAFETY_TIER_ATTACK_SCALE_RELAXED" in approval.reasons
    assert "SAFETY_TIER_ATTACK_SCALE" in approval.reasons


def test_attack_scale_uses_base_multiplier_when_reject_rate_high() -> None:
    policy = compile_effective_policy(
        SimpleNamespace(trading_mode="balanced"),
        env={
            "AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE": "0",
            "AI_TRADING_POLICY_ATTACK_SIZE_MULTIPLIER": "1.40",
            "AI_TRADING_POLICY_ATTACK_RELAX_SIZE_MULTIPLIER": "1.10",
            "AI_TRADING_POLICY_ATTACK_RELAX_REJECT_RATE_PCT": "2.0",
        },
    )

    approval = approve_execution_candidate(policy, _base_candidate(reject_rate_pct=5.0))

    assert approval.allowed is True
    assert approval.adjusted_delta_shares == 14
    assert "SAFETY_TIER_ATTACK_SCALE_RELAXED" not in approval.reasons
    assert "SAFETY_TIER_ATTACK_SCALE" in approval.reasons
