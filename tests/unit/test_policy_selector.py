from __future__ import annotations

from ai_trading.execution.policy_selector import (
    ExecutionPolicy,
    ExecutionPolicySelector,
)


def test_policy_selector_prefers_passive_on_wide_spread() -> None:
    selector = ExecutionPolicySelector()
    decision = selector.select_policy(
        spread_bps=30.0,
        volatility_pct=0.01,
        order_notional=2_000.0,
        avg_daily_volume_notional=1_000_000.0,
        urgency="low",
        data_provenance="iex",
    )
    assert decision.policy == ExecutionPolicy.PASSIVE_LIMIT


def test_policy_selector_prefers_twap_on_high_participation() -> None:
    selector = ExecutionPolicySelector()
    decision = selector.select_policy(
        spread_bps=8.0,
        volatility_pct=0.02,
        order_notional=200_000.0,
        avg_daily_volume_notional=1_500_000.0,
        urgency="normal",
        data_provenance="iex",
    )
    assert decision.policy == ExecutionPolicy.TWAP


def test_policy_selector_prefers_marketable_when_urgent() -> None:
    selector = ExecutionPolicySelector()
    decision = selector.select_policy(
        spread_bps=4.0,
        volatility_pct=0.01,
        order_notional=5_000.0,
        avg_daily_volume_notional=5_000_000.0,
        urgency="urgent",
        data_provenance="iex",
    )
    assert decision.policy == ExecutionPolicy.MARKETABLE_LIMIT

