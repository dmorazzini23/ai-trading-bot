from __future__ import annotations

from ai_trading.core.netting import (
    clear_netting_cost_cache,
    estimate_cost_bps,
)


def test_estimate_cost_bps_range_proxy_is_bounded() -> None:
    clear_netting_cost_cache()
    estimate = estimate_cost_bps(
        price=100.0,
        spread=5.0,
        vol=0.10,
        size_dollars=25_000.0,
        volume=2_000.0,
    )
    assert estimate <= 25.0
    assert estimate >= 1.0


def test_estimate_cost_bps_participation_is_sublinear() -> None:
    clear_netting_cost_cache()
    cost_small = estimate_cost_bps(
        price=100.0,
        spread=0.5,
        vol=0.01,
        size_dollars=2_000.0,
        volume=10_000.0,
    )
    cost_large = estimate_cost_bps(
        price=100.0,
        spread=0.5,
        vol=0.01,
        size_dollars=18_000.0,
        volume=10_000.0,
    )
    assert cost_large > cost_small
    assert (cost_large - cost_small) < 20.0

