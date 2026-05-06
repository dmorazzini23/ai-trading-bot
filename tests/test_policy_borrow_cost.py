from __future__ import annotations

from ai_trading.policy.compiler import compute_expected_net_edge_bps


def test_borrow_cost_applies_only_to_short_candidates() -> None:
    long_edge = compute_expected_net_edge_bps(
        10.0,
        2.0,
        fee_bps=1.0,
        borrow_bps=3.0,
        side="buy",
    )
    short_edge = compute_expected_net_edge_bps(
        10.0,
        2.0,
        fee_bps=1.0,
        borrow_bps=3.0,
        side="sell_short",
    )

    assert long_edge == 7.0
    assert short_edge == 4.0


def test_borrow_cost_applies_to_short_side_aliases() -> None:
    assert (
        compute_expected_net_edge_bps(
            10.0,
            2.0,
            fee_bps=1.0,
            borrow_bps=3.0,
            side="short_sell",
        )
        == 4.0
    )
    assert (
        compute_expected_net_edge_bps(
            10.0,
            2.0,
            fee_bps=1.0,
            borrow_bps=3.0,
            side="sell-to-open",
        )
        == 4.0
    )
