from __future__ import annotations

from decimal import Decimal

import pytest

from ai_trading.tools.small_account_capacity import (
    CostAssumptions,
    evaluate_capacity,
    evaluate_default_capitals,
)


def _costs(fee: Decimal | None = Decimal("1")) -> CostAssumptions:
    return CostAssumptions(
        entry_spread_bps=Decimal("5"),
        entry_slippage_bps=Decimal("2"),
        exit_spread_bps=Decimal("5"),
        exit_slippage_bps=Decimal("2"),
        estimated_total_fee_usd=fee,
    )


def test_default_capitals_include_pending_commitments_and_concentration() -> None:
    small, large = evaluate_default_capitals(
        symbol="AAPL",
        price_usd="150",
        requested_shares=5,
        positions_usd={},
        pending_buy_usd={"AAPL": "200"},
        max_gross_fraction="1",
        max_symbol_fraction="0.25",
        max_order_notional_usd="500",
        min_order_notional_usd="50",
        cash_reserve_usd="100",
        costs=_costs(),
    )

    assert (small.capital_usd, large.capital_usd) == (Decimal(1000), Decimal(2000))
    assert small.pending_buy_commitments_usd == large.pending_buy_commitments_usd == 200
    assert small.feasible_shares == 0
    assert "symbol_exposure_limit" in small.reasons
    assert large.feasible_shares == 2
    assert large.order_notional_usd == 300
    assert large.projected_symbol_fraction == Decimal("0.25")
    assert large.projected_gross_fraction == Decimal("0.25")
    assert large.verified_total_fee_usd is None
    assert large.actual_total_fee_unknown


def test_cost_estimate_is_separate_from_unknown_verified_fee() -> None:
    result = evaluate_capacity(
        capital_usd="1000",
        symbol="AAPL",
        price_usd="100",
        requested_shares=1,
        positions_usd={},
        pending_buy_usd={},
        max_gross_fraction="1",
        max_symbol_fraction="1",
        max_order_notional_usd="200",
        min_order_notional_usd="10",
        cash_reserve_usd="0",
        costs=_costs(),
    )
    assert result.feasible_shares == 1
    assert result.estimated_market_cost_usd == Decimal("0.14")
    assert result.estimated_fee_usd == 1
    assert result.estimated_total_cost_usd == Decimal("1.14")
    assert result.verified_total_fee_usd is None

    unknown_fee = evaluate_capacity(
        capital_usd="1000",
        symbol="AAPL",
        price_usd="100",
        requested_shares=1,
        positions_usd={},
        pending_buy_usd={},
        max_gross_fraction="1",
        max_symbol_fraction="1",
        max_order_notional_usd="200",
        min_order_notional_usd="10",
        cash_reserve_usd="0",
        costs=_costs(None),
    )
    assert unknown_fee.estimated_market_cost_usd == Decimal("0.14")
    assert unknown_fee.estimated_total_cost_usd is None
    assert unknown_fee.actual_total_fee_unknown


def test_whole_share_and_cash_reserve_can_make_order_infeasible() -> None:
    result = evaluate_capacity(
        capital_usd=1000,
        symbol="AAPL",
        price_usd=250,
        requested_shares=1,
        positions_usd={"MSFT": 600},
        pending_buy_usd={"AMZN": 200},
        max_gross_fraction="1",
        max_symbol_fraction="0.5",
        max_order_notional_usd=300,
        min_order_notional_usd=100,
        cash_reserve_usd=100,
        costs=_costs(),
    )
    assert result.feasible_shares == 0
    assert result.available_cash_before_order_usd == 100
    assert "cash_limit" in result.reasons
    assert "minimum_order_or_whole_share_unmet" in result.reasons


def test_invalid_limits_and_costs_reject_instead_of_sizing() -> None:
    base = dict(
        capital_usd=1000,
        symbol="AAPL",
        price_usd=100,
        requested_shares=1,
        positions_usd={},
        pending_buy_usd={},
        max_gross_fraction=1,
        max_symbol_fraction=1,
        max_order_notional_usd=100,
        min_order_notional_usd=1,
        cash_reserve_usd=0,
        costs=_costs(),
    )
    with pytest.raises(ValueError, match="exposure_fraction_invalid"):
        evaluate_capacity(**(base | {"max_symbol_fraction": 1.2}))
    with pytest.raises(ValueError, match="pending_buy_usd_invalid"):
        evaluate_capacity(**(base | {"pending_buy_usd": {"AAPL": -1}}))
    with pytest.raises(ValueError, match="entry_spread_bps_invalid"):
        evaluate_capacity(
            **(base | {"costs": CostAssumptions(-1, 0, 0, 0)})
        )
