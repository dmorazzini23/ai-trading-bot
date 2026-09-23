"""Pure, cost-explicit whole-share capacity checks for small-account scenarios.

This is a mechanical feasibility calculation, not a strategy backtest or a
source of live risk limits. Every limit and cost assumption must be supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Mapping


def _amount(value: Decimal | int | float | str, name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name}_invalid") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{name}_invalid")
    return parsed


def _symbol_amounts(
    values: Mapping[str, Decimal | int | float | str], name: str, amount_name: str
) -> dict[str, Decimal]:
    amounts: dict[str, Decimal] = {}
    for raw_symbol, value in values.items():
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise ValueError(f"{name}_symbol_invalid")
        symbol = raw_symbol.strip().upper()
        if symbol in amounts:
            raise ValueError(f"{name}_symbol_duplicate")
        amounts[symbol] = _amount(value, amount_name)
    return amounts


@dataclass(frozen=True)
class CostAssumptions:
    entry_spread_bps: Decimal
    entry_slippage_bps: Decimal
    exit_spread_bps: Decimal
    exit_slippage_bps: Decimal
    estimated_total_fee_usd: Decimal | None = None

    def validated(self) -> CostAssumptions:
        values = {
            name: _amount(getattr(self, name), name)
            for name in (
                "entry_spread_bps",
                "entry_slippage_bps",
                "exit_spread_bps",
                "exit_slippage_bps",
            )
        }
        fee = (
            _amount(self.estimated_total_fee_usd, "estimated_total_fee_usd")
            if self.estimated_total_fee_usd is not None
            else None
        )
        return CostAssumptions(**values, estimated_total_fee_usd=fee)


@dataclass(frozen=True)
class CapacityResult:
    capital_usd: Decimal
    requested_shares: int
    feasible_shares: int
    order_notional_usd: Decimal
    pending_buy_commitments_usd: Decimal
    available_cash_before_order_usd: Decimal
    projected_gross_fraction: Decimal
    projected_symbol_fraction: Decimal
    estimated_market_cost_usd: Decimal
    estimated_fee_usd: Decimal | None
    estimated_total_cost_usd: Decimal | None
    verified_total_fee_usd: None
    actual_total_fee_unknown: bool
    reasons: tuple[str, ...]


def evaluate_capacity(
    *,
    capital_usd: Decimal | int | float | str,
    symbol: str,
    price_usd: Decimal | int | float | str,
    requested_shares: int,
    positions_usd: Mapping[str, Decimal | int | float | str],
    pending_buy_usd: Mapping[str, Decimal | int | float | str],
    max_gross_fraction: Decimal | int | float | str,
    max_symbol_fraction: Decimal | int | float | str,
    max_order_notional_usd: Decimal | int | float | str,
    min_order_notional_usd: Decimal | int | float | str,
    cash_reserve_usd: Decimal | int | float | str,
    costs: CostAssumptions,
) -> CapacityResult:
    """Bound an opening buy without counting pending sells as free capital."""

    capital = _amount(capital_usd, "capital_usd")
    price = _amount(price_usd, "price_usd")
    gross_limit = _amount(max_gross_fraction, "max_gross_fraction")
    symbol_limit = _amount(max_symbol_fraction, "max_symbol_fraction")
    order_limit = _amount(max_order_notional_usd, "max_order_notional_usd")
    order_minimum = _amount(min_order_notional_usd, "min_order_notional_usd")
    cash_reserve = _amount(cash_reserve_usd, "cash_reserve_usd")
    validated_costs = costs.validated()
    if capital <= 0 or price <= 0 or not symbol.strip() or requested_shares < 1:
        raise ValueError("capacity_inputs_invalid")
    if gross_limit > 1 or symbol_limit > 1:
        raise ValueError("exposure_fraction_invalid")
    position_amounts = _symbol_amounts(positions_usd, "positions_usd", "position_usd")
    pending_amounts = _symbol_amounts(pending_buy_usd, "pending_buy_usd", "pending_buy_usd")
    position_gross = sum(position_amounts.values(), Decimal(0))
    pending_total = sum(pending_amounts.values(), Decimal(0))
    existing_gross = position_gross + pending_total
    normalized_symbol = symbol.strip().upper()
    existing_symbol = position_amounts.get(normalized_symbol, Decimal(0)) + pending_amounts.get(
        normalized_symbol, Decimal(0)
    )
    available_cash = max(Decimal(0), capital - existing_gross - cash_reserve)
    entry_cost_per_share = price * (
        Decimal(1)
        + (validated_costs.entry_spread_bps + validated_costs.entry_slippage_bps)
        / Decimal(10_000)
    )
    fee_assumption_missing = validated_costs.estimated_total_fee_usd is None
    fee_reserve = (
        validated_costs.estimated_total_fee_usd
        if validated_costs.estimated_total_fee_usd is not None
        else Decimal(0)
    )
    budgets = {
        "cash_limit": max(Decimal(0), available_cash - fee_reserve) / entry_cost_per_share,
        "gross_exposure_limit": max(Decimal(0), capital * gross_limit - existing_gross) / price,
        "symbol_exposure_limit": max(Decimal(0), capital * symbol_limit - existing_symbol) / price,
        "order_notional_limit": order_limit / price,
    }
    share_caps = {
        name: int(value.to_integral_value(rounding=ROUND_FLOOR))
        for name, value in budgets.items()
    }
    feasible = max(0, min(requested_shares, *share_caps.values()))
    notional = price * feasible
    reasons = [name for name, cap in share_caps.items() if cap < requested_shares]
    if fee_assumption_missing:
        # A zero reserve here would make an unknown fee look affordable.
        reasons.append("estimated_fee_assumption_missing")
        feasible = 0
        notional = Decimal(0)
    elif feasible == 0 or notional < order_minimum:
        reasons.append("minimum_order_or_whole_share_unmet")
        feasible = 0
        notional = Decimal(0)
    projected_gross = (existing_gross + notional) / capital
    projected_symbol = (existing_symbol + notional) / capital
    estimated_market_cost = (
        notional
        * (
            validated_costs.entry_spread_bps
            + validated_costs.entry_slippage_bps
            + validated_costs.exit_spread_bps
            + validated_costs.exit_slippage_bps
        )
        / Decimal(10_000)
    )
    estimated_fee = validated_costs.estimated_total_fee_usd if feasible > 0 else None
    return CapacityResult(
        capital_usd=capital,
        requested_shares=requested_shares,
        feasible_shares=feasible,
        order_notional_usd=notional,
        pending_buy_commitments_usd=pending_total,
        available_cash_before_order_usd=available_cash,
        projected_gross_fraction=projected_gross,
        projected_symbol_fraction=projected_symbol,
        estimated_market_cost_usd=estimated_market_cost,
        estimated_fee_usd=estimated_fee,
        estimated_total_cost_usd=(
            estimated_market_cost + estimated_fee if estimated_fee is not None else None
        ),
        verified_total_fee_usd=None,
        actual_total_fee_unknown=True,
        reasons=tuple(reasons),
    )


def evaluate_default_capitals(
    *,
    symbol: str,
    price_usd: Decimal | int | float | str,
    requested_shares: int,
    positions_usd: Mapping[str, Decimal | int | float | str],
    pending_buy_usd: Mapping[str, Decimal | int | float | str],
    max_gross_fraction: Decimal | int | float | str,
    max_symbol_fraction: Decimal | int | float | str,
    max_order_notional_usd: Decimal | int | float | str,
    min_order_notional_usd: Decimal | int | float | str,
    cash_reserve_usd: Decimal | int | float | str,
    costs: CostAssumptions,
) -> tuple[CapacityResult, CapacityResult]:
    """Evaluate the owner's $1,000 and $2,000 scenarios with identical inputs."""

    return (
        evaluate_capacity(
            capital_usd=1000,
            symbol=symbol,
            price_usd=price_usd,
            requested_shares=requested_shares,
            positions_usd=positions_usd,
            pending_buy_usd=pending_buy_usd,
            max_gross_fraction=max_gross_fraction,
            max_symbol_fraction=max_symbol_fraction,
            max_order_notional_usd=max_order_notional_usd,
            min_order_notional_usd=min_order_notional_usd,
            cash_reserve_usd=cash_reserve_usd,
            costs=costs,
        ),
        evaluate_capacity(
            capital_usd=2000,
            symbol=symbol,
            price_usd=price_usd,
            requested_shares=requested_shares,
            positions_usd=positions_usd,
            pending_buy_usd=pending_buy_usd,
            max_gross_fraction=max_gross_fraction,
            max_symbol_fraction=max_symbol_fraction,
            max_order_notional_usd=max_order_notional_usd,
            min_order_notional_usd=min_order_notional_usd,
            cash_reserve_usd=cash_reserve_usd,
            costs=costs,
        ),
    )
