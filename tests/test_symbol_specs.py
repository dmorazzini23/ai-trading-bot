"""Unit tests for symbol specifications."""

from decimal import Decimal

import pytest

from ai_trading.market.symbol_specs import DEFAULT_SPEC, get_symbol_spec


@pytest.mark.parametrize("symbol", ["ADBE", "HD", "AMGN"])
def test_known_symbols_have_specific_specs(symbol: str) -> None:
    spec = get_symbol_spec(symbol)
    assert spec is not DEFAULT_SPEC
    assert spec.tick == Decimal("0.01")
    assert spec.lot == 1
    assert spec.trading_hours == "09:30-16:00"


def test_unknown_symbol_uses_default_spec() -> None:
    spec = get_symbol_spec("UNKNOWN")
    assert spec is DEFAULT_SPEC


@pytest.mark.parametrize("symbol", ["COST", "NFLX"])
def test_default_specs(symbol: str) -> None:
    spec = get_symbol_spec(symbol)
    assert spec is not DEFAULT_SPEC
    assert spec.tick == Decimal("0.01")
    assert spec.lot == 1

