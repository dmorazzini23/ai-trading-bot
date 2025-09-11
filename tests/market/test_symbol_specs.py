from decimal import Decimal

from ai_trading.market.symbol_specs import DEFAULT_SPEC, get_symbol_spec


def test_ma_symbol_spec() -> None:
    spec = get_symbol_spec("MA")
    assert spec.tick == Decimal("0.01")
    assert spec.lot == 1
    assert spec.currency == "USD"


def test_gs_symbol_spec() -> None:
    spec = get_symbol_spec("GS")
    assert spec.tick == Decimal("0.01")
    assert spec.lot == 1
    assert spec.currency == "USD"


def test_tmo_symbol_spec() -> None:
    spec = get_symbol_spec("TMO")
    assert spec is not DEFAULT_SPEC
    assert spec.tick == Decimal("0.01")
    assert spec.lot == 1


def test_cat_symbol_spec() -> None:
    spec = get_symbol_spec("CAT")
    assert spec is not DEFAULT_SPEC
    assert spec.tick == Decimal("0.01")
    assert spec.lot == 1
