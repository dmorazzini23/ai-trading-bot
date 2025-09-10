from decimal import Decimal

from ai_trading.market.symbol_specs import get_symbol_spec


def test_ma_symbol_spec() -> None:
    spec = get_symbol_spec("MA")
    assert spec.tick == Decimal("0.01")
    assert spec.lot == 1
    assert spec.currency == "USD"
