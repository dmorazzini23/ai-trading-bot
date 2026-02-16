from __future__ import annotations

from ai_trading.analytics.tca import implementation_shortfall_bps


def test_implementation_shortfall_buy_direction() -> None:
    # Buy worse fill than decision price should be positive cost bps
    value = implementation_shortfall_bps("buy", 100.0, 101.0, fees=0.0, qty=10)
    assert round(value, 6) == 100.0


def test_implementation_shortfall_sell_direction() -> None:
    # Sell worse fill than decision price should also be positive cost bps
    value = implementation_shortfall_bps("sell", 100.0, 99.0, fees=0.0, qty=10)
    assert round(value, 6) == 100.0
