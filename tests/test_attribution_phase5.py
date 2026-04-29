from __future__ import annotations

import pytest

from ai_trading.analytics.attribution import arrival_slippage_bps, spread_paid_bps


@pytest.mark.parametrize(
    "side",
    ["sell", "short", "sell_short", "sell-short", "sellshort", "entry_short", "open_short", "short_sell"],
)
def test_slippage_attribution_treats_short_aliases_as_sell_side(side: str) -> None:
    assert arrival_slippage_bps(100.0, 99.0, side) == pytest.approx(100.0)
    assert spread_paid_bps(99.0, 101.0, 99.0, side) == pytest.approx(100.0)
