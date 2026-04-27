from __future__ import annotations

import pytest

from ai_trading.portfolio import sizing


def test_turnover_penalty_does_not_normalize_underinvested_partial_adjustment() -> None:
    sizer = sizing.TurnoverPenaltySizer(max_turnover=0.05)
    current = {"AAPL": 0.05}
    proposed = {"AAPL": 0.20}

    adjusted = sizer.apply_turnover_penalty(proposed, current)

    assert adjusted["AAPL"] == pytest.approx(0.15)
    assert sizer._calculate_turnover(adjusted, current) == pytest.approx(0.05)
