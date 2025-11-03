from __future__ import annotations

import pandas as pd
import pytest

from ai_trading.features.indicators import compute_atr


def test_compute_atr_applies_price_floor() -> None:
    """ATR should clamp to a small fraction of price on flat series."""

    rows = 20
    base = pd.DataFrame(
        {
            "high": [100.0] * rows,
            "low": [100.0] * rows,
            "close": [100.0] * rows,
        }
    )

    result = compute_atr(base.copy())

    assert "atr" in result
    expected_floor = (base["close"].abs() * 0.003).fillna(0.01)
    assert pytest.approx(expected_floor.iloc[-1]) == result["atr"].iloc[-1]
    assert (result["atr"] >= expected_floor).all()
    assert (result["atr"] > 0).all()
