from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trading.features.day_sleeve import build_day_sleeve_features
from ai_trading.models.contracts import DAY_SLEEVE_ML_FEATURE_COLUMNS


def _bars(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2026-07-09 13:30", periods=rows, freq="5min", tz="UTC")
    step = np.arange(rows, dtype=float)
    close = 100.0 + (step * 0.02) + np.sin(step / 8.0)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "volume": 10_000.0 + step,
        },
        index=index,
    )


def test_day_sleeve_feature_builder_returns_exact_finite_latest_row() -> None:
    features = build_day_sleeve_features(_bars())

    assert tuple(features.columns) == DAY_SLEEVE_ML_FEATURE_COLUMNS
    assert len(features) == 1
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    row = features.iloc[-1]
    assert row["macd_signal_gap"] == pytest.approx(row["macd"] - row["signal"])
    assert row["rsi_centered"] == pytest.approx((row["rsi"] - 50.0) / 50.0)


@pytest.mark.parametrize("mutation", ["missing_volume", "nonfinite_close"])
def test_day_sleeve_feature_builder_rejects_invalid_inputs(mutation: str) -> None:
    bars = _bars()
    if mutation == "missing_volume":
        bars = bars.drop(columns=["volume"])
    else:
        bars.loc[bars.index[-1], "close"] = np.nan

    with pytest.raises(ValueError, match="missing columns|non-finite"):
        build_day_sleeve_features(bars)

