import math

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.features import compute_macd


def test_compute_macd_preserves_datetime_index_alignment():
    index = pd.date_range("2024-01-01", periods=60, freq="min")
    df = pd.DataFrame({"close": pd.Series(range(1, 61), index=index)})

    result = compute_macd(df.copy())

    assert "macd" in result.columns
    assert "signal" in result.columns

    macd_values = result["macd"].dropna()
    signal_values = result["signal"].dropna()

    assert not macd_values.empty
    assert not signal_values.empty
    assert macd_values.map(math.isfinite).all()
    assert signal_values.map(math.isfinite).all()
