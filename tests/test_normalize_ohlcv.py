from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data.fetch.normalize import REQUIRED, normalize_ohlcv_df


def test_normalize_basic_yfinance_columns():
    idx = pd.date_range("2024-01-01", periods=3, freq="1D")
    df = pd.DataFrame(
        {
            "Open": [10.0, 10.5, 10.25],
            "High": [10.5, 10.75, 10.5],
            "Low": [9.5, 10.0, 9.75],
            "Close": [10.25, 10.6, 10.4],
            "Volume": [1000, 1200, 1100],
        },
        index=idx,
    )

    out = normalize_ohlcv_df(df)

    assert list(out.columns) == list(REQUIRED)
    assert out.index.tz is not None
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "timestamp"


def test_normalize_multiindex_columns_dedup():
    idx = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL", "MSFT"]]
    )
    data = {
        col: [float(i), float(i) + 0.5]
        for i, col in enumerate(columns)
    }
    df = pd.DataFrame(data, index=idx)

    out = normalize_ohlcv_df(df)

    assert list(out.columns) == list(REQUIRED)
    # Ensure the first ticker's values survive the deduplication.
    expected = pd.Series(
        [0.0, 0.5],
        index=idx.rename("timestamp"),
        name="open",
    )
    pd.testing.assert_series_equal(out["open"], expected)


def test_normalize_adj_close_mapping():
    idx = pd.date_range("2024-01-01", periods=2, freq="1D")
    df = pd.DataFrame(
        {
            "Adj Close": [100.0, 101.0],
            "Volume": [500, 600],
        },
        index=idx,
    )

    out = normalize_ohlcv_df(df)

    assert "close" in out.columns
    expected = pd.Series([100.0, 101.0], index=out.index, name="close")
    pd.testing.assert_series_equal(out["close"], expected)
