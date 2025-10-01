from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data.fetch import (
    MissingOHLCVColumnsError,
    _flatten_and_normalize_ohlcv,
    ensure_ohlcv_schema,
)
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


def test_flatten_handles_ticker_multiindex_level1():
    idx = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    columns = pd.MultiIndex.from_product([["MSFT"], ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(
        [[10.0, 10.5, 9.8, 10.2, 1000], [10.1, 10.6, 9.9, 10.3, 1100]],
        index=idx,
        columns=columns,
    )

    try:
        out = _flatten_and_normalize_ohlcv(df.copy(), symbol="MSFT")
    except MissingOHLCVColumnsError as exc:  # pragma: no cover - explicit failure path
        pytest.fail(f"unexpected MissingOHLCVColumnsError: {exc}")

    assert {"open", "high", "low", "close", "volume"}.issubset(out.columns)
    unexpected = {col for col in out.columns if isinstance(col, str) and col.startswith("msft_")}
    assert not unexpected


def test_flatten_backfills_missing_price_columns_from_close():
    idx = pd.date_range("2024-01-01", periods=2, freq="1D")
    columns = pd.MultiIndex.from_tuples(
        [("Close", "AAPL"), ("Adj Close", "AAPL")]
    )
    df = pd.DataFrame(
        [[150.0, 150.5], [151.0, 151.4]],
        index=idx,
        columns=columns,
    )

    try:
        out = _flatten_and_normalize_ohlcv(df.copy(), symbol="AAPL", timeframe="1Day")
    except MissingOHLCVColumnsError as exc:  # pragma: no cover - explicit failure path
        pytest.fail(f"unexpected MissingOHLCVColumnsError: {exc}")

    required = {"open", "high", "low", "close", "volume"}
    assert required.issubset(out.columns)

    pd.testing.assert_series_equal(out["open"], out["close"], check_names=False)
    pd.testing.assert_series_equal(out["high"], out["close"], check_names=False)
    pd.testing.assert_series_equal(out["low"], out["close"], check_names=False)
    assert (out["volume"] == 0).all()


def test_ensure_schema_handles_multiindex_alias_level():
    idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    level0 = ["Open", "High", "Low", "Close", "Volume"]
    level1 = ["AAPL", "MSFT"]
    columns = pd.MultiIndex.from_product([level0, level1])
    data = {col: [float(i), float(i) + 1.0, float(i) + 2.0] for i, col in enumerate(columns)}
    df = pd.DataFrame(data, index=idx)

    try:
        ensured = ensure_ohlcv_schema(df.copy(), source="fixture", frequency="1Day")
    except MissingOHLCVColumnsError as exc:  # pragma: no cover - explicit failure path
        pytest.fail(f"ensure_ohlcv_schema unexpectedly raised: {exc}")

    assert set(REQUIRED).issubset(ensured.columns)


def test_normalize_handles_latest_price_columns():
    timestamps = pd.to_datetime(["2024-01-02T09:30:00Z", "2024-01-02T09:31:00Z"], utc=True)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sessionOpen": [188.45, 188.5],
            "sessionHigh": [189.12, 189.2],
            "sessionLow": [187.3, 187.8],
            "latestPrice": [188.77, 188.9],
            "sessionVolume": [1_234_567, 1_300_000],
        }
    )
    df = df.set_index("timestamp")

    normalized = normalize_ohlcv_df(df)

    assert list(normalized.columns[:6]) == ["timestamp", *REQUIRED]
    first = normalized.iloc[0]
    assert pytest.approx(first["open"]) == 188.45
    assert pytest.approx(first["close"]) == 188.77
    assert pytest.approx(first["volume"]) == 1_234_567
