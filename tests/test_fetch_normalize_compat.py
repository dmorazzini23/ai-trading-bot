"""Compatibility tests for OHLCV normalization helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data.fetch import _is_normalized_ohlcv_frame
from ai_trading.data.fetch.normalize import normalize_ohlcv_df


def _make_index(ts: list[datetime]) -> "pd.DatetimeIndex":
    return pd.DatetimeIndex(ts, name="timestamp")


def test_is_normalized_accepts_superset_and_any_order() -> None:
    ts = [datetime(2024, 1, 1, tzinfo=UTC)]
    df = pd.DataFrame(
        {
            "volume": [1],
            "close": [10.0],
            "open": [9.0],
            "high": [10.5],
            "low": [8.9],
            "timestamp": ts,
            "trade_count": [5],
        },
        index=_make_index(ts),
    )
    assert _is_normalized_ohlcv_frame(df) is True


def test_normalize_aliases_compact_and_adjclose() -> None:
    ts = [datetime(2024, 1, 2, tzinfo=UTC)]
    df = pd.DataFrame(
        {
            "o": [9.0],
            "h": [10.5],
            "l": [8.9],
            "c": [10.0],
            "v": [1],
            "adjclose": [9.9],
            "noise": ["ignored"],
        },
        index=_make_index(ts),
    )

    normalized = normalize_ohlcv_df(df, include_columns=("timestamp",))
    assert list(normalized.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert normalized.iloc[0]["close"] == 10.0
    assert "adj close" not in normalized.columns
