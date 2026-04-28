from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data.splits import (
    PurgedGroupTimeSeriesSplit,
    validate_no_leakage,
    walkforward_splits,
)


def test_purged_group_time_series_split_handles_arrays_and_fractional_test_size() -> None:
    splitter = PurgedGroupTimeSeriesSplit(n_splits=3, test_size=0.2, embargo_pct=0.05, purge_pct=0.05)
    splits = list(splitter.split(np.arange(50)))

    assert splitter.get_n_splits() == 3
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) == 10
        assert train_idx.max() < test_idx.min()


def test_purged_group_time_series_split_uses_t1_to_remove_overlaps() -> None:
    index = pd.date_range("2026-01-01", periods=30, freq="D")
    frame = pd.DataFrame({"value": np.arange(30)}, index=index)
    t1 = pd.Series(index + pd.Timedelta(days=10), index=index)
    splitter = PurgedGroupTimeSeriesSplit(n_splits=2, test_size=5, embargo_pct=0.0, purge_pct=0.0)

    train_idx, test_idx = next(splitter.split(frame, t1=t1))

    assert len(train_idx) > 0
    assert all(t1.iloc[idx] < index[test_idx[0]] for idx in train_idx)


def test_purge_overlapping_defensive_paths() -> None:
    splitter = PurgedGroupTimeSeriesSplit(n_splits=2)
    train = np.array([0, 1, 2, 3])
    empty_test = np.array([], dtype=int)
    bad_full_index = np.array([object(), object(), object(), object()])
    t1 = pd.Series([pd.NaT, pd.Timestamp("2026-01-01")])

    assert splitter._purge_overlapping(train, empty_test, t1, bad_full_index).tolist() == [
        0,
        1,
        2,
        3,
    ]
    purged = splitter._purge_overlapping(train, np.array([3]), t1, np.arange(4))
    assert purged.tolist() == [0]


def test_walkforward_splits_support_rolling_expanding_and_bad_input() -> None:
    dates = pd.date_range("2026-01-01", periods=90, freq="D")

    rolling = walkforward_splits(
        dates,
        mode="rolling",
        train_span=timedelta(days=30),
        test_span=timedelta(days=10),
        embargo_pct=0.1,
    )
    expanding = walkforward_splits(dates.tolist(), mode="anchored", train_span=30, test_span=10)
    bad = walkforward_splits([object(), object()], train_span=30, test_span=10)

    assert rolling
    assert expanding
    assert bad == []
    assert rolling[0]["mode"] == "rolling"
    assert rolling[0]["embargo_days"] == 3
    assert expanding[1]["train_start"] == dates.min()


def test_validate_no_leakage_direct_temporal_t1_and_error_paths() -> None:
    timeline = pd.date_range("2026-01-01", periods=8, freq="D")

    assert validate_no_leakage(np.array([0, 1, 2]), np.array([3, 4]), timeline) is True
    assert validate_no_leakage(np.array([0, 1, 2]), np.array([2, 3]), timeline) is False
    assert validate_no_leakage(np.array([3]), np.array([2]), timeline) is False

    t1 = pd.Series(timeline + pd.Timedelta(days=5))
    assert validate_no_leakage(np.array([0, 1]), np.array([3, 4]), timeline, t1=t1) is False

    class BadTimeline:
        def __getitem__(self, _idx: Any) -> Any:
            raise TypeError("bad timeline")

    assert validate_no_leakage(np.array([0]), np.array([1]), BadTimeline()) is False
