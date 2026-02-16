"""Leakage guardrails for walk-forward and temporal datasets."""
from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import pandas as pd


def assert_feature_times_not_future(
    feature_timestamps: Iterable[pd.Timestamp],
    label_timestamps: Iterable[pd.Timestamp],
) -> None:
    feature_list = [pd.Timestamp(ts) for ts in feature_timestamps]
    label_list = [pd.Timestamp(ts) for ts in label_timestamps]
    if len(feature_list) != len(label_list):
        raise AssertionError("Feature/label timestamp length mismatch")
    for idx, (feature_ts, label_ts) in enumerate(zip(feature_list, label_list, strict=True)):
        if feature_ts > label_ts:
            raise AssertionError(
                f"Feature timestamp leaks future label at index {idx}: {feature_ts} > {label_ts}"
            )


def assert_no_horizon_overlap(
    train_label_times: Iterable[pd.Timestamp],
    test_label_times: Iterable[pd.Timestamp],
    *,
    horizon: timedelta,
    embargo: timedelta,
) -> None:
    train = sorted(pd.Timestamp(ts) for ts in train_label_times)
    test = sorted(pd.Timestamp(ts) for ts in test_label_times)
    if not train or not test:
        return
    train_end = train[-1] + horizon + embargo
    test_start = test[0]
    if train_end > test_start:
        raise AssertionError(
            f"Overlapping label horizon detected: train_end={train_end} test_start={test_start}"
        )


def run_leakage_guards(
    *,
    feature_timestamps: Iterable[pd.Timestamp],
    label_timestamps: Iterable[pd.Timestamp],
    train_label_times: Iterable[pd.Timestamp],
    test_label_times: Iterable[pd.Timestamp],
    horizon_days: int,
    embargo_days: int,
) -> None:
    assert_feature_times_not_future(feature_timestamps, label_timestamps)
    assert_no_horizon_overlap(
        train_label_times=train_label_times,
        test_label_times=test_label_times,
        horizon=timedelta(days=int(horizon_days)),
        embargo=timedelta(days=int(embargo_days)),
    )
