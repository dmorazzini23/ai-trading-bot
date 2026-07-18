"""Walk-forward evaluation helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from math import ceil
from statistics import mean, pstdev
from typing import Any, Callable, Iterable

import pandas as pd


@dataclass(slots=True)
class WalkForwardConfig:
    train_days: int = 180
    test_days: int = 30
    step_days: int = 30
    embargo_days: int = 5
    purge_days: int = 0


@dataclass(slots=True)
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True, slots=True)
class ContiguousWalkForwardConfig:
    """Configuration for expanding, row-time-ordered research folds.

    Bars are unique timestamps, not dataframe rows, so a multi-symbol timestamp
    cannot be split across train and test. ``label_end_timestamp`` performs the
    horizon purge; the embargo is an additional gap after that purge.
    """

    folds: int = 5
    horizon_bars: int = 1
    embargo_bars: int = 1
    embargo_percent: float = 0.0
    timestamp_col: str = "timestamp"
    label_end_timestamp_col: str = "label_end_timestamp"


@dataclass(frozen=True, slots=True)
class ContiguousWalkForwardFold:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    initial_train_rows: int
    train_rows: int
    test_rows: int
    purged_train_rows: int
    embargoed_train_rows: int
    horizon_bars: int
    embargo_bars: int
    embargo_percent: float
    chronological_non_overlap: bool
    label_purge_ok: bool


def contiguous_walk_forward_splits(
    data: pd.DataFrame,
    config: ContiguousWalkForwardConfig,
) -> list[tuple[ContiguousWalkForwardFold, pd.DataFrame, pd.DataFrame]]:
    """Build expanding contiguous OOS folds with horizon purge and embargo."""

    timestamp_col = str(config.timestamp_col)
    label_end_col = str(config.label_end_timestamp_col)
    if data.empty:
        return []
    missing = [name for name in (timestamp_col, label_end_col) if name not in data.columns]
    if missing:
        raise ValueError(f"Walk-forward data missing required columns: {', '.join(missing)}")

    work = data.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce", utc=True)
    work[label_end_col] = pd.to_datetime(work[label_end_col], errors="coerce", utc=True)
    work = work.dropna(subset=[timestamp_col, label_end_col]).sort_values(
        [timestamp_col], kind="mergesort"
    )
    unique_times = pd.DatetimeIndex(work[timestamp_col].drop_duplicates().sort_values())
    requested_folds = max(1, int(config.folds))
    fold_count = min(requested_folds, max(0, len(unique_times) - 1))
    if fold_count <= 0:
        return []
    test_bars = len(unique_times) // (fold_count + 1)
    if test_bars <= 0:
        return []
    first_test_position = len(unique_times) - (fold_count * test_bars)
    splits: list[tuple[ContiguousWalkForwardFold, pd.DataFrame, pd.DataFrame]] = []
    for fold_offset in range(fold_count):
        test_position = first_test_position + (fold_offset * test_bars)
        test_times = unique_times[test_position : test_position + test_bars]
        if test_times.empty:
            continue
        test_start = pd.Timestamp(test_times[0])
        test_end = pd.Timestamp(test_times[-1])
        initial_train = work.loc[work[timestamp_col] < test_start].copy()
        test = work.loc[
            (work[timestamp_col] >= test_start) & (work[timestamp_col] <= test_end)
        ].copy()
        if initial_train.empty or test.empty:
            continue

        label_end = pd.to_datetime(initial_train[label_end_col], errors="coerce", utc=True)
        purge_keep = label_end < test_start
        purged_rows = int((~purge_keep).sum())
        train = initial_train.loc[purge_keep].copy()

        percent_bars = int(ceil(test_bars * max(0.0, float(config.embargo_percent))))
        embargo_bars = max(0, int(config.embargo_bars), percent_bars)
        embargoed_rows = 0
        if embargo_bars and not train.empty:
            remaining_times = pd.DatetimeIndex(
                train[timestamp_col].drop_duplicates().sort_values()
            )
            embargo_times = remaining_times[-min(embargo_bars, len(remaining_times)) :]
            embargo_mask = train[timestamp_col].isin(embargo_times)
            embargoed_rows = int(embargo_mask.sum())
            train = train.loc[~embargo_mask].copy()
        if train.empty:
            continue

        train_start = pd.Timestamp(train[timestamp_col].min())
        train_end = pd.Timestamp(train[timestamp_col].max())
        max_label_end = pd.Timestamp(train[label_end_col].max())
        chronological_non_overlap = bool(train_end < test_start)
        label_purge_ok = bool(max_label_end < test_start)
        fold = ContiguousWalkForwardFold(
            fold_index=fold_offset + 1,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            initial_train_rows=int(len(initial_train)),
            train_rows=int(len(train)),
            test_rows=int(len(test)),
            purged_train_rows=purged_rows,
            embargoed_train_rows=embargoed_rows,
            horizon_bars=max(1, int(config.horizon_bars)),
            embargo_bars=embargo_bars,
            embargo_percent=max(0.0, float(config.embargo_percent)),
            chronological_non_overlap=chronological_non_overlap,
            label_purge_ok=label_purge_ok,
        )
        if not chronological_non_overlap or not label_purge_ok:
            raise RuntimeError("Walk-forward boundary construction leaked across the OOS fold")
        splits.append((fold, train.copy(), test.copy()))
    return splits


def rolling_folds(timeline: pd.DatetimeIndex, config: WalkForwardConfig) -> list[WalkForwardFold]:
    if timeline.empty:
        return []
    sorted_index = timeline.sort_values()
    folds: list[WalkForwardFold] = []
    cursor = sorted_index.min() + timedelta(days=config.train_days)
    last = sorted_index.max()
    while cursor + timedelta(days=config.test_days) <= last:
        train_end = cursor - timedelta(days=config.embargo_days + config.purge_days)
        train_start = train_end - timedelta(days=config.train_days)
        test_start = cursor
        test_end = cursor + timedelta(days=config.test_days)
        if train_end <= train_start:
            cursor += timedelta(days=config.step_days)
            continue
        folds.append(
            WalkForwardFold(
                train_start=pd.Timestamp(train_start),
                train_end=pd.Timestamp(train_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
            )
        )
        cursor += timedelta(days=config.step_days)
    return folds


def run_walk_forward(
    data: pd.DataFrame,
    *,
    score_fn: Callable[[pd.DataFrame, pd.DataFrame], dict[str, float]],
    config: WalkForwardConfig,
) -> dict[str, Any]:
    if data.empty or "timestamp" not in data.columns:
        return {"folds": [], "distribution": {}}
    timeline = pd.DatetimeIndex(pd.to_datetime(data["timestamp"], utc=True))
    folds = rolling_folds(timeline, config)
    fold_metrics: list[dict[str, float]] = []
    for fold in folds:
        train_mask = (timeline >= fold.train_start) & (timeline < fold.train_end)
        test_mask = (timeline >= fold.test_start) & (timeline < fold.test_end)
        train_df = data.loc[train_mask]
        test_df = data.loc[test_mask]
        if train_df.empty or test_df.empty:
            continue
        fold_metrics.append(score_fn(train_df, test_df))

    metric_keys = ("post_cost_return", "turnover", "drawdown", "hit_rate")
    distribution: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = [float(metrics[key]) for metrics in fold_metrics if key in metrics]
        if not values:
            continue
        distribution[key] = {
            "mean": float(mean(values)),
            "std": float(pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return {
        "folds": [asdict(fold) for fold in folds],
        "distribution": distribution,
        "fold_count": len(fold_metrics),
    }
