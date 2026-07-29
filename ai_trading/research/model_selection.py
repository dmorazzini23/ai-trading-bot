"""Leakage-safe chronological model-selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import pandas as pd


@dataclass(frozen=True, slots=True)
class ChronologicalPartition:
    development: pd.DataFrame
    holdout: pd.DataFrame
    holdout_start: pd.Timestamp
    initial_development_rows: int
    purged_development_rows: int
    embargoed_development_rows: int
    embargo_bars: int


@dataclass(frozen=True, slots=True)
class NestedSelectionPartition:
    fit: pd.DataFrame
    selection: pd.DataFrame
    selection_start: pd.Timestamp
    initial_fit_rows: int
    purged_fit_rows: int
    embargoed_fit_rows: int
    embargo_bars: int


def _chronological_work(
    data: pd.DataFrame,
    *,
    timestamp_col: str,
    label_end_timestamp_col: str,
) -> pd.DataFrame:
    missing = [
        name
        for name in (timestamp_col, label_end_timestamp_col)
        if name not in data.columns
    ]
    if missing:
        raise ValueError(
            "Chronological partition data missing required columns: "
            + ", ".join(missing)
        )
    work = data.copy()
    work[timestamp_col] = pd.to_datetime(
        work[timestamp_col], errors="coerce", utc=True
    )
    work[label_end_timestamp_col] = pd.to_datetime(
        work[label_end_timestamp_col], errors="coerce", utc=True
    )
    return (
        work.dropna(subset=[timestamp_col, label_end_timestamp_col])
        .sort_values([timestamp_col], kind="mergesort")
        .reset_index(drop=True)
    )


def _purge_and_embargo(
    initial_fit: pd.DataFrame,
    *,
    boundary: pd.Timestamp,
    embargo_bars: int,
    timestamp_col: str,
    label_end_timestamp_col: str,
) -> tuple[pd.DataFrame, int, int]:
    label_end = pd.to_datetime(
        initial_fit[label_end_timestamp_col], errors="coerce", utc=True
    )
    purge_keep = label_end < boundary
    purged_rows = int((~purge_keep).sum())
    fit = initial_fit.loc[purge_keep].copy()
    embargoed_rows = 0
    if embargo_bars > 0 and not fit.empty:
        fit_times = pd.DatetimeIndex(
            fit[timestamp_col].drop_duplicates().sort_values()
        )
        embargo_times = fit_times[-min(embargo_bars, len(fit_times)) :]
        embargo_mask = fit[timestamp_col].isin(embargo_times)
        embargoed_rows = int(embargo_mask.sum())
        fit = fit.loc[~embargo_mask].copy()
    return fit, purged_rows, embargoed_rows


def chronological_development_holdout(
    data: pd.DataFrame,
    *,
    train_fraction: float,
    embargo_bars: int,
    timestamp_col: str = "timestamp",
    label_end_timestamp_col: str = "label_end_timestamp",
) -> ChronologicalPartition:
    """Reserve a final unique-timestamp holdout and purge its development edge."""

    work = _chronological_work(
        data,
        timestamp_col=timestamp_col,
        label_end_timestamp_col=label_end_timestamp_col,
    )
    unique_times = pd.DatetimeIndex(
        work[timestamp_col].drop_duplicates().sort_values()
    )
    if len(unique_times) < 3:
        raise ValueError("Chronological holdout requires at least three timestamps")
    bounded_fraction = min(0.90, max(0.50, float(train_fraction)))
    boundary_position = max(
        1,
        min(len(unique_times) - 1, int(len(unique_times) * bounded_fraction)),
    )
    holdout_start = pd.Timestamp(unique_times[boundary_position])
    initial_development = work.loc[work[timestamp_col] < holdout_start].copy()
    holdout = work.loc[work[timestamp_col] >= holdout_start].copy()
    development, purged_rows, embargoed_rows = _purge_and_embargo(
        initial_development,
        boundary=holdout_start,
        embargo_bars=max(0, int(embargo_bars)),
        timestamp_col=timestamp_col,
        label_end_timestamp_col=label_end_timestamp_col,
    )
    if development.empty or holdout.empty:
        raise ValueError("Chronological holdout produced an empty partition")
    return ChronologicalPartition(
        development=development,
        holdout=holdout,
        holdout_start=holdout_start,
        initial_development_rows=int(len(initial_development)),
        purged_development_rows=purged_rows,
        embargoed_development_rows=embargoed_rows,
        embargo_bars=max(0, int(embargo_bars)),
    )


def trailing_nested_selection_split(
    data: pd.DataFrame,
    *,
    validation_fraction: float,
    embargo_bars: int,
    timestamp_col: str = "timestamp",
    label_end_timestamp_col: str = "label_end_timestamp",
) -> NestedSelectionPartition:
    """Carve a purged trailing validation block from an outer training fold."""

    work = _chronological_work(
        data,
        timestamp_col=timestamp_col,
        label_end_timestamp_col=label_end_timestamp_col,
    )
    unique_times = pd.DatetimeIndex(
        work[timestamp_col].drop_duplicates().sort_values()
    )
    if len(unique_times) < 3:
        raise ValueError("Nested threshold selection requires at least three timestamps")
    bounded_fraction = min(0.40, max(0.10, float(validation_fraction)))
    validation_bars = max(1, int(ceil(len(unique_times) * bounded_fraction)))
    boundary_position = max(1, len(unique_times) - validation_bars)
    selection_start = pd.Timestamp(unique_times[boundary_position])
    initial_fit = work.loc[work[timestamp_col] < selection_start].copy()
    selection = work.loc[work[timestamp_col] >= selection_start].copy()
    fit, purged_rows, embargoed_rows = _purge_and_embargo(
        initial_fit,
        boundary=selection_start,
        embargo_bars=max(0, int(embargo_bars)),
        timestamp_col=timestamp_col,
        label_end_timestamp_col=label_end_timestamp_col,
    )
    if fit.empty or selection.empty:
        raise ValueError("Nested threshold selection produced an empty partition")
    return NestedSelectionPartition(
        fit=fit,
        selection=selection,
        selection_start=selection_start,
        initial_fit_rows=int(len(initial_fit)),
        purged_fit_rows=purged_rows,
        embargoed_fit_rows=embargoed_rows,
        embargo_bars=max(0, int(embargo_bars)),
    )
