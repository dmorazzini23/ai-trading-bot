"""Walk-forward evaluation helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from statistics import mean, pstdev
from typing import Any, Callable, Iterable

import pandas as pd


@dataclass(slots=True)
class WalkForwardConfig:
    train_days: int = 180
    test_days: int = 30
    step_days: int = 30
    embargo_days: int = 5


@dataclass(slots=True)
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_folds(timeline: pd.DatetimeIndex, config: WalkForwardConfig) -> list[WalkForwardFold]:
    if timeline.empty:
        return []
    sorted_index = timeline.sort_values()
    folds: list[WalkForwardFold] = []
    cursor = sorted_index.min() + timedelta(days=config.train_days)
    last = sorted_index.max()
    while cursor + timedelta(days=config.test_days) <= last:
        train_end = cursor - timedelta(days=config.embargo_days)
        train_start = train_end - timedelta(days=config.train_days)
        test_start = cursor
        test_end = cursor + timedelta(days=config.test_days)
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
