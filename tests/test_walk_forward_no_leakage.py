from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from ai_trading.research.leakage_tests import run_leakage_guards
from ai_trading.research.walk_forward import (
    ContiguousWalkForwardConfig,
    WalkForwardConfig,
    contiguous_walk_forward_splits,
    run_walk_forward,
)


def _score(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    return {
        "post_cost_return": float(test_df["ret"].mean()),
        "turnover": float(test_df["turnover"].mean()),
        "drawdown": float(test_df["drawdown"].max()),
        "hit_rate": float((test_df["ret"] > 0).mean()),
    }


def test_walk_forward_and_no_leakage() -> None:
    ts = pd.date_range("2024-01-01", periods=420, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "ret": [0.001 if idx % 2 == 0 else -0.0005 for idx in range(len(ts))],
            "turnover": [1000.0 + idx for idx in range(len(ts))],
            "drawdown": [0.01 for _ in range(len(ts))],
        }
    )
    result = run_walk_forward(
        df,
        score_fn=_score,
        config=WalkForwardConfig(
            train_days=180,
            test_days=30,
            step_days=30,
            embargo_days=5,
            purge_days=2,
        ),
    )
    assert result["fold_count"] > 0
    assert "post_cost_return" in result["distribution"]
    first_fold = result["folds"][0]
    train_end = pd.Timestamp(first_fold["train_end"])
    test_start = pd.Timestamp(first_fold["test_start"])
    assert test_start - train_end >= timedelta(days=7)

    run_leakage_guards(
        feature_timestamps=ts[:-1],
        label_timestamps=ts[:-1],
        train_label_times=ts[:200],
        test_label_times=ts[250:300],
        horizon_days=5,
        embargo_days=5,
    )


def test_leakage_guard_fails_on_future_features() -> None:
    ts = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    with pytest.raises(AssertionError):
        run_leakage_guards(
            feature_timestamps=[ts[1]],
            label_timestamps=[ts[0]],
            train_label_times=ts[:5],
            test_label_times=ts[8:],
            horizon_days=1,
            embargo_days=1,
        )


def test_leakage_guard_does_not_double_count_label_horizon() -> None:
    ts = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")

    run_leakage_guards(
        feature_timestamps=ts[:4],
        label_timestamps=ts[5:9],
        train_label_times=[ts[8]],
        test_label_times=[ts[10]],
        horizon_days=5,
        embargo_days=2,
    )

    with pytest.raises(AssertionError):
        run_leakage_guards(
            feature_timestamps=ts[:4],
            label_timestamps=ts[5:9],
            train_label_times=[ts[8]],
            test_label_times=[ts[9]],
            horizon_days=5,
            embargo_days=2,
        )


def test_contiguous_walk_forward_purges_labels_and_embargoes_timestamp_bars() -> None:
    timestamps = pd.date_range("2026-01-02 14:30:00+00:00", periods=36, freq="min")
    rows = [
        {
            "timestamp": timestamp,
            "label_end_timestamp": timestamp + pd.Timedelta(minutes=3),
            "symbol": symbol,
            "value": idx,
        }
        for idx, timestamp in enumerate(timestamps)
        for symbol in ("AAPL", "MSFT")
    ]

    splits = contiguous_walk_forward_splits(
        pd.DataFrame(rows),
        ContiguousWalkForwardConfig(
            folds=5,
            horizon_bars=3,
            embargo_bars=2,
            embargo_percent=0.10,
        ),
    )

    assert len(splits) == 5
    previous_test_end = None
    for fold, train, test in splits:
        assert fold.chronological_non_overlap is True
        assert fold.label_purge_ok is True
        assert fold.purged_train_rows == 6
        assert fold.embargoed_train_rows == 4
        assert fold.embargo_bars == 2
        assert train["label_end_timestamp"].max() < test["timestamp"].min()
        assert train["timestamp"].max() < test["timestamp"].min()
        assert set(test.groupby("timestamp")["symbol"].nunique()) == {2}
        if previous_test_end is not None:
            assert pd.Timestamp(previous_test_end) < test["timestamp"].min()
        previous_test_end = test["timestamp"].max()
