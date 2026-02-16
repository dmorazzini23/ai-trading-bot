from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from ai_trading.research.leakage_tests import run_leakage_guards
from ai_trading.research.walk_forward import WalkForwardConfig, run_walk_forward


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
        config=WalkForwardConfig(train_days=180, test_days=30, step_days=30, embargo_days=5),
    )
    assert result["fold_count"] > 0
    assert "post_cost_return" in result["distribution"]

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
