from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import joblib

from ai_trading.models.artifacts import verify_artifact
from ai_trading.tools.train_replay_aligned_model import (
    REPLAY_ALIGNED_FEATURE_COLUMNS,
    build_training_dataset,
    train_replay_aligned_model,
)


def _write_cycle_bars(csv_path: Path, *, periods: int = 260, phase: float = 0.0) -> None:
    idx = pd.date_range("2026-01-02 14:30:00+00:00", periods=periods, freq="min")
    x = np.linspace(phase, phase + 30.0, periods)
    close = 100.0 + 2.4 * np.sin(x) + 0.4 * np.sin(x * 0.33)
    open_ = close + 0.03 * np.cos(x)
    high = np.maximum(open_, close) + 0.12
    low = np.minimum(open_, close) - 0.12
    volume = 12_000.0 + 500.0 * np.cos(x * 0.5)
    pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    ).to_csv(csv_path, index=False)


def test_build_training_dataset_uses_future_net_markout_target(tmp_path: Path) -> None:
    _write_cycle_bars(tmp_path / "AAPL.csv", periods=120)

    dataset = build_training_dataset(
        data_dir=tmp_path,
        horizon_bars=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_net_edge_bps=0.0,
    )

    assert not dataset.empty
    assert set(REPLAY_ALIGNED_FEATURE_COLUMNS).issubset(dataset.columns)
    assert {
        "symbol",
        "timestamp",
        "session_regime",
        "gross_long_bps",
        "net_long_bps",
        "target",
    }.issubset(dataset.columns)
    assert dataset["target"].nunique() == 2
    assert set(dataset["target"].unique()).issubset({0, 1})
    assert bool((dataset["target"] == (dataset["net_long_bps"] > 0.0).astype(int)).all())


def test_train_replay_aligned_model_writes_verified_artifact_and_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_cycle_bars(data_dir / "AAPL.csv", phase=0.0)
    _write_cycle_bars(data_dir / "MSFT.csv", phase=1.7)
    args = argparse.Namespace(
        data_dir=data_dir,
        symbols="",
        timestamp_col="timestamp",
        output_dir=output_dir,
        model_name="replay_aligned_test",
        model_type="logistic",
        horizon_bars=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_net_edge_bps=0.0,
        train_fraction=0.65,
        edge_global_threshold=0.66,
        random_state=7,
    )

    report = train_replay_aligned_model(args)

    model_path = Path(cast(str, report["model_path"]))
    manifest_path = Path(cast(str, report["manifest_path"]))
    report_path = Path(cast(str, report["report_path"]))
    assert model_path.is_file()
    assert manifest_path.is_file()
    assert report_path.is_file()
    assert verify_artifact(model_path=model_path, manifest_path=manifest_path) == (True, "OK")
    loaded = joblib.load(model_path)
    assert getattr(loaded, "edge_global_threshold_") == 0.66
    assert getattr(loaded, "edge_thresholds_by_regime_")

    persisted = cast(dict[str, Any], json.loads(report_path.read_text(encoding="utf-8")))
    assert persisted["config"]["edge_global_threshold"] == 0.66
    assert persisted["thresholds_by_regime"]
    assert persisted["threshold_sweep_by_regime"]
    assert persisted["dataset"]["symbols"] == 2
    assert persisted["dataset"]["train_rows"] > 0
    assert persisted["dataset"]["validation_rows"] > 0
    assert persisted["validation"]["rows"] == persisted["dataset"]["validation_rows"]
    assert persisted["threshold_sweep"]
    assert persisted["recommendation"] == "evaluate_candidate_with_offline_replay_before_promotion"
