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
    _best_thresholds_by_regime,
    _split_train_validation_with_purge,
    _threshold_report,
    train_replay_aligned_model,
)
from ai_trading.tools import train_replay_aligned_model as trainer


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
    assert "label_end_timestamp" in dataset.columns
    assert bool(dataset["label_end_timestamp"].ge(dataset["timestamp"]).all())


def test_split_train_validation_purges_overlapping_label_horizon() -> None:
    timestamps = pd.date_range("2026-01-02 14:30:00+00:00", periods=10, freq="min")
    dataset = pd.DataFrame(
        {
            "timestamp": timestamps,
            "label_end_timestamp": timestamps + pd.Timedelta(minutes=3),
            "symbol": ["AAPL"] * 10,
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    train, validation, diagnostics = _split_train_validation_with_purge(
        dataset,
        train_fraction=0.6,
        horizon_bars=3,
    )

    assert diagnostics["purged_train_rows"] == 3
    assert not train.empty
    assert not validation.empty
    assert train["target"].nunique() == 2
    assert validation["target"].nunique() == 2
    assert train["label_end_timestamp"].max() < validation["timestamp"].min()


def test_threshold_report_does_not_turn_long_probabilities_into_shorts() -> None:
    dataset = pd.DataFrame({"net_long_bps": [-10.0, -20.0, -30.0]})
    long_only = _threshold_report(dataset, np.asarray([0.1, 0.2, 0.3]))
    directional = _threshold_report(
        dataset,
        np.asarray([0.1, 0.2, 0.3]),
        allow_short_labels=True,
    )

    assert all(row["candidates"] == 0 for row in long_only)
    assert any(
        row["candidates"] > 0 and float(row["total_net_markout_bps"]) > 0.0
        for row in directional
    )


def test_best_thresholds_ignore_zero_candidate_rows() -> None:
    reports = {
        "regular": [
            {
                "confidence_threshold": 0.95,
                "entry_score_threshold": 0.20,
                "candidates": 0,
                "mean_net_markout_bps": None,
            },
            {
                "confidence_threshold": 0.58,
                "entry_score_threshold": 0.05,
                "candidates": 7,
                "mean_net_markout_bps": 2.5,
            },
        ]
    }

    assert _best_thresholds_by_regime(reports) == {"regular": 0.58}


def test_build_training_dataset_supports_risk_adjusted_excursion_labels(tmp_path: Path) -> None:
    _write_cycle_bars(tmp_path / "AAPL.csv", periods=120)

    dataset = build_training_dataset(
        data_dir=tmp_path,
        horizon_bars=5,
        label_objective="risk_adjusted",
        fee_bps=0.0,
        slippage_bps=0.0,
        min_net_edge_bps=0.0,
    )

    assert not dataset.empty
    assert {
        "max_adverse_excursion_bps",
        "max_favorable_excursion_bps",
        "risk_adjusted_net_bps",
        "label_score_bps",
        "label_objective",
    }.issubset(dataset.columns)
    assert set(dataset["label_objective"].unique()) == {"risk_adjusted"}
    assert not bool(dataset["label_score_bps"].equals(dataset["net_long_bps"]))
    assert bool(
        (dataset["target"] == (dataset["label_score_bps"] > 0.0).astype(int)).all()
    )


def test_build_training_dataset_can_use_live_cost_model_labels(tmp_path: Path) -> None:
    _write_cycle_bars(tmp_path / "AAPL.csv", periods=120)
    live_cost_model = {
        "artifact_type": "live_cost_model",
        "generated_at": "2026-01-02T21:00:00Z",
        "status": {"available": True, "status": "ready"},
        "by_symbol_side_session": [
            {
                "symbol": "AAPL",
                "side": side,
                "session_regime": regime,
                "sample_count": 10,
                "sufficient_samples": True,
                "p90_adverse_slippage_bps": 25.0,
            }
            for side in ("buy", "sell")
            for regime in ("opening", "midday", "closing")
        ],
    }
    live_cost_path = tmp_path / "live_cost_model_latest.json"
    live_cost_path.write_text(json.dumps(live_cost_model), encoding="utf-8")

    static_dataset = build_training_dataset(
        data_dir=tmp_path,
        horizon_bars=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_net_edge_bps=0.0,
    )
    from ai_trading.tools.offline_replay import _load_live_cost_replay_model

    cost_model = _load_live_cost_replay_model(
        argparse.Namespace(live_cost_model_json=live_cost_path, use_live_cost_model=False)
    )
    cost_dataset = build_training_dataset(
        data_dir=tmp_path,
        horizon_bars=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_net_edge_bps=0.0,
        live_cost_model=cost_model,
    )

    assert cost_model is not None
    assert cost_dataset["round_trip_cost_bps"].mean() == 50.0
    assert cost_dataset["net_long_bps"].mean() < static_dataset["net_long_bps"].mean()


def test_build_training_dataset_reuses_feature_cache(tmp_path: Path, monkeypatch) -> None:
    _write_cycle_bars(tmp_path / "AAPL.csv", periods=120)
    calls = {"count": 0}
    original = trainer._feature_frame

    def _counting_feature_frame(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
        calls["count"] += 1
        return original(frame, symbol=symbol)

    monkeypatch.setattr(trainer, "_feature_frame", _counting_feature_frame)

    first = build_training_dataset(
        data_dir=tmp_path,
        horizon_bars=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        training_cache_dir=tmp_path / "cache",
    )
    second = build_training_dataset(
        data_dir=tmp_path,
        horizon_bars=5,
        fee_bps=0.0,
        slippage_bps=0.0,
        training_cache_dir=tmp_path / "cache",
    )

    assert not first.empty
    assert not second.empty
    assert calls["count"] == 1


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
        label_objective="mae_mfe",
        fee_bps=0.0,
        slippage_bps=0.0,
        min_net_edge_bps=0.0,
        train_fraction=0.65,
        edge_global_threshold=0.66,
        random_state=7,
        training_cache=True,
        training_cache_dir=tmp_path / "cache",
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
    assert persisted["config"]["label_objective"] == "mae_mfe"
    assert persisted["dataset"]["mean_label_score_bps"] != persisted["dataset"]["mean_round_trip_cost_bps"]
    assert persisted["thresholds_by_regime"]
    assert persisted["threshold_sweep_by_regime"]
    assert persisted["dataset"]["symbols"] == 2
    assert persisted["dataset"]["train_rows"] > 0
    assert persisted["dataset"]["validation_rows"] > 0
    assert persisted["validation"]["rows"] == persisted["dataset"]["validation_rows"]
    assert persisted["threshold_sweep"]
    assert persisted["recommendation"] == "evaluate_candidate_with_offline_replay_before_promotion"
