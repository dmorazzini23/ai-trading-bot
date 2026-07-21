from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import joblib
import pytest

from ai_trading.models.artifacts import verify_artifact
from ai_trading.models.contracts import infer_day_sleeve_regimes
from ai_trading.tools.train_replay_aligned_model import (
    REPLAY_ALIGNED_FEATURE_COLUMNS,
    _symbol_feature_cache_key,
    build_training_dataset,
    _best_thresholds_by_regime,
    _fold_market_regimes,
    _split_train_validation_with_purge,
    _threshold_report,
    _run_fold_local_walk_forward,
    _walk_forward_qualification,
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


def _write_governed_acquisition(tmp_path: Path) -> Path:
    data_dir = tmp_path / "governed-bars"
    data_dir.mkdir()
    _write_cycle_bars(data_dir / "AAPL.csv", phase=0.0)
    _write_cycle_bars(data_dir / "MSFT.csv", phase=1.7)
    authority = {
        "research_only": True,
        "evidence_type": "historical_research",
        "promotion_eligible": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "runtime_fill_authority": False,
    }
    symbol_rows = []
    for symbol in ("AAPL", "MSFT"):
        csv_path = data_dir / f"{symbol}.csv"
        symbol_rows.append(
            {
                "symbol": symbol,
                "csv_path": str(csv_path),
                "row_count": 260,
                "content_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "quality_passed": True,
                "completeness": {"missing_ratio": 0.0, "quality_passed": True},
            }
        )
    manifest_path = data_dir / "dataset.provenance.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_cache_key": "cache-key",
                "dataset_dir": str(data_dir),
                "dataset_identity": {
                    "provider": "alpaca",
                    "feed": "iex",
                    "adjustment": "raw",
                    "timeframe": "1Min",
                },
                "symbols": symbol_rows,
                "quality_passed": True,
                "authority": authority,
            }
        ),
        encoding="utf-8",
    )
    acquisition_path = tmp_path / "historical_backfill.json"
    acquisition_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(data_dir),
                "manifest_path": str(manifest_path),
                "cache_key": "cache-key",
                "symbols": symbol_rows,
                "quality_passed": True,
                "authority": authority,
            }
        ),
        encoding="utf-8",
    )
    return acquisition_path


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
    assert diagnostics["embargoed_train_rows"] == 0
    assert diagnostics["embargo_bars"] == 0
    assert not train.empty
    assert not validation.empty
    assert train["target"].nunique() == 2
    assert validation["target"].nunique() == 2
    assert train["label_end_timestamp"].max() < validation["timestamp"].min()


def test_split_train_validation_can_apply_extra_embargo() -> None:
    timestamps = pd.date_range("2026-01-02 14:30:00+00:00", periods=12, freq="min")
    dataset = pd.DataFrame(
        {
            "timestamp": timestamps,
            "label_end_timestamp": timestamps + pd.Timedelta(minutes=1),
            "symbol": ["AAPL"] * 12,
            "target": [0, 1] * 6,
        }
    )

    train, validation, diagnostics = _split_train_validation_with_purge(
        dataset,
        train_fraction=0.75,
        horizon_bars=1,
        embargo_bars=2,
    )

    assert diagnostics["embargoed_train_rows"] == 2
    assert len(train) == 6
    assert not validation.empty
    assert train["timestamp"].max() < validation["timestamp"].min()


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


def test_build_training_dataset_standard_horizons_net_edge_and_spread_labels(tmp_path: Path) -> None:
    _write_cycle_bars(tmp_path / "AAPL.csv", periods=140)
    frame = pd.read_csv(tmp_path / "AAPL.csv")
    frame["spread_bps"] = 3.0
    frame.to_csv(tmp_path / "AAPL.csv", index=False)

    row_counts: dict[int, int] = {}
    for horizon in (1, 3, 5, 15):
        net_dataset = build_training_dataset(
            data_dir=tmp_path,
            horizon_bars=horizon,
            label_objective="net_markout",
            fee_bps=1.0,
            slippage_bps=2.0,
            min_net_edge_bps=0.0,
            training_cache_dir=tmp_path / "cache",
        )
        spread_dataset = build_training_dataset(
            data_dir=tmp_path,
            horizon_bars=horizon,
            label_objective="spread_adjusted",
            fee_bps=1.0,
            slippage_bps=2.0,
            min_net_edge_bps=0.0,
            training_cache_dir=tmp_path / "cache",
        )
        row_counts[horizon] = len(net_dataset)
        assert not net_dataset.empty
        assert set(net_dataset["label_objective"].unique()) == {"net_markout"}
        assert set(spread_dataset["label_objective"].unique()) == {"spread_adjusted"}
        assert bool(net_dataset["label_end_timestamp"].gt(net_dataset["timestamp"]).all())
        assert bool((net_dataset["target"] == (net_dataset["net_long_bps"] > 0.0).astype(int)).all())
        assert net_dataset["round_trip_cost_bps"].mean() == pytest.approx(9.0)
        assert spread_dataset["label_score_bps"].mean() > net_dataset["label_score_bps"].mean()
        assert {"max_adverse_excursion_bps", "max_favorable_excursion_bps"} <= set(net_dataset.columns)

    assert row_counts[1] > row_counts[3] > row_counts[5] > row_counts[15]


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


def test_feature_cache_key_uses_content_hash_not_only_size_and_mtime(tmp_path: Path) -> None:
    csv_path = tmp_path / "AAPL.csv"
    csv_path.write_text("timestamp,close\n2026-01-02T14:30:00Z,100\n", encoding="utf-8")
    first_mtime_ns = csv_path.stat().st_mtime_ns
    first = _symbol_feature_cache_key(csv_path, timestamp_col="timestamp", symbol="AAPL")

    csv_path.write_text("timestamp,close\n2026-01-02T14:30:00Z,101\n", encoding="utf-8")
    second = _symbol_feature_cache_key(csv_path, timestamp_col="timestamp", symbol="AAPL")

    assert csv_path.stat().st_size == len("timestamp,close\n2026-01-02T14:30:00Z,101\n")
    assert first_mtime_ns != 0
    assert first != second


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
    assert persisted["authority"]["timestamp_authoritative"] is True
    assert persisted["authority"]["research_synthetic"] is False
    assert persisted["authority"]["promotion_authority"] is False
    assert persisted["dataset"]["load_reports"]["AAPL"]["timestamp_authoritative"] is True
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
    assert persisted["recommendation"] == "evaluate_candidate_in_shadow_with_governed_offline_replay"
    assert persisted["governance_status"] == "shadow"
    assert persisted["promotion_authority"] is False
    assert persisted["live_money_authority"] is False
    walk_forward = persisted["walk_forward"]
    assert walk_forward["evaluation_type"] == "expanding_contiguous_walk_forward"
    assert walk_forward["market_regime_classifier"] == "day_sleeve_past_only_v1"
    assert walk_forward["fold_local_fitting"] is True
    assert len(walk_forward["folds"]) == 5
    for fold in walk_forward["folds"]:
        assert fold["fit_scope"] == "fold_train_only"
        assert fold["threshold_scope"] == "fold_train_only"
        assert fold["chronological_non_overlap"] is True
        assert fold["label_purge_ok"] is True
        assert fold["embargo_bars"] >= 1
        assert fold["cost_model"]["version"] == "static_fee_spread_slippage_v1"
        assert "mean_post_cost_net_edge_bps" in fold
        assert "profit_factor" in fold
        assert "max_drawdown_bps" in fold
        assert "hit_rate" in fold
        assert "ranking_separation" in fold
        assert fold["by_market_regime"]
    aggregate = walk_forward["aggregate"]
    assert aggregate["governance_status"] == "shadow"
    assert aggregate["promotion_authority"] is False
    assert aggregate["live_money_authority"] is False
    assert "profitable_fold_ratio" in aggregate
    assert "stability_score" in aggregate
    assert "worst_fold" in aggregate
    aggregate_regimes = walk_forward["by_market_regime"]
    assert aggregate_regimes
    assert sum(row["trades"] for row in aggregate_regimes.values()) == aggregate["trades"]
    for regime, row in aggregate_regimes.items():
        assert regime in {"sideways", "uptrend", "downtrend", "volatile"}
        assert row["support"] == row["trades"]
        assert "mean_post_cost_net_edge_bps" in row
        assert "total_post_cost_net_edge_bps" in row
        assert "profit_factor" in row
        assert "max_drawdown_bps" in row
        assert "hit_rate" in row
        assert "ranking_separation" in row
        assert row["shadow_disposition"] in {"observe", "abstain"}
        assert row["promotion_authority"] is False
        assert row["live_money_authority"] is False
    assert persisted["feature_importance"]
    assert persisted["feature_importance"][0]["feature"] in REPLAY_ALIGNED_FEATURE_COLUMNS
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["metadata"]["feature_importance"]
    policy = persisted["market_regime_policy"]
    assert policy["market_regime_classifier"] == "day_sleeve_past_only_v1"
    assert policy["governance_status"] == "shadow"
    assert policy["promotion_authority"] is False
    assert set(policy["allowed_regimes"]).isdisjoint(policy["abstained_regimes"])
    assert manifest_payload["metadata"]["market_regime_policy"] == policy


def test_train_replay_aligned_model_persists_governed_acquisition_lineage(
    tmp_path: Path,
) -> None:
    acquisition_path = _write_governed_acquisition(tmp_path)
    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        data_dir=None,
        acquisition_manifest_json=acquisition_path,
        symbols="AAPL,MSFT",
        timestamp_col="timestamp",
        output_dir=output_dir,
        model_name="governed_historical",
        model_type="logistic",
        horizon_bars=1,
        label_objective="risk_adjusted",
        fee_bps=1.0,
        slippage_bps=2.0,
        min_net_edge_bps=0.0,
        train_fraction=0.65,
        edge_global_threshold=0.66,
        random_state=7,
        training_cache=True,
        training_cache_dir=tmp_path / "cache",
    )

    report = train_replay_aligned_model(args)
    persisted = cast(
        dict[str, Any],
        json.loads(Path(cast(str, report["report_path"])).read_text(encoding="utf-8")),
    )
    manifest = json.loads(
        Path(cast(str, report["manifest_path"])).read_text(encoding="utf-8")
    )

    assert persisted["status"] == "complete"
    assert persisted["authority"]["evidence_type"] == "historical_research"
    assert persisted["authority"]["promotion_eligible"] is False
    assert persisted["authority"]["runtime_fill_authority"] is False
    assert persisted["acquisition"]["mode"] == "governed_historical_backfill"
    assert persisted["acquisition"]["quality_passed"] is True
    assert persisted["acquisition"]["symbols"] == ["AAPL", "MSFT"]
    assert persisted["acquisition"]["completeness"]["AAPL"]["missing_ratio"] == 0.0
    assert persisted["dataset"]["dataset_hash"] == persisted["acquisition"]["dataset_hash"]
    assert persisted["dataset"]["mean_round_trip_cost_bps"] > 0.0
    assert persisted["dataset"]["split_purge"]["folds"]
    assert all(
        fold["embargo_bars"] >= 1
        for fold in persisted["dataset"]["split_purge"]["folds"]
    )
    assert manifest["metadata"]["acquisition"]["dataset_hash"] == persisted[
        "dataset"
    ]["dataset_hash"]
    assert manifest["metadata"]["promotion_authority"] is False


def test_governed_acquisition_rejects_failed_quality_before_training(
    tmp_path: Path,
) -> None:
    acquisition_path = _write_governed_acquisition(tmp_path)
    payload = json.loads(acquisition_path.read_text(encoding="utf-8"))
    payload["quality_passed"] = False
    acquisition_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="completeness quality gate"):
        trainer._resolve_training_input(
            argparse.Namespace(
                data_dir=None,
                acquisition_manifest_json=acquisition_path,
                symbols="AAPL,MSFT",
            )
        )


def test_governed_acquisition_rejects_content_hash_mismatch(
    tmp_path: Path,
) -> None:
    acquisition_path = _write_governed_acquisition(tmp_path)
    payload = json.loads(acquisition_path.read_text(encoding="utf-8"))
    payload["symbols"][0]["content_sha256"] = "0" * 64
    acquisition_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch for AAPL"):
        trainer._resolve_training_input(
            argparse.Namespace(
                data_dir=None,
                acquisition_manifest_json=acquisition_path,
                symbols="AAPL,MSFT",
            )
        )


def test_train_replay_aligned_model_records_requested_but_unusable_live_cost(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_cycle_bars(data_dir / "AAPL.csv", phase=0.0)
    _write_cycle_bars(data_dir / "MSFT.csv", phase=1.7)
    live_cost_path = tmp_path / "live_cost_model_latest.json"
    live_cost_path.write_text(
        json.dumps(
            {
                "artifact_type": "live_cost_model",
                "generated_at": "2026-01-02T21:00:00Z",
                "status": {
                    "available": True,
                    "status": "warming_up",
                    "reason": "insufficient_samples",
                },
                "by_symbol_side_session": [],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        data_dir=data_dir,
        symbols="",
        timestamp_col="timestamp",
        output_dir=output_dir,
        model_name="replay_aligned_cost_probe",
        model_type="logistic",
        horizon_bars=1,
        label_objective="risk_adjusted",
        fee_bps=0.0,
        slippage_bps=2.0,
        min_net_edge_bps=0.0,
        train_fraction=0.65,
        edge_global_threshold=0.66,
        random_state=7,
        training_cache=True,
        training_cache_dir=tmp_path / "cache",
        live_cost_model_json=live_cost_path,
        use_live_cost_model=True,
    )

    report = train_replay_aligned_model(args)

    live_cost = report["live_cost_model"]
    assert live_cost["requested"] is True
    assert live_cost["usable"] is False
    assert live_cost["enabled"] is False
    assert live_cost["reason"] == "status_warming_up"
    assert report["config"]["live_cost_model_requested"] is True
    assert report["config"]["live_cost_model_usable"] is False


def test_walk_forward_negative_or_insufficient_evidence_stays_unqualified() -> None:
    folds = [
        {
            "fold_index": 1,
            "trades": 20,
            "mean_post_cost_net_edge_bps": -2.0,
            "total_post_cost_net_edge_bps": -40.0,
            "ranking_separation": {"high_minus_low_net_edge_bps": -0.5},
        },
        {
            "fold_index": 2,
            "trades": 20,
            "mean_post_cost_net_edge_bps": 1.0,
            "total_post_cost_net_edge_bps": 20.0,
            "ranking_separation": {"high_minus_low_net_edge_bps": 0.1},
        },
    ]

    aggregate = _walk_forward_qualification(
        folds,
        required_folds=5,
        min_trades=250,
        min_mean_net_edge_bps=0.0,
        min_profitable_fold_ratio=0.60,
        min_ranking_separation_bps=0.0,
    )

    assert aggregate["evidence_qualified"] is False
    assert aggregate["governance_status"] == "shadow"
    assert aggregate["promotion_authority"] is False
    assert any(reason.startswith("insufficient_folds:") for reason in aggregate["qualification_reasons"])
    assert any(reason.startswith("insufficient_support:") for reason in aggregate["qualification_reasons"])
    assert any(
        reason.startswith("nonpositive_or_below_minimum_net_edge:")
        for reason in aggregate["qualification_reasons"]
    )


def test_fold_local_walk_forward_never_fits_on_oos_rows(monkeypatch) -> None:
    timestamps = pd.date_range("2026-01-02 14:30:00+00:00", periods=72, freq="min")
    dataset = pd.DataFrame(
        {
            **{
                feature: np.linspace(-1.0, 1.0, len(timestamps))
                for feature in REPLAY_ALIGNED_FEATURE_COLUMNS
            },
            "timestamp": timestamps,
            "label_end_timestamp": timestamps + pd.Timedelta(minutes=1),
            "symbol": ["AAPL"] * len(timestamps),
            "close": np.linspace(100.0, 104.0, len(timestamps)),
            "target": [idx % 2 for idx in range(len(timestamps))],
            "net_long_bps": [2.0 if idx % 2 else -1.0 for idx in range(len(timestamps))],
            "session_regime": ["midday"] * len(timestamps),
        }
    )
    trackers: list[Any] = []

    class _TrackingModel:
        classes_ = np.asarray([0, 1])

        def __init__(self) -> None:
            self.fit_index: list[int] = []
            self.predict_indexes: list[list[int]] = []

        def fit(self, features: pd.DataFrame, _target: pd.Series) -> "_TrackingModel":
            self.fit_index = [int(value) for value in features.index]
            return self

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            self.predict_indexes.append([int(value) for value in features.index])
            probability = np.where(features["signal"].to_numpy(dtype=float) >= 0.0, 0.75, 0.55)
            return np.column_stack([1.0 - probability, probability])

    def _tracking_factory(_model_type: str, *, random_state: int) -> _TrackingModel:
        assert random_state > 0
        model = _TrackingModel()
        trackers.append(model)
        return model

    monkeypatch.setattr(trainer, "_make_model", _tracking_factory)

    report, *_ = _run_fold_local_walk_forward(
        dataset,
        args=argparse.Namespace(
            model_type="logistic",
            random_state=10,
            horizon_bars=1,
            walk_forward_folds=5,
            walk_forward_embargo_bars=1,
            walk_forward_embargo_percent=0.0,
            walk_forward_min_trades=1,
            walk_forward_min_profitable_fold_ratio=0.10,
            walk_forward_min_mean_net_edge_bps=-10.0,
            walk_forward_min_ranking_separation_bps=-10.0,
        ),
        edge_global_threshold=0.52,
        cost_model_identity={"version": "test_cost_v1"},
    )

    assert len(trackers) == 5
    assert len(report["folds"]) == 5
    for tracker in trackers:
        assert len(tracker.predict_indexes) == 2
        test_index = tracker.predict_indexes[1]
        assert set(tracker.fit_index).isdisjoint(test_index)
        assert max(tracker.fit_index) < min(test_index)


def test_wfa_market_regimes_use_canonical_per_symbol_history_with_constant_atr() -> None:
    timestamps = pd.date_range("2026-01-02 14:30:00+00:00", periods=100, freq="5min")
    history = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "symbol": "AAPL",
                    "close": np.geomspace(100.0, 135.0, len(timestamps)),
                    "atr_pct": 0.30,
                }
            ),
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "symbol": "MSFT",
                    "close": np.geomspace(135.0, 100.0, len(timestamps)),
                    "atr_pct": 0.30,
                }
            ),
        ],
        ignore_index=True,
    ).sort_values(["timestamp", "symbol"], kind="mergesort")
    test = history.groupby("symbol", sort=True).tail(20).copy()

    regimes, definition = _fold_market_regimes(history, test)

    for symbol in ("AAPL", "MSFT"):
        symbol_history = history.loc[history["symbol"] == symbol]
        expected = infer_day_sleeve_regimes(symbol_history["close"].to_numpy(dtype=float))[-20:]
        actual = tuple(regimes.loc[test["symbol"] == symbol].astype(str))
        assert actual == expected
    assert "uptrend" in set(regimes.loc[test["symbol"] == "AAPL"])
    assert "downtrend" in set(regimes.loc[test["symbol"] == "MSFT"])
    assert set(regimes) != {"volatile"}
    assert definition["market_regime_classifier"] == "day_sleeve_past_only_v1"
    assert definition["past_only"] is True


def test_build_training_dataset_rejects_non_timestamped_csv_by_default(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "seq": list(range(8)),
            "open": np.linspace(100.0, 107.0, 8),
            "high": np.linspace(100.5, 107.5, 8),
            "low": np.linspace(99.5, 106.5, 8),
            "close": np.linspace(100.2, 107.2, 8),
            "volume": [1000.0] * 8,
        }
    ).to_csv(data_dir / "AAPL.csv", index=False)

    with pytest.raises(ValueError, match="requires timestamp-authoritative bars"):
        build_training_dataset(data_dir=data_dir, symbols="AAPL")
