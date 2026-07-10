from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from ai_trading.features.day_sleeve import build_day_sleeve_features
from ai_trading.model_loader import load_day_sleeve_production_model
from ai_trading.model_registry import ModelRegistry
from ai_trading.models.artifacts import write_artifact_manifest
from ai_trading.models.contracts import (
    DAY_SLEEVE_ML_BAR_TIMEFRAME,
    DAY_SLEEVE_ML_FEATURE_COLUMNS,
    DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION,
    model_feature_contract_hash,
)


class _DummyDayModel:
    def __init__(self, marker: str = "model") -> None:
        self.marker = marker
        self.feature_names_in_ = np.asarray(DAY_SLEEVE_ML_FEATURE_COLUMNS)

    def predict(self, values: Any) -> np.ndarray:
        return np.ones(len(values), dtype=int)

    def predict_proba(self, values: Any) -> np.ndarray:
        return np.tile(np.asarray([[0.2, 0.8]], dtype=float), (len(values), 1))


def _manifest_metadata(**overrides: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "required_bar_timeframe": DAY_SLEEVE_ML_BAR_TIMEFRAME,
        "training_bar_timeframe": DAY_SLEEVE_ML_BAR_TIMEFRAME,
        "feature_columns": list(DAY_SLEEVE_ML_FEATURE_COLUMNS),
        "feature_contract_version": DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION,
        "feature_contract_hash": model_feature_contract_hash(
            DAY_SLEEVE_ML_FEATURE_COLUMNS,
            bar_timeframe=DAY_SLEEVE_ML_BAR_TIMEFRAME,
            contract_version=DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION,
        ),
        "dataset_fingerprint": "dataset-hash-1",
    }
    metadata.update(overrides)
    return metadata


def _register_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "production",
    marker: str = "model",
    metadata_overrides: dict[str, Any] | None = None,
) -> tuple[ModelRegistry, str, Path, Path]:
    registry_path = tmp_path / "registry"
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(registry_path))
    monkeypatch.setenv("AI_TRADING_MODEL_MAX_AGE_DAYS", "14")
    artifact_path = tmp_path / f"{marker}.joblib"
    manifest_path = Path(f"{artifact_path}.manifest.json")
    model = _DummyDayModel(marker)
    joblib.dump(model, artifact_path)
    metadata = _manifest_metadata(**(metadata_overrides or {}))
    write_artifact_manifest(
        model_path=artifact_path,
        model_version=f"{marker}-v1",
        metadata=metadata,
    )
    registry = ModelRegistry(registry_path)
    model_id = registry.register_model(
        model,
        strategy="ml_edge",
        model_type="dummy",
        metadata={
            "model_path": str(artifact_path),
            "manifest_path": str(manifest_path),
        },
        dataset_fingerprint="dataset-hash-1",
    )
    registry.update_governance_status(model_id, status)
    return registry, model_id, artifact_path, manifest_path


def _bars(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2026-07-09 13:30", periods=rows, freq="5min", tz="UTC")
    step = np.arange(rows, dtype=float)
    close = 100.0 + (step * 0.02) + np.sin(step / 8.0)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "volume": 10_000.0 + step,
        },
        index=index,
    )


def test_day_sleeve_feature_builder_returns_exact_finite_latest_row() -> None:
    features = build_day_sleeve_features(_bars())

    assert tuple(features.columns) == DAY_SLEEVE_ML_FEATURE_COLUMNS
    assert len(features) == 1
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    row = features.iloc[-1]
    assert row["macd_signal_gap"] == pytest.approx(row["macd"] - row["signal"])
    assert row["rsi_centered"] == pytest.approx((row["rsi"] - 50.0) / 50.0)


@pytest.mark.parametrize("mutation", ["missing_volume", "nonfinite_close"])
def test_day_sleeve_feature_builder_rejects_invalid_inputs(mutation: str) -> None:
    bars = _bars()
    if mutation == "missing_volume":
        bars = bars.drop(columns=["volume"])
    else:
        bars.loc[bars.index[-1], "close"] = np.nan

    with pytest.raises(ValueError, match="missing columns|non-finite"):
        build_day_sleeve_features(bars)


def test_production_loader_returns_verified_model_and_immutable_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _registry, model_id, _artifact, _manifest = _register_model(
        tmp_path,
        monkeypatch,
    )

    loaded = load_day_sleeve_production_model()
    cached = load_day_sleeve_production_model()

    assert loaded is cached
    assert getattr(loaded.model, "marker") == "model"
    assert loaded.lineage == {
        "model_id": model_id,
        "model_version": "model-v1",
        "dataset_hash": "dataset-hash-1",
        "feature_version": DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION,
        "model_artifact_hash": loaded.lineage["model_artifact_hash"],
    }
    assert len(loaded.lineage["model_artifact_hash"]) == 64
    with pytest.raises(TypeError):
        loaded.lineage["model_id"] = "changed"  # type: ignore[index]


def test_production_loader_rejects_shadow_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_model(tmp_path, monkeypatch, status="shadow")

    with pytest.raises(RuntimeError, match="unavailable"):
        load_day_sleeve_production_model()


@pytest.mark.parametrize(
    "metadata_overrides",
    [
        {"required_bar_timeframe": "1Day"},
        {"feature_columns": list(reversed(DAY_SLEEVE_ML_FEATURE_COLUMNS))},
        {"feature_contract_version": "wrong-version"},
        {"feature_contract_hash": "wrong-hash"},
    ],
)
def test_production_loader_rejects_incompatible_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_overrides: dict[str, Any],
) -> None:
    _register_model(
        tmp_path,
        monkeypatch,
        metadata_overrides=metadata_overrides,
    )

    with pytest.raises(RuntimeError, match="contract is incompatible"):
        load_day_sleeve_production_model()


def test_production_loader_rejects_stale_registry_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, model_id, _artifact, _manifest = _register_model(tmp_path, monkeypatch)
    registry.model_index[model_id]["registered_at"] = "2000-01-01T00:00:00+00:00"
    registry._save_index()

    with pytest.raises(RuntimeError, match="stale"):
        load_day_sleeve_production_model()


def test_production_loader_rejects_unverifiable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _registry, _model_id, artifact, _manifest = _register_model(tmp_path, monkeypatch)
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="verification failed"):
        load_day_sleeve_production_model()


def test_production_loader_rejects_missing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _registry, _model_id, _artifact, manifest = _register_model(tmp_path, monkeypatch)
    manifest.unlink()

    with pytest.raises(RuntimeError, match="manifest is missing"):
        load_day_sleeve_production_model()


def test_production_loader_cache_tracks_active_model_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_model(tmp_path, monkeypatch, marker="first")
    first = load_day_sleeve_production_model()
    _register_model(tmp_path, monkeypatch, marker="second")

    second = load_day_sleeve_production_model()

    assert second is not first
    assert getattr(second.model, "marker") == "second"
    assert second.lineage["model_id"] != first.lineage["model_id"]
