from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import pytest

from ai_trading.tools.migrate_pickle_artifacts import migrate_pickle_artifact


def test_migrate_pickle_model_writes_joblib_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "legacy.pkl"
    destination = tmp_path / "model.joblib"
    source.write_bytes(pickle.dumps({"weights": [1, 2, 3]}))

    output = migrate_pickle_artifact(
        source,
        destination,
        kind="model",
        allow_trusted_pickle=True,
    )

    assert output == destination
    assert joblib.load(destination) == {"weights": [1, 2, 3]}
    manifest = Path(f"{destination}.manifest.json")
    assert manifest.exists()


def test_migrate_pickle_checkpoint_writes_json(tmp_path: Path) -> None:
    source = tmp_path / "legacy_checkpoint.pkl"
    destination = tmp_path / "checkpoint.json"
    source.write_bytes(pickle.dumps({"foo": 1}))

    output = migrate_pickle_artifact(
        source,
        destination,
        kind="checkpoint",
        allow_trusted_pickle=True,
    )

    assert output == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == {"foo": 1}


def test_migrate_pickle_requires_explicit_trust(tmp_path: Path) -> None:
    source = tmp_path / "legacy.pkl"
    destination = tmp_path / "model.joblib"
    source.write_bytes(pickle.dumps({"weights": [1, 2, 3]}))

    with pytest.raises(RuntimeError, match="explicit trusted"):
        migrate_pickle_artifact(source, destination, kind="model")
