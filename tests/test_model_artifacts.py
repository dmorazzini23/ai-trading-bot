from __future__ import annotations

from pathlib import Path

import json

from ai_trading.models.artifacts import verify_artifact, write_artifact_manifest


def test_verify_artifact_round_trip(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model-payload-v1")

    write_artifact_manifest(
        model_path=model_path,
        model_version="v1",
        training_data_range={"start": "2025-01-01", "end": "2025-01-31"},
    )
    ok, reason = verify_artifact(model_path=model_path)
    assert ok is True
    assert reason == "OK"


def test_verify_artifact_detects_checksum_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model-payload-v1")

    write_artifact_manifest(
        model_path=model_path,
        model_version="v1",
        training_data_range={"start": "2025-01-01", "end": "2025-01-31"},
    )
    model_path.write_bytes(b"tampered")

    ok, reason = verify_artifact(model_path=model_path)
    assert ok is False
    assert reason == "CHECKSUM_MISMATCH"


def test_write_artifact_manifest_includes_metadata(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model-payload-v1")

    manifest_path = write_artifact_manifest(
        model_path=model_path,
        model_version="v2",
        training_data_range={"start": "2025-02-01", "end": "2025-02-28"},
        metadata={"strategy": "after_hours_ml_edge", "dataset_fingerprint": "abc123"},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["strategy"] == "after_hours_ml_edge"
    assert payload["metadata"]["dataset_fingerprint"] == "abc123"
