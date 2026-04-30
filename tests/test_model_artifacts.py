from __future__ import annotations

from pathlib import Path

import json

import pytest

from ai_trading.models import artifacts
from ai_trading.models.artifacts import enforce_artifact_verification, verify_artifact, write_artifact_manifest


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


def test_enforce_artifact_verification_fails_closed_outside_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model-payload-v1")
    manifest_path = write_artifact_manifest(model_path=model_path, model_version="v1")
    model_path.write_bytes(b"tampered")

    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_MODEL_VERIFY_CHECKSUM", "1")
    monkeypatch.setattr(artifacts, "is_test_runtime", lambda: False)

    with pytest.raises(RuntimeError, match="MODEL_VERIFICATION_FAILED: CHECKSUM_MISMATCH"):
        enforce_artifact_verification(model_path=model_path, manifest_path=manifest_path)


def test_enforce_artifact_verification_cannot_be_disabled_in_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model-payload-v1")

    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_MODEL_VERIFY_CHECKSUM", "0")

    with pytest.raises(RuntimeError, match="MODEL_VERIFICATION_DISABLED_BLOCKED"):
        enforce_artifact_verification(model_path=model_path)


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
