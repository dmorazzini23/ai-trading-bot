from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_trading.utils import determinism


class _ArrayLike:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload


class _ColumnNames(list[str]):
    def tolist(self) -> list[str]:
        return list(self)


class _FrameLike:
    columns = _ColumnNames(["beta", "alpha"])
    shape = (12, 2)
    values = _ArrayLike(b"frame-bytes")

    def __len__(self) -> int:
        return self.shape[0]

    def head(self, count: int) -> _FrameLike:
        assert count == 10
        return self


class _BadString:
    def __str__(self) -> str:
        raise ValueError("cannot stringify")


class _BadShape:
    @property
    def shape(self) -> tuple[int, int]:
        raise ValueError("cannot inspect shape")


def test_hash_data_is_stable_for_order_insensitive_inputs() -> None:
    assert determinism.hash_data({"b": 2, "a": 1}) == determinism.hash_data({"a": 1, "b": 2})
    assert determinism.hash_data([3, 1, 2]) == determinism.hash_data((1, 2, 3))
    assert determinism.hash_data(_ArrayLike(b"abc")) == determinism.hash_data(_ArrayLike(b"abc"))


def test_hash_data_returns_unknown_for_unhashable_payload() -> None:
    assert determinism.hash_data(_BadString()) == "unknown"


def test_hash_features_handles_none_frame_like_and_error_payloads() -> None:
    assert determinism.hash_features(None) == "no_features"
    assert determinism.hash_features(_FrameLike()) == determinism.hash_features(_FrameLike())
    assert determinism.hash_features(_BadShape()) == "feature_hash_error"


def test_generate_spec_hash_changes_with_training_parameters() -> None:
    base = determinism.generate_spec_hash("features", "labels", {"start": "2026-01-01"})
    tuned = determinism.generate_spec_hash(
        "features",
        "labels",
        {"start": "2026-01-01"},
        additional_params={"learning_rate": 0.1},
    )

    assert base != tuned
    assert determinism.hash_labels(None) == "no_labels"


def test_set_random_seeds_does_not_mutate_pythonhashseed_runtime() -> None:
    source = Path(determinism.__file__).read_text(encoding="utf-8")

    assert "os.putenv" not in source
    assert "PYTHONHASHSEED_STARTUP_REQUIRED" in source


def test_model_spec_persists_locking_and_compatibility(tmp_path: Path) -> None:
    spec_path = tmp_path / "meta.json"
    spec = determinism.ModelSpecification(str(spec_path))

    spec_hash = spec.update_spec(
        feature_data=[3, 1, 2],
        label_data=[1, 0, 1],
        data_window={"start": "2026-01-01", "end": "2026-01-31"},
        cost_model_version="2.0",
    )
    assert spec_hash
    assert json.loads(spec_path.read_text())["spec_hash"] == spec_hash

    assert spec.validate_compatibility(
        feature_data=[3, 1, 2],
        label_data=[1, 0, 1],
        data_window={"start": "2026-01-01", "end": "2026-01-31"},
        cost_model_version="2.0",
    ) == (True, "Specification matches")

    spec.lock_spec()
    assert spec.is_locked() is True
    assert spec.update_spec(feature_data=[9], force=False) == spec_hash

    forced_hash = spec.update_spec(feature_data=[9], force=True)
    assert forced_hash != spec_hash


def test_validate_compatibility_override_uses_config_management(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = determinism.ModelSpecification()
    spec._spec = {"spec_hash": "stored", "feature_hash": "old"}

    def _fake_get_env(*_args: Any, **_kwargs: Any) -> str:
        return "true"

    monkeypatch.setattr(determinism, "get_env", _fake_get_env)

    assert spec.validate_compatibility(feature_data=[1]) == (True, "Override enabled")


def test_ensure_deterministic_training_updates_forced_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class _Spec:
        def __init__(self) -> None:
            self.updated = False

        def update_spec(self, **kwargs: Any) -> str:
            self.updated = True
            assert kwargs["force"] is True
            return "updated"

        def validate_compatibility(self, **_kwargs: Any) -> tuple[bool, str]:
            return (False, "should not validate")

    fake_spec = _Spec()
    monkeypatch.setattr(determinism, "set_random_seeds", lambda seed: calls.append(seed))
    monkeypatch.setattr(determinism, "get_model_spec", lambda: fake_spec)

    assert determinism.ensure_deterministic_training(seed=7, force_update=True) == (
        True,
        "Specification updated",
    )
    assert calls == [7]
    assert fake_spec.updated is True
