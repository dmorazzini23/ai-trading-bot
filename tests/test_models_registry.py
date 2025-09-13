from pathlib import Path

import pytest

from ai_trading.models.registry import register_model, reset_registry


def test_register_same_path_returns_existing(tmp_path: Path) -> None:
    reset_registry()
    path = tmp_path / "model.pkl"
    first = register_model("demo", path)
    second = register_model("demo", path)
    assert first == second


def test_register_conflicting_path_raises(tmp_path: Path) -> None:
    reset_registry()
    register_model("demo", tmp_path / "model1.pkl")
    with pytest.raises(ValueError):
        register_model("demo", tmp_path / "model2.pkl")
