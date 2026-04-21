from pathlib import Path

import pytest

from ai_trading.ml import model_io


def test_save_and_load_json_safe_model(tmp_path: Path) -> None:
    model = {"weights": [1, 2, 3], "threshold": 0.42}
    path = tmp_path / "model.json"

    model_io.save_model(model, path)
    loaded = model_io.load_model(path)

    assert loaded == model


def test_save_model_rejects_non_json_payload(tmp_path: Path) -> None:
    path = tmp_path / "model.json"

    with pytest.raises(RuntimeError, match="only JSON-safe artifacts are supported"):
        model_io.save_model({"f": lambda x: x + 1}, path)


def test_load_missing_model_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="Model file not found"):
        model_io.load_model(missing)


def test_load_model_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="only JSON-safe artifacts are supported"):
        model_io.load_model(path)
