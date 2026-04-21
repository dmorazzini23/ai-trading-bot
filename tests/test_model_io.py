import importlib.util
from pathlib import Path

import pytest

from ai_trading.ml import model_io
from ai_trading.utils import safe_pickle


def test_save_and_load_lambda(tmp_path):
    supports_lambda = (
        importlib.util.find_spec("dill") is not None
        or importlib.util.find_spec("cloudpickle") is not None
    )
    model = {"f": lambda x: x + 1}
    path = tmp_path / "lambda.pkl"
    if not supports_lambda:
        with pytest.raises(RuntimeError, match="Failed to save model"):
            model_io.save_model(model, path)
        return
    model_io.save_model(model, path)
    loaded = model_io.load_model(path)
    assert loaded["f"](1) == 2


def test_load_missing_model_raises(tmp_path):
    missing = tmp_path / "missing.pkl"
    with pytest.raises(RuntimeError, match="Model file not found"):
        model_io.load_model(missing)


def test_load_model_blocks_unsafe_deserialization_outside_test_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pkl"
    model_io.save_model({"value": 1}, path)
    monkeypatch.setattr(safe_pickle, "is_test_runtime", lambda: False)
    monkeypatch.delenv("AI_TRADING_ALLOW_UNSAFE_MODEL_DESERIALIZATION", raising=False)

    with pytest.raises(RuntimeError, match="unsafe generic model deserialization"):
        model_io.load_model(path)
