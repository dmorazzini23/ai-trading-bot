import ast
import types
import os
from pathlib import Path

import joblib
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

# Extract _load_ml_model from bot_engine using AST to avoid heavy import
SRC = Path(__file__).resolve().parents[1] / "bot_engine.py"
source = SRC.read_text()
tree = ast.parse(source)
func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_ml_model")
mod = types.ModuleType("bot_ml")
mod.logger = __import__("logging").getLogger("bot_ml")
mod.joblib = joblib
mod.Path = Path
mod.pickle = __import__("pickle")
mod._ML_MODEL_CACHE = {}
exec(compile(ast.Module([func], []), filename=str(SRC), mode="exec"), mod.__dict__)

# Provide stub for ai_trading.model_loader used by _load_ml_model
import sys
stub = types.ModuleType("ai_trading.model_loader")
def _stub_load(symbol: str):
    path = Path("models") / f"{symbol}.pkl"
    if not path.exists():
        mod.logger.warning(f"No ML model for {symbol} at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
stub.load_model = _stub_load
sys.modules["ai_trading.model_loader"] = stub


def test_load_missing_logs_error(caplog):
    caplog.set_level("INFO")
    result = mod._load_ml_model("FAKE")
    assert result is None
    assert any("No ML model for FAKE found" in r.message for r in caplog.records)


def test_load_real_model(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model.fit(X, y)
    path = models_dir / "TESTSYM.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    monkeypatch.chdir(tmp_path)
    loaded = mod._load_ml_model("TESTSYM")
    assert loaded is not None
    pred = loaded.predict([[0]])[0]
    assert pred in [0, 1]
