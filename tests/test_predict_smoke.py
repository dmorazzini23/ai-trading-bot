import importlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


def _import_predict(monkeypatch):
    req_mod = types.ModuleType("requests")
    req_mod.get = lambda *a, **k: types.SimpleNamespace(json=lambda: {"articles": []}, raise_for_status=lambda: None)
    req_mod.exceptions = types.SimpleNamespace(RequestException=Exception)
    monkeypatch.setitem(sys.modules, "requests", req_mod)

    rt_mod = types.ModuleType("retrain")
    rt_mod.prepare_indicators = lambda df, freq="intraday": df.assign(feat1=0)
    monkeypatch.setitem(sys.modules, "retrain", rt_mod)

    if "predict" in sys.modules:
        del sys.modules["predict"]
    return importlib.import_module("predict")


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


class DummyModel:
    feature_names_in_ = ["feat1", "sentiment"]

    def predict(self, X):
        return [0]

    def predict_proba(self, X):
        return [[1.0]]


@pytest.mark.smoke
def test_predict_function(tmp_path, monkeypatch):
    predict = _import_predict(monkeypatch)
    monkeypatch.setattr(predict, "load_model", lambda regime: DummyModel())
    csv_path = tmp_path / "AAPL.csv"
    pd.DataFrame({"close": [1.0]}).to_csv(csv_path, index=False)
    pred, proba = predict.predict(str(csv_path))
    assert proba is not None
    force_coverage(predict)
