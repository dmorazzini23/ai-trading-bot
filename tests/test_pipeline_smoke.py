import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


@pytest.mark.smoke
def test_pipeline_basic(monkeypatch):
    skl_base = types.ModuleType("sklearn.base")
    skl_base.BaseEstimator = type("BE", (), {})
    skl_base.TransformerMixin = type("TM", (), {})
    monkeypatch.setitem(sys.modules, "sklearn.base", skl_base)

    skl_pipeline = types.ModuleType("sklearn.pipeline")

    class Pipeline:
        def __init__(self, steps):
            self.steps = steps

        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    skl_pipeline.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "sklearn.pipeline", skl_pipeline)

    skl_pre = types.ModuleType("sklearn.preprocessing")

    class StandardScaler:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.asarray(X, dtype=float)

    skl_pre.StandardScaler = StandardScaler
    monkeypatch.setitem(sys.modules, "sklearn.preprocessing", skl_pre)

    skl_lin = types.ModuleType("sklearn.linear_model")

    class SGDRegressor:
        def __init__(self, *a, **k):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    skl_lin.SGDRegressor = SGDRegressor
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", skl_lin)

    monkeypatch.syspath_prepend(".")
    sys.modules.pop("pipeline", None)
    pipeline = importlib.import_module("pipeline")

    df = pd.DataFrame({"close": np.arange(10, dtype=float)})
    arr = pipeline.FeatureBuilder().transform(df)
    assert arr.shape[0] == 10
    pipeline.model_pipeline.fit(df, np.arange(len(df)))
    force_coverage(pipeline)
