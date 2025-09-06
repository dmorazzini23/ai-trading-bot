import sys
import types
import numpy as np


def test_basic_pipeline_runs(monkeypatch):
    skl_pipeline = types.ModuleType("sklearn.pipeline")

    class Pipeline:
        def __init__(self, steps):
            self.steps = steps

        def fit(self, X, y=None):
            for _, step in self.steps:
                if hasattr(step, "fit"):
                    step.fit(X, y)
            return self

        def transform(self, X):
            for _, step in self.steps:
                if hasattr(step, "transform"):
                    X = step.transform(X)
            return X

    skl_pipeline.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "sklearn.pipeline", skl_pipeline)
    monkeypatch.syspath_prepend(".")

    from ai_trading.pipeline.basic import create_pipeline

    pipeline = create_pipeline()
    X = [[1], [2], [3]]
    pipeline.fit(X)
    result = pipeline.transform(X)
    assert np.asarray(result).shape == (3, 1)
