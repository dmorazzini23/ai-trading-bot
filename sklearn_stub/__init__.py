"""Minimal sklearn stub for tests."""
import sys
from types import ModuleType

base = ModuleType("sklearn.base")
class BaseEstimator: ...
base.BaseEstimator = BaseEstimator
base.clone = lambda est: est

linear_model = ModuleType("sklearn.linear_model")
class Ridge: ...
class BayesianRidge: ...
linear_model.Ridge = Ridge
linear_model.BayesianRidge = BayesianRidge
ensemble = ModuleType("sklearn.ensemble")
class RandomForestClassifier: ...
ensemble.RandomForestClassifier = RandomForestClassifier
decomposition = ModuleType("sklearn.decomposition")
model_selection = ModuleType("sklearn.model_selection")
pipeline = ModuleType("sklearn.pipeline")
preprocessing = ModuleType("sklearn.preprocessing")

for mod in [base, linear_model, ensemble, decomposition, model_selection, pipeline, preprocessing]:
    sys.modules[mod.__name__] = mod

__all__ = ["base", "linear_model", "ensemble", "decomposition", "model_selection", "pipeline", "preprocessing"]
