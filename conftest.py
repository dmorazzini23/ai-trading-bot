import importlib
import os
import random
import sys
import types
from pathlib import Path

import pytest

# Provide a lightweight default model so bot initialization can preload it
_dummy_mod = types.ModuleType("dummy_model")


class _DummyModel:
    def predict(self, _x):
        return [0]

    def predict_proba(self, _x):
        return [[0.5, 0.5]]

def _get_model() -> _DummyModel:
    """Return an instance of the dummy model.

    Using a named function keeps the test helper picklable with the standard
    ``pickle`` module, whereas inline lambdas cannot be pickled.
    """

    return _DummyModel()


_dummy_mod.get_model = _get_model
sys.modules["dummy_model"] = _dummy_mod
os.environ.setdefault("AI_TRADING_MODEL_MODULE", "dummy_model")


def _missing(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is None
    except ValueError:
        return True


@pytest.fixture(scope="session", autouse=True)
def _seed_tests() -> None:
    """Ensure deterministic test execution."""
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    if not _missing("numpy"):
        import numpy as np

        np.random.seed(0)
    if not _missing("torch"):
        import torch

        torch.manual_seed(0)


def pytest_ignore_collect(path, config):
    """Ignore heavy or optional-dep tests when prerequisites are missing."""
    p = Path(str(path))
    needs_sklearn = p.name == "test_meta_learning_heavy.py" or "slow" in p.parts
    if needs_sklearn and _missing("sklearn"):
        return True
    if _missing("numpy"):
        repo = Path(__file__).parent.resolve()
        allowed = {
            repo / "tests" / "test_runner_smoke.py",
            repo / "tests" / "test_utils_timing.py",
            repo / "tests" / "unit" / "test_trading_config_aliases.py",
            repo / "tests" / "test_current_api.py",
        }
        try:
            rel = p.resolve()
        except FileNotFoundError:
            rel = p
        if p.is_dir():
            return not any(a.is_relative_to(rel) for a in allowed)
        return rel not in allowed
    return False
