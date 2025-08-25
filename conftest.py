import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import pytest


def _missing(mod: str) -> bool:
    return importlib.util.find_spec(mod) is None


@pytest.fixture(scope="session", autouse=True)
def _seed_tests() -> None:
    """Ensure deterministic test execution."""
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)
    if not _missing("torch"):
        import torch

        torch.manual_seed(0)


def pytest_ignore_collect(path, config):
    """Ignore sklearn-heavy slow tests when sklearn is unavailable."""
    p = Path(str(path))
    needs_sklearn = p.name == "test_meta_learning_heavy.py" or "slow" in p.parts
    return needs_sklearn and _missing("sklearn")
