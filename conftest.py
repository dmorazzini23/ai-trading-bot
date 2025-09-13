import os
import random
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from tests.dummy_model_util import _DummyModel, _get_model

import pytest


# Ensure optional light-weight stubs are available only when real deps are missing
def _ensure_test_stubs() -> None:
    repo = Path(__file__).parent.resolve()
    stubs = repo / "tests" / "stubs"
    if not stubs.exists():
        return

    def _missing(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is None
        except ValueError:
            return True

    need_stubs = any(
        _missing(m)
        for m in (
            "pydantic",
            "pydantic_settings",
            # Use a stub for Retry if urllib3 not installed
            "urllib3",
        )
    )
    if need_stubs and str(stubs) not in sys.path:
        sys.path.insert(0, str(stubs))


_ensure_test_stubs()

# Provide a lightweight default model so bot initialization can preload it

_dummy_mod = types.ModuleType("dummy_model")

setattr(_dummy_mod, "get_model", _get_model)
setattr(_dummy_mod, "_DummyModel", _DummyModel)
sys.modules["dummy_model"] = _dummy_mod
os.environ.setdefault("AI_TRADING_MODEL_MODULE", "dummy_model")
os.environ.setdefault("MAX_DRAWDOWN_THRESHOLD", "0.1")


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


def pytest_ignore_collect(path: Path, config: Any) -> bool:
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
