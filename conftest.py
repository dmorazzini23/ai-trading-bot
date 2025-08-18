import importlib.util
from pathlib import Path


def _missing(mod: str) -> bool:
    return importlib.util.find_spec(mod) is None


def pytest_ignore_collect(path, config):
    """Ignore sklearn-heavy slow tests when sklearn is unavailable."""
    p = Path(str(path))
    needs_sklearn = p.name == "test_meta_learning_heavy.py" or "slow" in p.parts
    return needs_sklearn and _missing("sklearn")
