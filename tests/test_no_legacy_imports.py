import importlib.util

import pytest

BANNED = [
    "metrics",
    "algorithm_optimizer",
    "indicator_manager",
    "predict",
    "runner",
    "data_fetcher",
    "validate_env",
]


@pytest.mark.unit
def test_legacy_modules_not_importable():  # AI-AGENT-REF
    for name in BANNED:
        assert importlib.util.find_spec(name) is None
        with pytest.raises(ModuleNotFoundError):
            __import__(name)

