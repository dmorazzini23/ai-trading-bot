import importlib
import sys
import time

import pytest

MODULES = [
    "ai_trading.rl_trading",
    "ai_trading.rl_trading.train",
    "ai_trading.rl_trading.env",
]


@pytest.mark.parametrize("module", MODULES)
def test_rl_import_is_fast(module):
    for mod in MODULES + ["stable_baselines3", "gymnasium", "torch"]:
        sys.modules.pop(mod, None)
    start = time.perf_counter()
    importlib.import_module(module)
    duration = time.perf_counter() - start
    assert duration < 0.25
    assert "stable_baselines3" not in sys.modules
    assert "torch" not in sys.modules
    assert "gymnasium" not in sys.modules
