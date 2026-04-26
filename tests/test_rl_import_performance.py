import importlib
import sys
import time
from types import ModuleType

import pytest

MODULES = [
    "ai_trading.rl_trading",
    "ai_trading.rl_trading.train",
    "ai_trading.rl_trading.env",
]


@pytest.mark.parametrize("module", MODULES)
def test_rl_import_is_fast(module: str) -> None:
    cleared = MODULES + ["stable_baselines3", "gymnasium", "torch", "torch.optim"]
    original_modules: dict[str, ModuleType | None] = {
        name: sys.modules.get(name) for name in cleared
    }
    try:
        for mod in cleared:
            sys.modules.pop(mod, None)
        start = time.perf_counter()
        importlib.import_module(module)
        duration = time.perf_counter() - start
        assert duration < 0.25
        assert "stable_baselines3" not in sys.modules
        assert "torch" not in sys.modules
        assert "torch.optim" not in sys.modules
        assert "gymnasium" not in sys.modules
    finally:
        for name in reversed(cleared):
            original = original_modules[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
