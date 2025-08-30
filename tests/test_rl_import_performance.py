import importlib
import sys
import time

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
from ai_trading.utils.device import TORCH_AVAILABLE
if not TORCH_AVAILABLE:
    pytest.skip("torch not installed", allow_module_level=True)

MODULES = [
    "ai_trading.rl_trading",
    "ai_trading.rl_trading.train",
    "ai_trading.rl_trading.env",
]


@pytest.mark.parametrize("module", MODULES)
def test_rl_import_is_fast(module):
    for mod in MODULES + ["stable_baselines3", "gymnasium", "torch", "torch.optim"]:
        sys.modules.pop(mod, None)
    start = time.perf_counter()
    importlib.import_module(module)
    duration = time.perf_counter() - start
    assert duration < 0.25
    assert "stable_baselines3" not in sys.modules
    assert "torch" not in sys.modules
    assert "torch.optim" not in sys.modules
    assert "gymnasium" not in sys.modules
