import importlib
import sys
import time


def test_rl_import_is_fast():
    for mod in (
        "ai_trading.rl_trading",
        "stable_baselines3",
        "gymnasium",
        "torch",
    ):
        sys.modules.pop(mod, None)
    start = time.perf_counter()
    importlib.import_module("ai_trading.rl_trading")
    duration = time.perf_counter() - start
    assert duration < 0.25
    assert "stable_baselines3" not in sys.modules
    assert "torch" not in sys.modules
    assert "gymnasium" not in sys.modules
