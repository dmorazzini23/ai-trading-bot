import importlib
import sys
import types
from importlib import import_module
from types import SimpleNamespace

import ai_trading.env as env_mod
import ai_trading.util.env_check as env_check


def test_execution_algorithms_and_result_importable():
    """Ensure algorithms submodule and ExecutionResult can be imported."""
    algos = import_module("ai_trading.execution.algorithms")
    from ai_trading.execution import ExecutionResult

    assert algos is not None
    assert ExecutionResult is not None


def test_execution_engine_real_when_dotenv_unresolved(monkeypatch):
    """Importing ai_trading.execution should yield the real engine when dotenv is missing."""

    original_env_flag = env_mod.PYTHON_DOTENV_RESOLVED
    original_guard_flag = env_check.PYTHON_DOTENV_RESOLVED

    monkeypatch.setattr(env_mod, "PYTHON_DOTENV_RESOLVED", False)
    monkeypatch.setattr(env_check, "PYTHON_DOTENV_RESOLVED", False)

    import ai_trading.config as config_mod
    config_mod = importlib.reload(config_mod)

    import ai_trading.execution as execution_mod
    execution_mod = importlib.reload(execution_mod)

    assert execution_mod.ExecutionEngine.__module__ == "ai_trading.execution.engine"
    assert not getattr(execution_mod.ExecutionEngine, "_IS_STUB", False)

    indicators_stub = types.ModuleType("ai_trading.indicators")
    indicators_stub.atr = lambda *args, **kwargs: None
    indicators_stub.compute_atr = lambda *args, **kwargs: 0.0
    indicators_stub.mean_reversion_zscore = lambda *args, **kwargs: 0.0
    indicators_stub.rsi = lambda *args, **kwargs: 0.0
    indicators_stub.ichimoku_fallback = lambda *args, **kwargs: None

    def _indicator_default(name: str):  # pragma: no cover - fallback for unspecified attributes
        return lambda *args, **kwargs: None

    indicators_stub.__getattr__ = _indicator_default  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.indicators", indicators_stub)

    ipm_stub = types.ModuleType("ai_trading.position.intelligent_manager")
    ipm_stub.IntelligentPositionManager = object
    ipm_stub.PositionAction = object
    monkeypatch.setitem(sys.modules, "ai_trading.position.intelligent_manager", ipm_stub)

    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1
    portalocker_stub.lock = lambda *args, **kwargs: None
    portalocker_stub.unlock = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "portalocker", portalocker_stub)

    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - helper stub for import side effects
        def __init__(self, *args, **kwargs) -> None:
            self.text = ""

        def find_all(self, *args, **kwargs):
            return []

    bs4_stub.BeautifulSoup = _BeautifulSoup
    monkeypatch.setitem(sys.modules, "bs4", bs4_stub)

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.random = SimpleNamespace(seed=lambda *args, **kwargs: None)
    numpy_stub.nan = float("nan")
    numpy_stub.NaN = float("nan")
    numpy_stub.inf = float("inf")
    numpy_stub.floating = float
    numpy_stub.isfinite = lambda value: True
    numpy_stub.asarray = lambda arr, dtype=None: arr
    numpy_stub.array = lambda arr, dtype=None: arr
    numpy_stub.cumsum = lambda arr: arr
    numpy_stub.insert = lambda arr, index, value: arr
    numpy_stub.polyfit = lambda *args, **kwargs: [0.0, 0.0]
    numpy_stub.std = lambda arr: 0.0
    numpy_stub.mean = lambda arr: 0.0
    numpy_stub.clip = lambda arr, a_min=None, a_max=None: arr
    numpy_stub.where = lambda condition, x=None, y=None: x if condition else y
    numpy_stub.full_like = lambda arr, fill_value: arr
    numpy_stub.divide = lambda a, b, out=None, where=None: a
    numpy_stub.finfo = lambda dtype: SimpleNamespace(eps=1e-9)
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    from ai_trading.core import bot_engine

    bot_engine = importlib.reload(bot_engine)

    runtime = SimpleNamespace()
    bot_engine._ensure_execution_engine(runtime)

    engine = getattr(runtime, "execution_engine", None)
    assert isinstance(engine, execution_mod.ExecutionEngine)
    assert not getattr(engine, "_IS_STUB", False)

    # Restore the original configuration flags to avoid leaking state.
    monkeypatch.setattr(env_mod, "PYTHON_DOTENV_RESOLVED", original_env_flag)
    monkeypatch.setattr(env_check, "PYTHON_DOTENV_RESOLVED", original_guard_flag)
    config_mod = importlib.reload(config_mod)
    execution_mod = importlib.reload(execution_mod)
    importlib.reload(bot_engine)

