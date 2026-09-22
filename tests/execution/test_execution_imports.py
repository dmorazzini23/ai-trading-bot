import importlib
import sys
import types
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest
import numpy as np

import ai_trading.env as env_mod
import ai_trading.util.env_check as env_check


def _set_module_attr(module: types.ModuleType, attr_name: str, value: Any) -> None:
    setattr(module, attr_name, value)


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
    _set_module_attr(indicators_stub, "atr", lambda *args, **kwargs: None)
    _set_module_attr(indicators_stub, "compute_atr", lambda *args, **kwargs: 0.0)
    _set_module_attr(indicators_stub, "mean_reversion_zscore", lambda *args, **kwargs: 0.0)
    _set_module_attr(indicators_stub, "rsi", lambda *args, **kwargs: 0.0)
    _set_module_attr(indicators_stub, "ichimoku_fallback", lambda *args, **kwargs: None)

    def _indicator_default(name: str):  # pragma: no cover - fallback for unspecified attributes
        return lambda *args, **kwargs: None

    _set_module_attr(indicators_stub, "__getattr__", _indicator_default)
    monkeypatch.setitem(sys.modules, "ai_trading.indicators", indicators_stub)

    ipm_stub = types.ModuleType("ai_trading.position.intelligent_manager")
    _set_module_attr(ipm_stub, "IntelligentPositionManager", object)
    _set_module_attr(ipm_stub, "PositionAction", object)
    monkeypatch.setitem(sys.modules, "ai_trading.position.intelligent_manager", ipm_stub)

    portalocker_stub = types.ModuleType("portalocker")
    _set_module_attr(portalocker_stub, "LOCK_EX", 1)
    _set_module_attr(portalocker_stub, "lock", lambda *args, **kwargs: None)
    _set_module_attr(portalocker_stub, "unlock", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "portalocker", portalocker_stub)

    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - helper stub for import side effects
        def __init__(self, *args, **kwargs) -> None:
            self.text = ""

        def find_all(self, *args, **kwargs):
            return []

    _set_module_attr(bs4_stub, "BeautifulSoup", _BeautifulSoup)
    monkeypatch.setitem(sys.modules, "bs4", bs4_stub)

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
    assert bot_engine.np is np


def test_optional_export_programming_errors_are_not_swallowed(monkeypatch):
    """Optional import guards should not hide runtime/programming failures."""

    import ai_trading.execution as execution_mod

    debug_tracker_stub = types.ModuleType("ai_trading.execution.debug_tracker")

    def _raise_runtime_error(_name: str) -> Any:
        raise RuntimeError("debug tracker import bug")

    debug_tracker_stub.__getattr__ = _raise_runtime_error  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.execution.debug_tracker", debug_tracker_stub)

    try:
        with pytest.raises(RuntimeError, match="debug tracker import bug"):
            importlib.reload(execution_mod)
    finally:
        monkeypatch.delitem(sys.modules, "ai_trading.execution.debug_tracker", raising=False)
        importlib.reload(execution_mod)


def test_execution_settings_errors_are_not_swallowed(monkeypatch):
    import ai_trading.config as config_mod
    import ai_trading.execution as execution_mod

    def _raise_config_error() -> Any:
        raise RuntimeError("settings are invalid")

    monkeypatch.setattr(config_mod, "get_execution_settings", _raise_config_error)

    with pytest.raises(RuntimeError, match="settings are invalid"):
        execution_mod._load_execution_settings()


def test_unknown_execution_mode_selection_fails_fast(monkeypatch):
    import ai_trading.execution as execution_mod
    from ai_trading.core.runtime_contract import UnknownExecutionModeError

    monkeypatch.setattr(
        execution_mod,
        "_load_execution_settings",
        lambda: (SimpleNamespace(mode="mystery", shadow_mode=False), None),
    )
    monkeypatch.setattr("ai_trading.core.runtime_contract.is_testing_mode", lambda: False)

    with pytest.raises(UnknownExecutionModeError, match="EXECUTION_MODE must be one of"):
        execution_mod.select_execution_engine(force_refresh=True)
