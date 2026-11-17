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
