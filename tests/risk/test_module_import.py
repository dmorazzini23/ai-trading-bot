"""Tests covering lazy risk module configuration imports."""
from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from typing import Any, cast


@contextmanager
def _reload(module_name: str):
    """Isolate a risk-package import and restore the original module identities."""

    original = {
        name: module for name, module in sys.modules.items()
        if name == module_name or name.startswith(f"{module_name}.")
    }
    parent = sys.modules.get("ai_trading")
    original_attribute = getattr(parent, "risk", None) if parent is not None else None

    to_delete = [
        name
        for name in list(sys.modules)
        if name == module_name or name.startswith(f"{module_name}.")
    ]
    for name in to_delete:
        sys.modules.pop(name, None)

    stub = None
    stub_name = "ai_trading.risk.engine"
    if module_name == "ai_trading.risk" and stub_name not in sys.modules:
        stub = types.ModuleType(stub_name)
        setattr(cast(Any, stub), "RiskEngine", type("RiskEngine", (), {}))
        setattr(cast(Any, stub), "__all__", ["RiskEngine"])
        sys.modules[stub_name] = stub

    try:
        module = importlib.import_module(module_name)
        if stub is not None and sys.modules.get(stub_name) is stub:
            sys.modules.pop(stub_name, None)
        yield module
    finally:
        for name in list(sys.modules):
            if name == module_name or name.startswith(f"{module_name}."):
                sys.modules.pop(name, None)
        sys.modules.update(original)
        if parent is not None:
            if original_attribute is None:
                parent.__dict__.pop("risk", None)
            else:
                parent.risk = original_attribute


def test_risk_import_without_drawdown_threshold(monkeypatch, default_env):
    """Importing ai_trading.risk should not require MAX_DRAWDOWN_THRESHOLD."""

    monkeypatch.delenv("MAX_DRAWDOWN_THRESHOLD", raising=False)
    monkeypatch.delenv("AI_TRADING_MAX_DRAWDOWN_THRESHOLD", raising=False)
    with _reload("ai_trading.risk") as module:
        cfg = module.kelly.get_default_config()
        # relaxed loader should default to 0.08 when drawdown env is absent
        assert cfg.max_drawdown_threshold == 0.08


def test_kelly_default_config_uses_env_when_available(monkeypatch, default_env):
    """Once the environment provides the drawdown threshold we surface it."""

    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.25")
    with _reload("ai_trading.risk") as module:
        module.kelly.reset_default_config()
        cfg = module.kelly.get_default_config()
        assert cfg.max_drawdown_threshold == 0.25


def test_isolated_import_restores_original_engine_module(default_env):
    original_engine = importlib.import_module("ai_trading.risk.engine")
    original_package = importlib.import_module("ai_trading.risk")
    with _reload("ai_trading.risk"):
        assert sys.modules["ai_trading.risk"] is not original_package
    assert sys.modules["ai_trading.risk"] is original_package
    assert sys.modules["ai_trading.risk.engine"] is original_engine
    assert sys.modules["ai_trading"].risk is original_package
