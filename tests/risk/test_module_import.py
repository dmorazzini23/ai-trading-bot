"""Tests covering lazy risk module configuration imports."""
from __future__ import annotations

import importlib
import sys
import types


def _reload(module_name: str):
    """Reload a module and its ai_trading.risk submodules for isolated tests."""

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
        stub.RiskEngine = type("RiskEngine", (), {})  # type: ignore[attr-defined]
        stub.__all__ = ["RiskEngine"]
        sys.modules[stub_name] = stub

    try:
        return importlib.import_module(module_name)
    finally:
        if stub is not None and sys.modules.get(stub_name) is stub:
            sys.modules.pop(stub_name, None)


def test_risk_import_without_drawdown_threshold(monkeypatch, default_env):
    """Importing ai_trading.risk should not require MAX_DRAWDOWN_THRESHOLD."""

    monkeypatch.delenv("MAX_DRAWDOWN_THRESHOLD", raising=False)
    monkeypatch.delenv("AI_TRADING_MAX_DRAWDOWN_THRESHOLD", raising=False)
    module = _reload("ai_trading.risk")
    cfg = module.kelly.get_default_config()
    # relaxed loader should default to 0.08 when drawdown env is absent
    assert cfg.max_drawdown_threshold == 0.08


def test_kelly_default_config_uses_env_when_available(monkeypatch, default_env):
    """Once the environment provides the drawdown threshold we surface it."""

    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.25")
    module = _reload("ai_trading.risk")
    module.kelly.reset_default_config()
    cfg = module.kelly.get_default_config()
    assert cfg.max_drawdown_threshold == 0.25
