"""Test for Alpaca import handling when the required SDK is missing."""

import sys
from types import ModuleType
from typing import Any, cast

def test_ai_trading_import_without_alpaca(monkeypatch):
    """Runtime modules fail fast instead of installing Alpaca stand-ins."""
    restore_modules: dict[str, ModuleType | None] = {}
    target_prefixes = ("alpaca", "ai_trading")
    for name, module in list(sys.modules.items()):
        if name == target_prefixes[0] or name.startswith(f"{target_prefixes[0]}."):
            restore_modules[name] = module
        elif name == target_prefixes[1] or name.startswith(f"{target_prefixes[1]}."):
            restore_modules[name] = module

    try:
        for name in list(restore_modules):
            sys.modules.pop(name, None)

        # Simulate missing Alpaca package.
        sys.modules["alpaca"] = None
        monkeypatch.setenv("TESTING", "true")

        # The package itself remains importable for diagnostics.
        import ai_trading

        assert ai_trading is not None

        import ai_trading.alpaca_api as alpaca_api

        assert alpaca_api.ALPACA_AVAILABLE is False
        assert alpaca_api.TimeFrame is None
        assert alpaca_api.StockBarsRequest is None

        # Import diagnostics may load runtime modules, but must not install an
        # Alpaca stand-in over the missing SDK sentinel.
        assert sys.modules.get("alpaca") is None
    finally:
        for name in list(sys.modules):
            if name == target_prefixes[0] or name.startswith(f"{target_prefixes[0]}."):
                sys.modules.pop(name, None)
            elif name == target_prefixes[1] or name.startswith(f"{target_prefixes[1]}."):
                sys.modules.pop(name, None)
        for module_name, module in restore_modules.items():
            sys.modules[module_name] = cast(Any, module)
