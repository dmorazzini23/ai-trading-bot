from __future__ import annotations

import importlib

_REGISTRY = {
    "yfinance": "ai_trading.data.providers.yfinance_provider",
}

def resolve(name: str):
    mod_path = _REGISTRY.get(name)
    if not mod_path:
        raise KeyError(f"Unknown data provider: {name}")
    mod = importlib.import_module(mod_path)
    return getattr(mod, "Provider")

