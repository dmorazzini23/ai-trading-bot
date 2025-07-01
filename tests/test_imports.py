import importlib

MODULES = [
    "requests",
    "urllib3.util.retry",
    "alpaca_trade_api.rest",
    "run",
    "alpaca_api",
]

def test_imports():
    for mod in MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            raise AssertionError(f"Failed to import {mod}: {exc}")

