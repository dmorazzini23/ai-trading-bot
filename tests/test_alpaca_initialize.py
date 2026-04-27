import importlib
import pytest

from ai_trading import alpaca_api


def test_initialize_missing_sdk(monkeypatch):
    def fail(_name, package=None):  # pragma: no cover - used in test
        raise ModuleNotFoundError("missing")

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        alpaca_api.initialize()
