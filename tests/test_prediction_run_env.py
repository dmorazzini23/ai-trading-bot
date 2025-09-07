import importlib
import pytest

def test_run_valid_env(monkeypatch):
    monkeypatch.setenv("TRADING_ENV", "paper")
    executor = importlib.import_module("ai_trading.prediction.executor")
    assert executor.run() == "paper"

def test_run_invalid_env(monkeypatch):
    monkeypatch.setenv("TRADING_ENV", "invalid")
    executor = importlib.import_module("ai_trading.prediction.executor")
    with pytest.raises(ValueError):
        executor.run()
