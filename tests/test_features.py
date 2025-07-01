import sys
import types
import pandas as pd
from features import build_features_pipeline

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules.setdefault("dotenv", dotenv_stub)
ps_stub = types.ModuleType("pydantic_settings")
ps_stub.BaseSettings = object
ps_stub.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", ps_stub)
validate_stub = types.ModuleType("validate_env")
validate_stub.settings = types.SimpleNamespace(
    ALPACA_API_KEY="k",
    ALPACA_SECRET_KEY="s",
    ALPACA_BASE_URL="http://example.com",
    WEBHOOK_SECRET="w",
)
sys.modules.setdefault("validate_env", validate_stub)
import os
os.environ.setdefault("ALPACA_BASE_URL", "http://example.com")
os.environ.setdefault("WEBHOOK_SECRET", "dummy")

import pytest

@pytest.fixture(autouse=True)
def reload_utils_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "utils", types.ModuleType("utils"))
    yield


def test_features_pipeline():
    data = {
        'open': list(range(100, 120)),
        'high': list(range(101, 121)),
        'low': list(range(99, 119)),
        'close': [x + 0.5 for x in range(100, 120)],
        'volume': list(range(1000, 1020))
    }
    df = pd.DataFrame(data)
    df = build_features_pipeline(df, 'TEST')
    assert all(col in df.columns for col in ['macd', 'macds', 'atr', 'vwap']), "Missing computed columns"
    assert not df[['macd', 'macds', 'atr', 'vwap']].isnull().all().any(), "Indicators have all NaNs"
    print(df.tail())


if __name__ == "__main__":
    test_features_pipeline()
