import sys
import types
import pandas as pd
import pytest
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
    ALPACA_API_KEY="test_api_key_placeholder",
    ALPACA_SECRET_KEY="test_secret_placeholder",
    ALPACA_BASE_URL="http://example.com",
    WEBHOOK_SECRET="test_webhook_placeholder",
)
sys.modules.setdefault("validate_env", validate_stub)

pytestmark = pytest.mark.usefixtures("default_env", "features_env")

import pytest

@pytest.fixture(autouse=True)
def reload_utils_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "utils", types.ModuleType("utils"))
    yield


@pytest.fixture(autouse=True)
def features_env(monkeypatch):
    """Set environment vars required for feature tests."""
    monkeypatch.setenv("ALPACA_BASE_URL", "http://example.com")
    monkeypatch.setenv("WEBHOOK_SECRET", "dummy")
    yield


def test_features_pipeline():
    n = 60
    data = {
        'open': list(range(100, 100 + n)),
        'high': list(range(101, 101 + n)),
        'low': list(range(99, 99 + n)),
        'close': [x + 0.5 for x in range(100, 100 + n)],
        'volume': list(range(1000, 1000 + n))
    }
    df = pd.DataFrame(data)
    df = build_features_pipeline(df, 'TEST')
    assert all(col in df.columns for col in ['macd', 'macds', 'atr', 'vwap']), "Missing computed columns"
    assert not df[['macd', 'macds', 'atr', 'vwap']].isnull().all().any(), "Indicators have all NaNs"
    na_counts = df[['macd', 'atr', 'vwap', 'macds']].isna().sum()
    assert (na_counts <= 20).all(), f"Excessive NaNs in features: {na_counts}" 


if __name__ == "__main__":
    test_features_pipeline()
