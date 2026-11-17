import sys
import types
from typing import Any, cast

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.features.indicators import compute_vwap

ps_stub = types.ModuleType("pydantic_settings")
setattr(cast(Any, ps_stub), "BaseSettings", object)
setattr(cast(Any, ps_stub), "SettingsConfigDict", dict)
sys.modules.setdefault("pydantic_settings", ps_stub)

pytestmark = pytest.mark.usefixtures("default_env", "features_env")



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




def test_intraday_vwap_resets_by_timestamp_date():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-04-17T14:30:00Z",
                    "2026-04-17T14:31:00Z",
                    "2026-04-20T14:30:00Z",
                ],
                utc=True,
            ),
            "high": [10.0, 20.0, 100.0],
            "low": [10.0, 20.0, 100.0],
            "close": [10.0, 20.0, 100.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )

    out = compute_vwap(df)

    assert out["vwap"].tolist() == [10.0, 15.0, 100.0]


