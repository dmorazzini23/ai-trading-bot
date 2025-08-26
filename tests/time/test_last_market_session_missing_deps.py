import builtins
import pytest

pd = pytest.importorskip("pandas")

import ai_trading.utils.time as t


def test_missing_calendars_returns_none(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ai_trading.market.calendars":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert t.last_market_session(pd.Timestamp.now(tz="UTC")) is None


def test_missing_pandas_returns_none(monkeypatch):
    monkeypatch.setattr(t, "load_pandas", lambda: None)
    assert t.last_market_session(pd.Timestamp.now(tz="UTC")) is None
