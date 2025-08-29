import pandas as pd
import pytest
from datetime import datetime, timedelta, UTC


def test_get_bars_falls_back_when_settings_missing(monkeypatch, caplog):
    from ai_trading.data import fetch

    monkeypatch.setattr(fetch, "get_settings", lambda: None)

    called = {}

    def fake_fetch(symbol, start, end, timeframe, *, feed=None, adjustment=None):
        called["feed"] = feed
        called["adjustment"] = adjustment
        return pd.DataFrame()

    monkeypatch.setattr(fetch, "_fetch_bars", fake_fetch)

    start = datetime.now(UTC) - timedelta(minutes=1)
    end = datetime.now(UTC)
    with caplog.at_level("WARNING"):
        df = fetch.get_bars("AAPL", "1Min", start, end)
    assert df.empty
    assert called["feed"] == fetch._DEFAULT_FEED
    assert called["adjustment"] == "raw"


def test_main_exits_when_env_invalid(monkeypatch):
    import ai_trading.main as m

    def bad_validate():
        raise RuntimeError("missing env")

    monkeypatch.setattr(m, "validate_required_env", bad_validate)
    with pytest.raises(SystemExit) as excinfo:
        m.main([])
    assert excinfo.value.code == 1
