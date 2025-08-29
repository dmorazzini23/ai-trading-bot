import pytest
from datetime import datetime, timedelta, UTC


def test_get_bars_raises_when_settings_missing(monkeypatch):
    from ai_trading.data import fetch

    monkeypatch.setattr(fetch, "get_settings", lambda: None)

    start = datetime.now(UTC) - timedelta(minutes=1)
    end = datetime.now(UTC)
    with pytest.raises(RuntimeError):
        fetch.get_bars("AAPL", "1Min", start, end)


def test_main_exits_when_env_invalid(monkeypatch):
    import ai_trading.main as m

    def bad_validate():
        raise RuntimeError("missing env")

    monkeypatch.setattr(m, "validate_required_env", bad_validate)
    with pytest.raises(SystemExit) as excinfo:
        m.main([])
    assert excinfo.value.code == 1
