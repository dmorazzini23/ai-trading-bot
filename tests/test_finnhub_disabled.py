import datetime as dt
import logging
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import fetch as data_fetcher


@pytest.fixture(autouse=True)
def _force_window(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)


def test_get_minute_df_raises_when_finnhub_disabled(monkeypatch, caplog):
    data_fetcher._SKIPPED_SYMBOLS.clear()
    data_fetcher._EMPTY_BAR_COUNTS.clear()
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")

    def fail_fetch(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("Finnhub fetch should be skipped")

    monkeypatch.setattr(data_fetcher.fh_fetcher, "fetch", fail_fetch)
    monkeypatch.setattr(data_fetcher, "_fetch_bars", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", lambda *a, **k: pd.DataFrame())

    with caplog.at_level(logging.INFO):
        with pytest.raises(data_fetcher.EmptyBarsError):
            data_fetcher.get_minute_df(
                "AAPL",
                dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
                dt.datetime(2023, 1, 2, tzinfo=dt.UTC),
            )
    assert any(r.message == "FINNHUB_DISABLED_NO_DATA" for r in caplog.records)
    key = ("AAPL", "1Min")
    assert key in data_fetcher._SKIPPED_SYMBOLS
    assert data_fetcher._EMPTY_BAR_COUNTS.get(key, 0) >= 1


def test_duplicate_info_suppressed(monkeypatch, caplog):
    from ai_trading.logging import logger_once

    monkeypatch.setattr(logger_once, "_emitted_keys", {})
    data_fetcher._SKIPPED_SYMBOLS.clear()
    data_fetcher._EMPTY_BAR_COUNTS.clear()
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setattr(data_fetcher, "_fetch_bars", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", lambda *a, **k: pd.DataFrame())
    start = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2023, 1, 2, tzinfo=dt.UTC)
    with caplog.at_level(logging.INFO):
        with pytest.raises(data_fetcher.EmptyBarsError):
            data_fetcher.get_minute_df("MSFT", start, end)
        with pytest.raises(data_fetcher.EmptyBarsError):
            data_fetcher.get_minute_df("MSFT", start, end)
    infos = [r for r in caplog.records if r.message == "FINNHUB_DISABLED_NO_DATA"]
    assert len(infos) == 1
