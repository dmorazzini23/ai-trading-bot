import pytest

pd = pytest.importorskip("pandas")
from ai_trading.core import bot_engine as bot


def test_compute_time_range():
    """compute_time_range should span the requested minutes."""
    start, end = bot.compute_time_range(5)
    assert (end - start).total_seconds() == 300


def test_get_latest_close_edge_cases():
    """get_latest_close handles missing data gracefully."""
    assert bot.get_latest_close(pd.DataFrame()) == 0.0
    assert bot.get_latest_close(None) == 0.0
    df = pd.DataFrame({"close": [1.5]}, index=[pd.Timestamp("2024-01-01")])
    assert bot.get_latest_close(df) == 1.5


def test_fetch_minute_df_safe_market_closed(monkeypatch):
    """Error is raised when no data is returned."""
    monkeypatch.setattr(bot, "get_minute_df", lambda *a, **k: pd.DataFrame())
    with pytest.raises(bot.DataFetchError):
        bot.fetch_minute_df_safe("AAPL")


def test_fetch_minute_df_safe_open(monkeypatch):
    """DataFrame is returned when the market is open."""
    df = pd.DataFrame({"close": [1]}, index=[pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=1)])
    monkeypatch.setattr(bot, "get_minute_df", lambda symbol, start_date, end_date: df)
    monkeypatch.setattr(bot, "_count_trading_minutes", lambda *a, **k: 1)
    monkeypatch.setattr(bot, "_expected_minute_bars_window", lambda *a, **k: 1)
    monkeypatch.setattr(bot.staleness, "_ensure_data_fresh", lambda *a, **k: None)
    result = bot.fetch_minute_df_safe("AAPL")
    pd.testing.assert_frame_equal(result, df)
