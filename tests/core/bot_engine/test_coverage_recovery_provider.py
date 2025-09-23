from datetime import UTC, datetime, timedelta
import types

import pytest

import ai_trading.core.bot_engine as bot_engine
import ai_trading.data.market_calendar as market_calendar
from ai_trading.utils import base as base_utils

pd = pytest.importorskip("pandas")


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        base = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
        if tz is None:
            return base.replace(tzinfo=None)
        return base.astimezone(tz)


class _DummyMonitor:
    def is_disabled(self, *args, **kwargs):
        return False

    def record_failure(self, *args, **kwargs):
        return None

    def disable(self, *args, **kwargs):
        return None

    def record_switchover(self, *args, **kwargs):
        return None


def _annotate(frame: pd.DataFrame) -> pd.DataFrame:
    frame.attrs["data_provider"] = "yahoo"
    frame.attrs["data_feed"] = "yahoo"
    frame.attrs["fallback_provider"] = "yahoo"
    frame.attrs["fallback_feed"] = "yahoo"
    return frame


def test_coverage_recovery_uses_backup_provider_annotation(monkeypatch, caplog):
    session_end = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
    session_start = session_end - timedelta(minutes=180)

    primary_times = pd.date_range(start=session_start, periods=10, freq="1min", tz=UTC)
    primary_df = pd.DataFrame(
        {
            "timestamp": primary_times,
            "open": [1.0] * len(primary_times),
            "high": [1.1] * len(primary_times),
            "low": [0.9] * len(primary_times),
            "close": [1.0] * len(primary_times),
            "volume": [50] * len(primary_times),
        }
    )

    fallback_times = pd.date_range(start=session_start, periods=200, freq="1min", tz=UTC)
    fallback_template = _annotate(
        pd.DataFrame(
            {
                "timestamp": fallback_times,
                "open": [1.0] * len(fallback_times),
                "high": [1.2] * len(fallback_times),
                "low": [0.8] * len(fallback_times),
                "close": [1.05] * len(fallback_times),
                "volume": [75] * len(fallback_times),
            }
        )
    )

    def _fallback_copy() -> pd.DataFrame:
        frame = fallback_template.copy()
        return _annotate(frame)

    call_history: list = []

    def _fake_get_minute_df(symbol, start_dt, end_dt, feed=None, **_):
        call_history.append(feed)
        if feed == "sip":
            bot_engine.data_fetcher_module._SIP_UNAUTHORIZED = True
            return _fallback_copy()
        return primary_df.copy()

    cached_feeds: list[str] = []

    monkeypatch.setattr(bot_engine, "datetime", _FixedDatetime)
    monkeypatch.setattr(market_calendar, "rth_session_utc", lambda *_: (session_start, session_end))
    monkeypatch.setattr(market_calendar, "previous_trading_session", lambda date: date - timedelta(days=1))
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_prefer_feed_this_cycle", lambda: None)
    monkeypatch.setattr(bot_engine, "_cache_cycle_fallback_feed", lambda feed: cached_feeds.append(feed))
    monkeypatch.setattr(bot_engine, "_sip_authorized", lambda: True)
    monkeypatch.setattr(bot_engine, "provider_monitor", _DummyMonitor())
    monkeypatch.setattr(bot_engine, "get_minute_df", _fake_get_minute_df)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "_sip_configured", lambda: True)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "_SIP_UNAUTHORIZED", False, raising=False)
    monkeypatch.setattr(bot_engine, "_SIP_UNAUTHORIZED_LOGGED", False, raising=False)
    monkeypatch.setattr(bot_engine, "_GLOBAL_INTRADAY_FALLBACK_FEED", None, raising=False)

    cfg = types.SimpleNamespace(
        data_feed="iex",
        minute_gap_backfill=None,
        intraday_lookback_minutes=120,
        intraday_indicator_window_minutes=120,
        longest_intraday_indicator_minutes=120,
        alpaca_feed_failover=(),
        market_cache_enabled=False,
    )
    monkeypatch.setattr(bot_engine, "CFG", cfg)
    monkeypatch.setattr(bot_engine, "S", cfg)
    monkeypatch.setattr(
        bot_engine,
        "state",
        types.SimpleNamespace(minute_feed_cache={}, minute_feed_cache_ts={}),
    )
    monkeypatch.setattr(bot_engine.staleness, "_ensure_data_fresh", lambda *args, **kwargs: None)

    with caplog.at_level("WARNING"):
        result = bot_engine.fetch_minute_df_safe("AAPL")

    assert isinstance(result, pd.DataFrame)
    assert call_history.count("sip") >= 1
    assert result.attrs.get("data_provider") == "yahoo"
    expected_event = "COVERAGE_RECOVERY_YAHOO"
    assert any(
        record.message == expected_event and getattr(record, "new_feed", None) == "yahoo"
        for record in caplog.records
    )
    assert cached_feeds == ["yahoo"]
    assert bot_engine.state.minute_feed_cache.get("iex") == "yahoo"
