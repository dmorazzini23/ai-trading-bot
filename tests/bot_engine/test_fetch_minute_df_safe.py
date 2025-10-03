import logging
import sys
import types
from datetime import UTC, datetime, timedelta

import pytest
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.utils.prof import SoftBudget

pytest.importorskip("pandas")

sys.modules.setdefault(
    "portalocker",
    types.SimpleNamespace(lock=lambda *a, **k: None, unlock=lambda *a, **k: None, LOCK_EX=1),
)
sys.modules.setdefault("bs4", types.SimpleNamespace(BeautifulSoup=object))
sys.modules.setdefault(
    "flask",
    types.SimpleNamespace(
        Flask=type(
            "Flask",
            (),
            {
                "__init__": lambda self, *a, **k: None,
                "route": lambda self, *a, **k: (lambda fn: fn),
                "after_request": lambda self, fn: fn,
            },
        )
    ),
)

from ai_trading.core import bot_engine
from ai_trading.guards import staleness


def test_minute_frame_attrs_extracts_known_keys():
    pd = load_pandas()
    frame = pd.DataFrame({"close": [1.0]})
    frame.attrs["data_provider"] = "alpaca_iex"
    frame.attrs["source_label"] = "alpaca_iex"
    frame.attrs["ignored"] = "should_not_appear"

    attrs = bot_engine._minute_frame_attrs(frame)

    assert attrs == {
        "data_provider": "alpaca_iex",
        "source_label": "alpaca_iex",
    }


def test_log_iex_minute_stale_includes_attrs(monkeypatch):
    pd = load_pandas()
    frame = pd.DataFrame({"close": [1.0]})
    frame.attrs["data_feed"] = "iex"
    frame.attrs["fallback_feed"] = "sip"

    class _Recorder:
        def __init__(self):
            self.calls: list[tuple[str, dict[str, object] | None]] = []

        def warning(self, message: str, *, extra: dict[str, object] | None = None):
            self.calls.append((message, extra))

    recorder = _Recorder()
    monkeypatch.setattr(bot_engine, "logger", recorder)

    bot_engine._log_iex_minute_stale(
        symbol="AAPL",
        age_seconds=720,
        retry_feed="sip",
        frame=frame,
    )

    assert recorder.calls
    message, extra = recorder.calls[-1]
    assert message == "IEX_MINUTE_DATA_STALE"
    assert extra is not None
    assert extra["symbol"] == "AAPL"
    assert extra["retry_feed"] == "sip"
    assert extra["minute_attrs"] == {"data_feed": "iex", "fallback_feed": "sip"}


@pytest.fixture(autouse=True)
def _short_intraday_defaults(monkeypatch):
    """Keep intraday lookback small for unit test fixtures."""

    monkeypatch.setattr(bot_engine, "_LONGEST_INTRADAY_INDICATOR_MINUTES", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "intraday_lookback_minutes", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "intraday_indicator_window_minutes", 0, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "minute_gap_backfill", None, raising=False)
    try:
        monkeypatch.setattr(bot_engine, "S", bot_engine.CFG, raising=False)
    except Exception:
        pass
    try:
        monkeypatch.setattr(bot_engine, "state", bot_engine.BotState(), raising=False)
    except Exception:
        pass

    from ai_trading.data import market_calendar
    from ai_trading.data.provider_monitor import ProviderMonitor
    from ai_trading.utils import base as base_utils

    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "provider_monitor",
        ProviderMonitor(),
        raising=False,
    )
    try:
        bot_engine._reset_cycle_cache()
    except Exception:
        pass
    try:
        cache_override = getattr(bot_engine.data_fetcher_module, "_cycle_feed_override", None)
        if isinstance(cache_override, dict):
            cache_override.clear()
    except Exception:
        pass

    def _tiny_session(current_date):
        end = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
        return end - timedelta(minutes=1), end

    monkeypatch.setattr(market_calendar, "rth_session_utc", _tiny_session, raising=False)
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
        raising=False,
    )
    monkeypatch.setattr(base_utils, "is_market_open", lambda: False, raising=False)


def _sample_df():
    pd = load_pandas()
    return pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp("2024-01-01", tz="UTC")])


def test_fetch_minute_df_safe_returns_dataframe(monkeypatch):
    pd = load_pandas()
    monkeypatch.setattr(
        bot_engine, "get_minute_df", lambda s, start, end, **_: _sample_df()
    )
    monkeypatch.setattr(staleness, "_ensure_data_fresh", lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None)
    result = bot_engine.fetch_minute_df_safe("AAPL")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_fetch_minute_df_safe_raises_on_empty(monkeypatch):
    pd = load_pandas()
    monkeypatch.setattr(
        bot_engine, "get_minute_df", lambda s, start, end, **_: pd.DataFrame()
    )
    monkeypatch.setattr(staleness, "_ensure_data_fresh", lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None)
    with pytest.raises(bot_engine.DataFetchError):
        bot_engine.fetch_minute_df_safe("AAPL")


def test_fetch_minute_df_safe_accepts_data_within_configured_tolerance(monkeypatch):
    pd = load_pandas()
    base_now = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    idx = pd.date_range(end=base_now - timedelta(minutes=2), periods=5, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0] * len(idx),
            "high": [1.1] * len(idx),
            "low": [0.9] * len(idx),
            "close": [1.0] * len(idx),
            "volume": [100] * len(idx),
        },
        index=idx,
    )

    captured: dict[str, object] = {}

    def capture_ensure(df_arg, max_age_seconds, *, symbol=None, now=None, tz=None):
        captured["max_age"] = max_age_seconds
        captured["symbol"] = symbol
        return None

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda *a, **k: df.copy())
    monkeypatch.setattr(staleness, "_ensure_data_fresh", capture_ensure)
    monkeypatch.setattr(
        bot_engine,
        "_minute_data_freshness_limit",
        lambda: 180,
        raising=False,
    )

    result = bot_engine.fetch_minute_df_safe("AAPL")

    assert captured["max_age"] == 180
    assert captured["symbol"] == "AAPL"
    pd.testing.assert_frame_equal(result, df)


def test_fetch_minute_df_safe_raises_when_exceeding_configured_tolerance(monkeypatch):
    pd = load_pandas()
    base_now = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    idx = pd.date_range(end=base_now - timedelta(minutes=10), periods=5, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0] * len(idx),
            "high": [1.1] * len(idx),
            "low": [0.9] * len(idx),
            "close": [1.0] * len(idx),
            "volume": [100] * len(idx),
        },
        index=idx,
    )

    def stale_ensure(df_arg, max_age_seconds, *, symbol=None, now=None, tz=None):
        assert max_age_seconds == 120
        raise RuntimeError("age=130s")

    config = types.SimpleNamespace(
        market_cache_enabled=False,
        intraday_lookback_minutes=5,
        data_feed="iex",
        alpaca_feed_failover=(),
    )

    monkeypatch.setattr(bot_engine, "CFG", config, raising=False)
    monkeypatch.setattr(bot_engine, "S", config, raising=False)
    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda *a, **k: df.copy())
    monkeypatch.setattr(staleness, "_ensure_data_fresh", stale_ensure)
    monkeypatch.setattr(
        bot_engine,
        "_minute_data_freshness_limit",
        lambda: 120,
        raising=False,
    )

    with pytest.raises(bot_engine.DataFetchError) as excinfo:
        bot_engine.fetch_minute_df_safe("AAPL")

    detail = getattr(excinfo.value, "detail", "")
    assert "age=130s" in detail


def test_fetch_minute_df_safe_recovers_from_single_stale(monkeypatch, caplog):
    pd = load_pandas()

    base_now = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    idx_stale = pd.date_range(
        end=base_now - timedelta(minutes=15), periods=5, freq="min", tz="UTC"
    )
    idx_fresh = pd.date_range(
        end=base_now - timedelta(minutes=1), periods=5, freq="min", tz="UTC"
    )

    stale_df = pd.DataFrame(
        {
            "open": [1.0] * len(idx_stale),
            "high": [1.1] * len(idx_stale),
            "low": [0.9] * len(idx_stale),
            "close": [1.0] * len(idx_stale),
            "volume": [100] * len(idx_stale),
        },
        index=idx_stale,
    )
    fresh_df = pd.DataFrame(
        {
            "open": [1.0] * len(idx_fresh),
            "high": [1.1] * len(idx_fresh),
            "low": [0.9] * len(idx_fresh),
            "close": [1.0] * len(idx_fresh),
            "volume": [100] * len(idx_fresh),
        },
        index=idx_fresh,
    )

    calls: list[tuple[datetime, datetime, dict[str, object]]] = []

    def fake_get_minute_df(symbol, start, end, **kwargs):
        calls.append((start, end, kwargs))
        if len(calls) <= 2:
            return stale_df.copy()
        return fresh_df.copy()

    ensure_calls: list[pd.Index] = []

    def fake_ensure(df, max_age_seconds, *, symbol=None, now=None, tz=None):
        ensure_calls.append(df.index)
        if df.index.equals(idx_stale):
            raise RuntimeError("age=900s")
        return None

    config = types.SimpleNamespace(
        market_cache_enabled=False,
        intraday_lookback_minutes=5,
        data_feed="iex",
        alpaca_feed_failover=("sip",),
    )

    monkeypatch.setattr(bot_engine, "CFG", config, raising=False)
    monkeypatch.setattr(bot_engine, "S", config, raising=False)
    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(staleness, "_ensure_data_fresh", fake_ensure)
    from ai_trading.data import market_calendar

    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (base_now - timedelta(minutes=5), base_now),
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: base_now.date(),
    )

    with caplog.at_level(logging.INFO):
        result = bot_engine.fetch_minute_df_safe("AAPL")

    assert len(calls) >= 2
    assert ensure_calls[0].equals(idx_stale)
    assert ensure_calls[-1].equals(idx_fresh)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_index_equal(result.index, idx_fresh)
    assert any(rec.message == "FETCH_MINUTE_STALE_RECOVERED" for rec in caplog.records)


def test_fetch_minute_df_safe_raises_when_all_retries_stale(monkeypatch):
    pd = load_pandas()

    base_now = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    idx_stale = pd.date_range(
        end=base_now - timedelta(minutes=20), periods=5, freq="min", tz="UTC"
    )

    stale_df = pd.DataFrame(
        {
            "open": [1.0] * len(idx_stale),
            "high": [1.1] * len(idx_stale),
            "low": [0.9] * len(idx_stale),
            "close": [1.0] * len(idx_stale),
            "volume": [100] * len(idx_stale),
        },
        index=idx_stale,
    )

    calls: list[tuple[datetime, datetime, dict[str, object]]] = []

    def fake_get_minute_df(symbol, start, end, **kwargs):
        calls.append((start, end, kwargs))
        return stale_df.copy()

    ensure_calls: list[int] = []

    def fake_ensure(df, max_age_seconds, *, symbol=None, now=None, tz=None):
        ensure_calls.append(len(ensure_calls))
        raise RuntimeError(f"age=900s-call{len(ensure_calls)}")

    config = types.SimpleNamespace(
        market_cache_enabled=False,
        intraday_lookback_minutes=5,
        data_feed="iex",
        alpaca_feed_failover=(),
    )

    monkeypatch.setattr(bot_engine, "CFG", config, raising=False)
    monkeypatch.setattr(bot_engine, "S", config, raising=False)
    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(staleness, "_ensure_data_fresh", fake_ensure)

    with pytest.raises(bot_engine.DataFetchError) as excinfo:
        bot_engine.fetch_minute_df_safe("AAPL")

    assert len(calls) >= 2  # initial + retry
    assert getattr(excinfo.value, "fetch_reason", None) == "stale_minute_data"
    assert getattr(excinfo.value, "symbol", None) == "AAPL"
    detail = getattr(excinfo.value, "detail", "")
    assert "age=900s-call1" in detail
    assert "age=900s-call2" in detail


def test_fetch_minute_df_safe_after_hours_uses_session_close(monkeypatch):
    pd = load_pandas()

    session_end = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    session_start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    base_now = session_end + timedelta(hours=2)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    expected_bars = int((session_end - session_start).total_seconds() // 60)
    idx = pd.date_range(start=session_start, periods=expected_bars, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0 + i * 0.01 for i in range(expected_bars)],
            "high": [1.1 + i * 0.01 for i in range(expected_bars)],
            "low": [0.9 + i * 0.01 for i in range(expected_bars)],
            "close": [1.0 + i * 0.01 for i in range(expected_bars)],
            "volume": [100 + i for i in range(expected_bars)],
        },
        index=idx,
    )

    captured: dict[str, object] = {}

    def capture_freshness(df_arg, max_age_seconds, *, symbol=None, now=None, tz=None):
        captured["max_age"] = max_age_seconds
        captured["now"] = now
        return None

    from ai_trading.data import market_calendar

    from ai_trading.utils import base as base_utils

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(base_utils, "is_market_open", lambda: False)
    monkeypatch.setattr(
        bot_engine, "get_minute_df", lambda s, start, end, **_: df.copy()
    )
    monkeypatch.setattr(staleness, "_ensure_data_fresh", capture_freshness)
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (session_start, session_end),
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: session_end.date(),
    )
    monkeypatch.setattr(
        bot_engine,
        "_minute_data_freshness_limit",
        lambda: 900,
        raising=False,
    )

    result = bot_engine.fetch_minute_df_safe("AMGN")

    assert captured["max_age"] == 900
    assert captured["now"] == session_end
    pd.testing.assert_frame_equal(result, df)


def test_fetch_minute_df_safe_sparse_minute_data_triggers_sip_fallback(
    monkeypatch, caplog
):
    pd = load_pandas()

    base_now = datetime(2024, 1, 3, 17, 30, tzinfo=UTC)
    session_start = datetime(2024, 1, 3, 14, 30, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    expected = int((base_now - session_start).total_seconds() // 60)

    calls: list[tuple[str, object, object, object]] = []

    def fake_get_minute_df(symbol: str, start, end, feed=None, **_):
        calls.append((symbol, start, end, feed))
        if feed == "sip":
            idx = pd.date_range(
                end=end - timedelta(minutes=1),
                periods=expected,
                freq="min",
                tz="UTC",
            )
            return pd.DataFrame(
                {
                    "close": list(range(len(idx))),
                    "volume": [100] * len(idx),
                },
                index=idx,
            )
        idx = pd.date_range(start=start, periods=10, freq="min", tz="UTC")
        return pd.DataFrame(
            {
                "close": [1.0] * len(idx),
                "volume": [100] * len(idx),
            },
            index=idx,
        )

    from ai_trading.data import market_calendar
    from ai_trading.data.provider_monitor import ProviderMonitor
    from ai_trading.utils import base as base_utils

    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "provider_monitor",
        ProviderMonitor(),
        raising=False,
    )

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )
    monkeypatch.setattr(bot_engine.CFG, "data_feed", "iex", raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_sip_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (session_start, base_now + timedelta(hours=4)),
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
    )

    with caplog.at_level(logging.WARNING):
        result = bot_engine.fetch_minute_df_safe("AAPL")

    assert len(calls) == 2
    assert calls[1][3] == "sip"
    assert len(result) == expected
    assert any(rec.message == "MINUTE_DATA_COVERAGE_WARNING" for rec in caplog.records)


def test_fetch_minute_df_safe_aborts_when_fallback_remains_sparse(monkeypatch, caplog):
    pd = load_pandas()

    base_now = datetime(2024, 1, 3, 18, 30, tzinfo=UTC)
    session_start = datetime(2024, 1, 3, 14, 30, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    calls: list[tuple[str, object, object, object]] = []

    def fake_get_minute_df(symbol: str, start, end, feed=None, **_):
        calls.append((symbol, start, end, feed))
        idx = pd.date_range(
            end=end - timedelta(minutes=2),
            periods=5,
            freq="min",
            tz="UTC",
        )
        return pd.DataFrame(
            {
                "close": [1.0] * len(idx),
                "volume": [100] * len(idx),
            },
            index=idx,
        )

    from ai_trading.data import market_calendar
    from ai_trading.data.provider_monitor import ProviderMonitor
    from ai_trading.utils import base as base_utils

    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "provider_monitor",
        ProviderMonitor(),
        raising=False,
    )

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )
    monkeypatch.setattr(bot_engine.CFG, "data_feed", "iex", raising=False)
    monkeypatch.setattr(bot_engine.CFG, "market_cache_enabled", False, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "intraday_lookback_minutes", 60, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "alpaca_feed_failover", (), raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_sip_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (session_start, base_now + timedelta(hours=4)),
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(bot_engine.DataFetchError) as excinfo:
            bot_engine.fetch_minute_df_safe("AAPL")

    assert len(calls) == 3
    assert calls[1][3] == "sip"
    assert calls[2][3] == "yahoo"
    assert getattr(excinfo.value, "fetch_reason", None) == "minute_data_low_coverage"
    assert getattr(excinfo.value, "symbol", None) == "AAPL"
    assert any(rec.message == "MINUTE_DATA_COVERAGE_ABORT" for rec in caplog.records)


def test_fetch_minute_df_safe_early_session_sparse_data_triggers_sip_fallback(
    monkeypatch, caplog
):
    pd = load_pandas()

    base_now = datetime(2024, 1, 3, 15, 0, tzinfo=UTC)
    session_start = datetime(2024, 1, 3, 14, 30, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    expected = int((base_now - session_start).total_seconds() // 60)

    calls: list[tuple[str, object, object, object]] = []

    def fake_get_minute_df(symbol: str, start, end, feed=None, **_):
        calls.append((symbol, start, end, feed))
        if feed == "sip":
            idx = pd.date_range(
                end=end - timedelta(minutes=1),
                periods=expected,
                freq="min",
                tz="UTC",
            )
            return pd.DataFrame(
                {
                    "close": list(range(len(idx))),
                    "volume": [100] * len(idx),
                },
                index=idx,
            )
        idx = pd.date_range(start=start, periods=5, freq="min", tz="UTC")
        return pd.DataFrame(
            {
                "close": [1.0] * len(idx),
                "volume": [100] * len(idx),
            },
            index=idx,
        )

    from ai_trading.data import market_calendar
    from ai_trading.data.provider_monitor import ProviderMonitor
    from ai_trading.utils import base as base_utils

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "provider_monitor",
        ProviderMonitor(),
        raising=False,
    )
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )
    monkeypatch.setattr(bot_engine.CFG, "data_feed", "iex", raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_sip_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (session_start, base_now + timedelta(hours=4)),
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
    )

    with caplog.at_level(logging.WARNING):
        result = bot_engine.fetch_minute_df_safe("AAPL")

    assert len(calls) == 2
    assert calls[1][3] == "sip"
    assert len(result) == expected
    assert any(rec.message == "MINUTE_DATA_COVERAGE_WARNING" for rec in caplog.records)


def test_fetch_minute_df_safe_reuses_cached_fallback_feed_within_cycle(
    monkeypatch, caplog
):
    pd = load_pandas()

    base_now = datetime(2024, 1, 3, 16, 0, tzinfo=UTC)
    session_start = base_now - timedelta(minutes=60)
    expected = int((base_now - session_start).total_seconds() // 60)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    calls: list[tuple[str, str]] = []
    fake_time = [0.0]

    from ai_trading.utils import prof as prof_utils

    def fake_monotonic() -> float:
        return fake_time[0]

    def advance(seconds: float) -> None:
        fake_time[0] += seconds

    monkeypatch.setattr(prof_utils.time, "monotonic", fake_monotonic)

    def fake_get_minute_df(symbol: str, start, end, feed=None, **_):
        feed_name = str(feed or "iex")
        calls.append((symbol, feed_name))
        if feed_name == "iex":
            advance(0.04)
            idx = pd.date_range(
                end=end - timedelta(minutes=1), periods=5, freq="min", tz="UTC"
            )
        else:
            advance(0.005)
            idx = pd.date_range(
                end=end - timedelta(minutes=1), periods=expected, freq="min", tz="UTC"
            )
        return pd.DataFrame(
            {"close": list(range(len(idx))), "volume": [100] * len(idx)},
            index=idx,
        )

    config = types.SimpleNamespace(
        market_cache_enabled=False,
        data_feed="iex",
        intraday_lookback_minutes=60,
        alpaca_feed_failover=(),
        minute_gap_backfill=None,
    )

    from ai_trading.data import market_calendar
    from ai_trading.data.provider_monitor import ProviderMonitor
    from ai_trading.utils import base as base_utils

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "provider_monitor",
        ProviderMonitor(),
        raising=False,
    )
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )
    monkeypatch.setattr(bot_engine, "CFG", config, raising=False)
    monkeypatch.setattr(bot_engine, "S", config, raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_sip_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (session_start, base_now + timedelta(hours=4)),
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
    )
    monkeypatch.setattr(bot_engine, "state", bot_engine.BotState())

    budget = SoftBudget(70)

    with caplog.at_level(logging.WARNING):
        with budget:
            first = bot_engine.fetch_minute_df_safe("AAPL")
            second = bot_engine.fetch_minute_df_safe("MSFT")

    assert isinstance(first, pd.DataFrame)
    assert isinstance(second, pd.DataFrame)
    assert [feed for _, feed in calls] == ["iex", "sip", "sip"]
    assert bot_engine.state.minute_feed_cache.get("iex") == "sip"
    warning_messages = [rec.message for rec in caplog.records]
    assert warning_messages.count("MINUTE_DATA_COVERAGE_WARNING") == 1
    assert not budget.over_budget()


def test_fetch_minute_df_safe_attempts_sip_before_yahoo_when_authorized(monkeypatch):
    pd = load_pandas()

    base_now = datetime(2024, 1, 3, 17, 0, tzinfo=UTC)
    session_start = base_now - timedelta(minutes=60)
    expected = int((base_now - session_start).total_seconds() // 60)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    calls: list[str] = []

    def fake_get_minute_df(symbol: str, start, end, feed=None, **_):
        feed_name = str(feed or "iex")
        calls.append(feed_name)
        if feed_name == "iex":
            idx = pd.date_range(
                end=end - timedelta(minutes=2), periods=10, freq="min", tz="UTC"
            )
        elif feed_name == "sip":
            idx = pd.date_range(
                end=end - timedelta(minutes=1), periods=10, freq="min", tz="UTC"
            )
        else:
            idx = pd.date_range(
                end=end - timedelta(minutes=1), periods=expected, freq="min", tz="UTC"
            )
        return pd.DataFrame(
            {"close": list(range(len(idx))), "volume": [100] * len(idx)},
            index=idx,
        )

    config = types.SimpleNamespace(
        market_cache_enabled=False,
        data_feed="iex",
        intraday_lookback_minutes=60,
        alpaca_feed_failover=(),
        minute_gap_backfill=None,
    )

    from ai_trading.data import market_calendar
    from ai_trading.utils import base as base_utils

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True, raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )
    monkeypatch.setattr(bot_engine, "CFG", config, raising=False)
    monkeypatch.setattr(bot_engine, "S", config, raising=False)
    monkeypatch.setattr(bot_engine, "state", bot_engine.BotState(), raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_sip_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (session_start, base_now + timedelta(hours=4)),
        raising=False,
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
        raising=False,
    )

    result = bot_engine.fetch_minute_df_safe("AAPL")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == expected
    assert calls == ["iex", "sip", "yahoo"]


def test_fetch_minute_df_safe_sip_success_skips_yahoo_and_caches_feed(monkeypatch):
    pd = load_pandas()

    base_now = datetime(2024, 1, 3, 16, 30, tzinfo=UTC)
    session_start = base_now - timedelta(minutes=45)
    expected = int((base_now - session_start).total_seconds() // 60)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    calls: list[str] = []
    cached_feeds: list[str | None] = []
    cached_pairs: list[tuple[str, str]] = []

    def fake_cache_cycle(feed: str | None, *, symbol: str | None = None):
        cached_feeds.append(feed)

    def fake_cache_fallback(symbol: str, feed: str):
        cached_pairs.append((symbol, feed))

    def fake_get_minute_df(symbol: str, start, end, feed=None, **_):
        feed_name = str(feed or "iex")
        calls.append(feed_name)
        periods = 5 if feed_name == "iex" else expected
        idx = pd.date_range(
            end=end - timedelta(minutes=1), periods=periods, freq="min", tz="UTC"
        )
        return pd.DataFrame(
            {"close": list(range(len(idx))), "volume": [100] * len(idx)},
            index=idx,
        )

    config = types.SimpleNamespace(
        market_cache_enabled=False,
        data_feed="iex",
        intraday_lookback_minutes=45,
        alpaca_feed_failover=(),
        minute_gap_backfill=None,
    )

    from ai_trading.data import market_calendar
    from ai_trading.utils import base as base_utils

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True, raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )
    monkeypatch.setattr(bot_engine, "CFG", config, raising=False)
    monkeypatch.setattr(bot_engine, "S", config, raising=False)
    monkeypatch.setattr(bot_engine, "state", bot_engine.BotState(), raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_sip_configured",
        lambda: True,
    )
    monkeypatch.setattr(bot_engine, "_cache_cycle_fallback_feed", fake_cache_cycle)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_cache_fallback",
        fake_cache_fallback,
    )
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (session_start, base_now + timedelta(hours=4)),
        raising=False,
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
        raising=False,
    )

    result = bot_engine.fetch_minute_df_safe("MSFT")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == expected
    assert calls == ["iex", "sip"]
    assert cached_feeds[-1] == "sip"
    assert cached_pairs[-1] == ("MSFT", "sip")
    assert bot_engine.state.minute_feed_cache.get("iex") == "sip"
    assert bot_engine.state.minute_feed_cache.get("sip") == "sip"
    cache_ts = getattr(bot_engine.state, "minute_feed_cache_ts", {})
    assert "iex" in cache_ts
    assert "sip" in cache_ts


def test_fetch_minute_df_safe_sip_and_yahoo_sparse_abort(monkeypatch):
    pd = load_pandas()

    base_now = datetime(2024, 1, 3, 18, 0, tzinfo=UTC)
    session_start = base_now - timedelta(minutes=60)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    calls: list[str] = []

    def fake_get_minute_df(symbol: str, start, end, feed=None, **_):
        feed_name = str(feed or "iex")
        calls.append(feed_name)
        periods = 5 if feed_name == "iex" else 10
        idx = pd.date_range(
            end=end - timedelta(minutes=2), periods=periods, freq="min", tz="UTC"
        )
        return pd.DataFrame(
            {"close": [1.0] * len(idx), "volume": [100] * len(idx)},
            index=idx,
        )

    config = types.SimpleNamespace(
        market_cache_enabled=False,
        data_feed="iex",
        intraday_lookback_minutes=60,
        alpaca_feed_failover=(),
        minute_gap_backfill=None,
    )

    from ai_trading.data import market_calendar
    from ai_trading.utils import base as base_utils

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True, raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )
    monkeypatch.setattr(bot_engine, "CFG", config, raising=False)
    monkeypatch.setattr(bot_engine, "S", config, raising=False)
    monkeypatch.setattr(bot_engine, "state", bot_engine.BotState(), raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_sip_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (session_start, base_now + timedelta(hours=4)),
        raising=False,
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
        raising=False,
    )

    with pytest.raises(bot_engine.DataFetchError) as excinfo:
        bot_engine.fetch_minute_df_safe("SPY")

    assert getattr(excinfo.value, "fetch_reason", None) == "minute_data_low_coverage"
    assert calls == ["iex", "sip", "yahoo"]
    assert "iex" not in bot_engine.state.minute_feed_cache
    assert "sip" not in bot_engine.state.minute_feed_cache
    assert "yahoo" not in bot_engine.state.minute_feed_cache
    cache_ts = getattr(bot_engine.state, "minute_feed_cache_ts", {})
    assert cache_ts.get("iex") is None
    assert cache_ts.get("sip") is None
    assert cache_ts.get("yahoo") is None


def test_fetch_minute_df_safe_logs_backup_provider_when_sip_unauthorized(
    monkeypatch, caplog
):
    pd = load_pandas()

    base_now = datetime(2024, 1, 3, 16, 0, tzinfo=UTC)
    session_start = base_now - timedelta(minutes=30)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    calls: list[str] = []

    def fake_get_minute_df(symbol: str, start, end, feed=None, **_):
        calls.append(str(feed or "iex"))
        start_dt = bot_engine.data_fetcher_module.ensure_datetime(start)
        end_dt = bot_engine.data_fetcher_module.ensure_datetime(end)
        if len(calls) == 1:
            idx = pd.date_range(
                start=start_dt,
                periods=2,
                freq="min",
                tz="UTC",
            )
            return pd.DataFrame(
                {
                    "timestamp": idx,
                    "open": [1.0] * len(idx),
                    "high": [1.0] * len(idx),
                    "low": [1.0] * len(idx),
                    "close": [1.0] * len(idx),
                    "volume": [10] * len(idx),
                }
            )
        bot_engine.data_fetcher_module._mark_fallback(
            symbol,
            "1Min",
            start_dt,
            end_dt,
            from_provider="alpaca_iex",
        )
        idx = pd.date_range(
            end=end_dt - timedelta(minutes=1),
            periods=30,
            freq="min",
            tz="UTC",
        )
        return pd.DataFrame(
            {
                "timestamp": idx,
                "open": [1.0] * len(idx),
                "high": [1.1] * len(idx),
                "low": [0.9] * len(idx),
                "close": [1.0] * len(idx),
                "volume": [100] * len(idx),
            }
        )

    from ai_trading.data import market_calendar
    from ai_trading.utils import base as base_utils

    bot_engine.data_fetcher_module._FALLBACK_WINDOWS.clear()
    bot_engine.data_fetcher_module._FALLBACK_METADATA.clear()
    bot_engine.data_fetcher_module._FALLBACK_UNTIL.clear()

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True, raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )
    monkeypatch.setattr(bot_engine.CFG, "data_feed", "iex", raising=False)
    monkeypatch.setattr(bot_engine.CFG, "alpaca_feed_failover", (), raising=False)
    monkeypatch.setattr(bot_engine.CFG, "market_cache_enabled", False, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "intraday_lookback_minutes", 30, raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_sip_configured",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_SIP_UNAUTHORIZED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine.data_fetcher_module.provider_monitor,
        "record_switchover",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (session_start, base_now),
        raising=False,
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
        raising=False,
    )

    with caplog.at_level(logging.WARNING):
        result = bot_engine.fetch_minute_df_safe("AAPL")

    assert len(calls) == 2
    assert calls[1] == "yahoo"
    assert not result.empty
    warning_records = [rec for rec in caplog.records if rec.message == "MINUTE_DATA_COVERAGE_WARNING"]
    assert warning_records, "coverage warning not emitted"
    assert warning_records[-1].fallback_provider == "yahoo"


def test_fetch_minute_df_safe_extends_window_for_long_indicators(monkeypatch):
    pd = load_pandas()

    base_now = datetime(2024, 1, 3, 16, 0, tzinfo=UTC)
    session_start = datetime(2024, 1, 3, 14, 30, tzinfo=UTC)
    prev_session_start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    session_duration = timedelta(hours=6, minutes=30)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    calls: list[tuple[datetime, datetime]] = []

    def fake_get_minute_df(symbol: str, start, end, **_):
        calls.append((start, end))
        periods = max(1, int((end - start).total_seconds() // 60))
        idx = pd.date_range(
            end=end - timedelta(minutes=1), periods=periods, freq="min", tz="UTC"
        )
        return pd.DataFrame(
            {
                "close": [1.0 + i * 0.01 for i in range(len(idx))],
                "volume": [100 + i for i in range(len(idx))],
            },
            index=idx,
        )

    def fake_ensure(df, max_age_seconds, *, symbol=None, now=None, tz=None):
        return None

    config = types.SimpleNamespace(
        market_cache_enabled=False,
        intraday_lookback_minutes=120,
        intraday_indicator_window_minutes=0,
        data_feed="iex",
        alpaca_feed_failover=(),
    )

    from ai_trading.data import market_calendar
    from ai_trading.utils import base as base_utils

    current_session_date = session_start.date()
    previous_session_date = prev_session_start.date()

    def fake_rth_session_utc(day):
        if day == current_session_date:
            return session_start, session_start + session_duration
        if day == previous_session_date:
            return prev_session_start, prev_session_start + session_duration
        raise AssertionError(f"unexpected date {day}")

    def fake_previous_trading_session(day):
        if day == current_session_date:
            return previous_session_date
        return previous_session_date - timedelta(days=1)

    monkeypatch.setattr(bot_engine, "CFG", config, raising=False)
    monkeypatch.setattr(bot_engine, "S", config, raising=False)
    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(staleness, "_ensure_data_fresh", fake_ensure)
    monkeypatch.setattr(base_utils, "is_market_open", lambda: True)
    monkeypatch.setattr(market_calendar, "rth_session_utc", fake_rth_session_utc)
    monkeypatch.setattr(
        market_calendar, "previous_trading_session", fake_previous_trading_session
    )

    result = bot_engine.fetch_minute_df_safe("AAPL")

    assert isinstance(result, pd.DataFrame)
    assert calls, "get_minute_df should be invoked"
    start, end = calls[0]
    assert start == prev_session_start
    assert end == base_now
    assert (end - start) >= timedelta(minutes=200)


def test_data_fetcher_stale_iex_retries_realtime_feed(monkeypatch):
    pd = load_pandas()

    base_now = datetime(2024, 1, 4, 15, 30, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    stale_idx = pd.date_range(end=base_now - timedelta(minutes=15), periods=5, freq="min", tz="UTC")
    fresh_idx = pd.date_range(end=base_now - timedelta(minutes=1), periods=5, freq="min", tz="UTC")

    settings = types.SimpleNamespace(
        alpaca_api_key="key",
        alpaca_secret_key_plain="secret",
        alpaca_data_feed="iex",
        alpaca_feed_failover=("sip",),
    )

    class DummyClient:  # noqa: D401 - minimal stub
        def __init__(self, *args, **kwargs):
            pass

    feeds: list[str] = []

    def fake_safe_get_stock_bars(client, req, symbol, tag):
        feeds.append(req.feed)
        if req.feed == "iex":
            idx = stale_idx
        else:
            idx = fresh_idx
        frame = pd.DataFrame(
            {
                "open": [1.0 + i * 0.01 for i in range(len(idx))],
                "high": [1.1 + i * 0.01 for i in range(len(idx))],
                "low": [0.9 + i * 0.01 for i in range(len(idx))],
                "close": [1.0 + i * 0.01 for i in range(len(idx))],
                "volume": [100] * len(idx),
                "symbol": [symbol] * len(idx),
            },
            index=idx,
        )
        return frame

    captured: dict[str, object] = {}
    freshness_calls: list[dict[str, object]] = []

    def capture_ensure(df, max_age_seconds, *, symbol=None, now=None, tz=None):
        captured["df"] = df.copy()
        captured["max_age"] = max_age_seconds
        captured["symbol"] = symbol
        return None

    def capture_freshness(
        age_seconds,
        *,
        market_open_now,
        phase,
        symbol,
        retry_feed,
        frame,
    ):
        freshness_calls.append(
            {
                "age_seconds": age_seconds,
                "market_open_now": market_open_now,
                "phase": phase,
                "symbol": symbol,
                "retry_feed": retry_feed,
            }
        )
        return True

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(bot_engine, "get_settings", lambda: settings)
    monkeypatch.setattr(bot_engine, "StockHistoricalDataClient", DummyClient)
    monkeypatch.setattr(
        bot_engine.bars,
        "safe_get_stock_bars",
        fake_safe_get_stock_bars,
    )
    monkeypatch.setattr(staleness, "_ensure_data_fresh", capture_ensure)
    monkeypatch.setattr(bot_engine, "_maybe_check_minute_freshness", capture_freshness)
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True, raising=False)
    monkeypatch.setattr(
        bot_engine,
        "_minute_data_freshness_limit",
        lambda: 900,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "minute_cache_hit", None, raising=False)
    monkeypatch.setattr(bot_engine, "minute_cache_miss", None, raising=False)

    fetcher = bot_engine.DataFetcher()
    ctx = types.SimpleNamespace()

    result = fetcher.get_minute_df(ctx, "AAPL", lookback_minutes=5)

    assert feeds == ["iex", "sip"]
    assert "df" in captured
    assert captured["symbol"] == "AAPL"
    assert captured["max_age"] == 900
    assert isinstance(result, pd.DataFrame)
    expected_idx = fresh_idx.rename("timestamp")
    pd.testing.assert_index_equal(result.index, expected_idx)
    assert fresh_idx.name is None
    assert fetcher._minute_cache["AAPL"].index[-1] == fresh_idx[-1]
    assert fetcher._minute_timestamps["AAPL"] == base_now
    assert freshness_calls, "freshness gate was not invoked"
    first_call = freshness_calls[0]
    assert first_call["market_open_now"] is True
    assert first_call["phase"] == "runtime"
    assert first_call["retry_feed"] == "sip"


def test_get_minute_df_handles_missing_safe_get(monkeypatch):
    pd = load_pandas()

    base_now = datetime(2024, 1, 2, 15, 31, tzinfo=UTC)
    idx = pd.date_range(end=base_now - timedelta(minutes=1), periods=3, freq="min", tz="UTC")

    settings = types.SimpleNamespace(
        alpaca_api_key="key",
        alpaca_secret_key_plain="secret",
        data_feed="iex",
        alpaca_data_feed=None,
        alpaca_feed_failover=("sip",),
        alpaca_adjustment=None,
    )

    class DummyClient:  # noqa: D401 - minimal stub
        def __init__(self, *args, **kwargs):
            pass

    calls: list[str] = []

    def fake_get_stock_bars(client, req, symbol, context=None):
        calls.append(context or "")
        legacy_payload = pd.DataFrame(
            {
                "timestamp": list(idx),
                "open": [1.0, 1.01, 1.02],
                "high": [1.1, 1.11, 1.12],
                "low": [0.9, 0.91, 0.92],
                "close": [1.0, 1.01, 1.02],
                "volume": [100, 110, 120],
                "symbol": [symbol] * len(idx),
            }
        )
        return legacy_payload.reset_index(drop=True)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    bars_stub = types.SimpleNamespace(
        TimeFrame=types.SimpleNamespace(Minute="Minute", Day="Day"),
        StockBarsRequest=lambda **kwargs: types.SimpleNamespace(**kwargs),
        _create_empty_bars_dataframe=lambda _tf: pd.DataFrame(),
        get_stock_bars=fake_get_stock_bars,
    )

    monkeypatch.setattr(bot_engine, "bars", bars_stub, raising=False)
    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)
    monkeypatch.setattr(bot_engine, "get_settings", lambda: settings)
    monkeypatch.setattr(bot_engine, "StockHistoricalDataClient", DummyClient)
    monkeypatch.setattr(staleness, "_ensure_data_fresh", lambda *a, **k: None)
    monkeypatch.setattr(
        bot_engine,
        "_minute_data_freshness_limit",
        lambda: 900,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "minute_cache_hit", None, raising=False)
    monkeypatch.setattr(bot_engine, "minute_cache_miss", None, raising=False)

    fetcher = bot_engine.DataFetcher()
    ctx = types.SimpleNamespace()

    result = fetcher.get_minute_df(ctx, "AAPL", lookback_minutes=2)

    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.index, pd.DatetimeIndex)
    expected_idx = idx.rename("timestamp")
    pd.testing.assert_index_equal(result.index, expected_idx)
    assert result.index.name == "timestamp"
    assert idx.name is None
    assert result.index.tz is not None
    assert result.index.tz.tzname(None) == UTC.tzname(None)
    assert list(result.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert calls and calls[0] == "MINUTE"
    assert not hasattr(bot_engine.bars, "safe_get_stock_bars")


def test_process_symbol_reuses_prefetched_minute_data(monkeypatch):
    pd = load_pandas()
    sample = _sample_df()

    fetch_calls: list[str] = []

    def fake_fetch(symbol: str):
        fetch_calls.append(symbol)
        return sample

    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", fake_fetch)

    observed: list = []
    fallback_calls: list[str] = []

    def fake_fetch_feature_data(ctx, state, symbol, price_df=None):
        observed.append(price_df)
        if price_df is None:
            fallback_calls.append(symbol)
            local_df = bot_engine.fetch_minute_df_safe(symbol)
        else:
            local_df = price_df
        return local_df, local_df, False

    monkeypatch.setattr(bot_engine, "_fetch_feature_data", fake_fetch_feature_data)

    def fake_trade_logic(
        ctx,
        state,
        symbol,
        balance,
        model,
        regime_ok,
        *,
        price_df=None,
        now_provider=None,
    ):
        bot_engine._fetch_feature_data(ctx, state, symbol, price_df=price_df)
        return True

    monkeypatch.setattr(bot_engine, "trade_logic", fake_trade_logic)

    state = bot_engine.BotState()
    state.position_cache = {}
    bot_engine.state = state

    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "ensure_final_bar", lambda symbol, timeframe: True)
    monkeypatch.setattr(bot_engine, "log_skip_cooldown", lambda *a, **k: None)
    monkeypatch.setattr(
        bot_engine,
        "skipped_duplicates",
        types.SimpleNamespace(inc=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "skipped_cooldown",
        types.SimpleNamespace(inc=lambda: None),
        raising=False,
    )

    ctx = types.SimpleNamespace(
        halt_manager=None,
        data_fetcher=types.SimpleNamespace(get_daily_df=lambda *_: sample),
        api=types.SimpleNamespace(list_positions=lambda: []),
    )
    monkeypatch.setattr(bot_engine, "get_ctx", lambda: ctx)

    prediction_executor = types.SimpleNamespace(
        submit=lambda fn, sym: types.SimpleNamespace(result=lambda: fn(sym))
    )
    monkeypatch.setattr(bot_engine, "prediction_executor", prediction_executor, raising=False)
    monkeypatch.setattr(bot_engine.executors, "_ensure_executors", lambda: None)

    processed, _ = bot_engine._process_symbols(["AAPL"], 1000.0, None, True)

    assert processed == ["AAPL"]
    assert fetch_calls == ["AAPL"]
    assert fallback_calls == []
    assert len(observed) == 1
    assert observed[0] is not None
    pd.testing.assert_frame_equal(observed[0], sample)


def test_fetch_minute_df_safe_market_cache_hit(monkeypatch, tmp_path):
    pd = load_pandas()

    calls: list[tuple[str, object, object]] = []

    base_now = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    session_start = base_now - timedelta(minutes=1)
    idx = pd.date_range(end=base_now - timedelta(minutes=1), periods=1, freq="min", tz="UTC")
    sample = pd.DataFrame({"close": [1.0], "volume": [100]}, index=idx)

    def fake_get_minute_df(symbol: str, start, end, **_):
        calls.append((symbol, start, end))
        return sample

    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    monkeypatch.setattr(bot_engine, "datetime", FrozenDatetime, raising=False)

    from ai_trading.data import market_calendar

    monkeypatch.setattr(
        market_calendar,
        "rth_session_utc",
        lambda *_: (session_start, base_now),
    )
    monkeypatch.setattr(
        market_calendar,
        "previous_trading_session",
        lambda current_date: current_date,
    )

    cache_cfg = types.SimpleNamespace(
        market_cache_enabled=True,
        market_cache_ttl=900,
        market_cache_disk=False,
        market_cache_disk_enabled=False,
        market_cache_dir=str(tmp_path / "market"),
        intraday_lookback_minutes=5,
    )
    monkeypatch.setattr(bot_engine, "CFG", cache_cfg, raising=False)
    monkeypatch.setattr(bot_engine, "S", cache_cfg, raising=False)

    from ai_trading.market import cache as market_cache

    cached_df: pd.DataFrame | None = None
    load_keys: list[str] = []
    loader_calls = 0

    def fake_get_or_load(key, loader, ttl):
        nonlocal cached_df, loader_calls
        load_keys.append(key)
        if cached_df is None:
            loader_calls += 1
            cached_df = loader()
        return cached_df

    monkeypatch.setattr(market_cache, "get_or_load", fake_get_or_load)

    with market_cache._lock:
        market_cache._mem.clear()

    first = bot_engine.fetch_minute_df_safe("AAPL")
    second = bot_engine.fetch_minute_df_safe("AAPL")

    assert loader_calls == 1
    assert isinstance(first, pd.DataFrame)
    assert isinstance(second, pd.DataFrame)
    pd.testing.assert_frame_equal(first, sample)
    pd.testing.assert_frame_equal(second, sample)


def test_signal_manager_evaluate_with_shorter_history(monkeypatch):
    pd = load_pandas()

    idx = pd.date_range(
        start=pd.Timestamp("2024-01-02 09:30", tz="UTC"), periods=120, freq="min"
    )
    closes = [100 + i * 0.1 for i in range(len(idx))]
    df = pd.DataFrame(
        {
            "open": [price - 0.05 for price in closes],
            "high": [price + 0.1 for price in closes],
            "low": [price - 0.1 for price in closes],
            "close": closes,
            "volume": [1000 + i for i in range(len(idx))],
            "stochrsi": [0.5] * len(idx),
            "rsi": [55.0] * len(idx),
            "rsi_14": [55.0] * len(idx),
            "ichimoku_conv": [1.0] * len(idx),
            "ichimoku_base": [1.0] * len(idx),
            "macd": [0.0] * len(idx),
            "macds": [0.0] * len(idx),
            "vwap": closes,
            "atr": [1.0] * len(idx),
        },
        index=idx,
    )

    def _stub_sentiment(self, ctx, ticker, df=None, model=None):  # noqa: D401
        return 1, 0.05, "sentiment"

    def _stub_regime(self, ctx, state, df, model=None):  # noqa: D401
        return 1, 0.05, "regime"

    monkeypatch.setattr(bot_engine.SignalManager, "signal_sentiment", _stub_sentiment)
    monkeypatch.setattr(bot_engine.SignalManager, "signal_regime", _stub_regime)
    monkeypatch.setattr(
        bot_engine.SignalManager,
        "signal_ml",
        lambda self, df, model=None, symbol=None: (1, 0.05, "ml"),
    )
    monkeypatch.setattr(bot_engine, "load_global_signal_performance", lambda: {})
    monkeypatch.setattr(bot_engine, "signals_evaluated", None, raising=False)

    manager = bot_engine.SignalManager()
    state = bot_engine.BotState()

    signal, confidence, label = manager.evaluate(
        object(), state, df.copy(), "AAPL", model=None
    )

    assert label != "no_data"
    assert manager.last_components
    assert confidence >= 0
    assert signal in (-1, 1)


def test_fetch_feature_data_skips_when_minute_stale(monkeypatch):
    err = bot_engine.DataFetchError("stale_minute_data")
    setattr(err, "fetch_reason", "stale_minute_data")
    setattr(err, "detail", "age=900s")

    def _raise_fetch(symbol: str):
        raise err

    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", _raise_fetch)

    halt_calls: list[str] = []

    class HaltManager:
        def manual_halt_trading(self, reason: str) -> None:  # noqa: D401
            halt_calls.append(reason)

    ctx = types.SimpleNamespace(
        halt_manager=HaltManager(),
        data_fetcher=types.SimpleNamespace(get_daily_df=lambda *_: _sample_df()),
    )

    raw_df, feat_df, skip_flag = bot_engine._fetch_feature_data(ctx, None, "AAPL")

    assert raw_df is None
    assert feat_df is None
    assert skip_flag is True
    assert halt_calls == []
