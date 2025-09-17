import sys
import types
from datetime import UTC, datetime, timedelta

import pytest
from ai_trading.utils.lazy_imports import load_pandas

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


def _sample_df():
    pd = load_pandas()
    return pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp("2024-01-01", tz="UTC")])


def test_fetch_minute_df_safe_returns_dataframe(monkeypatch):
    pd = load_pandas()
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda s, start, end: _sample_df())
    monkeypatch.setattr(staleness, "_ensure_data_fresh", lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None)
    result = bot_engine.fetch_minute_df_safe("AAPL")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_fetch_minute_df_safe_raises_on_empty(monkeypatch):
    pd = load_pandas()
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda s, start, end: pd.DataFrame())
    monkeypatch.setattr(staleness, "_ensure_data_fresh", lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None)
    with pytest.raises(bot_engine.DataFetchError):
        bot_engine.fetch_minute_df_safe("AAPL")


def test_fetch_minute_df_safe_raises_on_stale(monkeypatch):
    pd = load_pandas()
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda s, start, end: _sample_df())

    def _raise_stale(df, max_age_seconds, *, symbol=None, now=None, tz=None):
        raise RuntimeError("age=900s")

    monkeypatch.setattr(staleness, "_ensure_data_fresh", _raise_stale)

    with pytest.raises(bot_engine.DataFetchError) as excinfo:
        bot_engine.fetch_minute_df_safe("AAPL")

    assert getattr(excinfo.value, "fetch_reason", None) == "stale_minute_data"
    assert getattr(excinfo.value, "symbol", None) == "AAPL"


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

    idx = pd.date_range(end=session_end, periods=3, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.0, 1.1, 1.2],
            "volume": [100, 110, 120],
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
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda s, start, end: df.copy())
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

    result = bot_engine.fetch_minute_df_safe("AMGN")

    assert captured["max_age"] == 600
    assert captured["now"] == session_end
    pd.testing.assert_frame_equal(result, df)


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
    sample = _sample_df()

    calls: list[tuple[str, object, object]] = []

    def fake_get_minute_df(symbol: str, start, end):
        calls.append((symbol, start, end))
        return sample

    monkeypatch.setattr(bot_engine, "get_minute_df", fake_get_minute_df)
    monkeypatch.setattr(
        staleness,
        "_ensure_data_fresh",
        lambda df, max_age_seconds, *, symbol=None, now=None, tz=None: None,
    )
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)

    base_now = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    session_start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)

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
    )
    monkeypatch.setattr(bot_engine, "CFG", cache_cfg, raising=False)
    monkeypatch.setattr(bot_engine, "S", cache_cfg, raising=False)

    from ai_trading.market import cache as market_cache

    with market_cache._lock:
        market_cache._mem.clear()

    first = bot_engine.fetch_minute_df_safe("AAPL")
    second = bot_engine.fetch_minute_df_safe("AAPL")

    assert len(calls) == 1
    assert isinstance(first, pd.DataFrame)
    assert isinstance(second, pd.DataFrame)
    pd.testing.assert_frame_equal(first, sample)
    pd.testing.assert_frame_equal(second, sample)


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

