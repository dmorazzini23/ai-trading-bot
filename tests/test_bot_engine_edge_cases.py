import pytest
pd = pytest.importorskip("pandas")

from tests.vendor_stubs import alpaca as _alpaca  # noqa: F401

# AI-AGENT-REF: Replaced unsafe _raise_dynamic_exec_disabled() with direct import from shim module
from ai_trading.core.bot_engine import prepare_indicators


def test_prepare_indicators_missing_close_column():
    df = pd.DataFrame({'open': [1, 2], 'high': [1, 2], 'low': [1, 2]})
    with pytest.raises(KeyError):
        prepare_indicators(df)


def test_prepare_indicators_non_numeric_close(monkeypatch):
    from ai_trading.core import bot_engine

    def fake_rsi(close, length=14):
        if not pd.api.types.is_numeric_dtype(close):
            raise TypeError('close column must be numeric')
        return pd.Series(range(len(close)), dtype=float)

    monkeypatch.setattr(bot_engine.ta, 'rsi', fake_rsi)
    df = pd.DataFrame({'open': [1, 2], 'high': [1, 2], 'low': [1, 2], 'close': ['a', 'b']})
    with pytest.raises(ValueError):
        prepare_indicators(df)


def test_prepare_indicators_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(RuntimeError):
        prepare_indicators(df)


def test_prepare_indicators_single_row():
    df = pd.DataFrame({
        'open': [100.0],
        'high': [101.0],
        'low': [99.0],
        'close': [100.5],
        'volume': [1000]
    })
    result = prepare_indicators(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_fetch_minute_df_nan_closes_triggers_guard(monkeypatch, caplog):
    from datetime import UTC, datetime, timedelta
    from ai_trading.core import bot_engine

    now_utc = datetime.now(UTC).replace(second=0, microsecond=0)
    index = pd.date_range(end=now_utc - timedelta(minutes=1), periods=3, freq="T", tz="UTC")
    nan_df = pd.DataFrame(
        {
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [float('nan')] * len(index),
            'volume': [1_000, 1_200, 1_500],
        },
        index=index,
    )

    monkeypatch.setattr("ai_trading.utils.base.is_market_open", lambda: True)
    session_start = now_utc - timedelta(hours=6)
    session_end = now_utc
    monkeypatch.setattr(
        "ai_trading.data.market_calendar.rth_session_utc",
        lambda _date: (session_start, session_end),
    )
    monkeypatch.setattr(
        bot_engine,
        "get_minute_df",
        lambda symbol, start, end: nan_df.copy(),
    )

    caplog.set_level("DEBUG", logger="ai_trading.core.bot_engine")

    class DummyHaltManager:
        def __init__(self):
            self.calls: list[str] = []

        def manual_halt_trading(self, reason: str) -> None:
            self.calls.append(reason)

    class DummyFetcher:
        def get_daily_df(self, ctx, symbol):  # pragma: no cover - guard path
            raise AssertionError("daily fallback should not run")

    halt_manager = DummyHaltManager()
    ctx = type("Ctx", (), {})()
    ctx.halt_manager = halt_manager
    ctx.data_fetcher = DummyFetcher()

    called = {"value": False}

    def marker(frame):
        called["value"] = True
        return frame

    monkeypatch.setattr(bot_engine, "prepare_indicators", marker)

    raw_df, feat_df, skip_flag = bot_engine._fetch_feature_data(ctx, None, "TEST")

    assert (raw_df, feat_df, skip_flag) == (None, None, False)
    assert halt_manager.calls == ["TEST:empty_frame"]
    assert not called["value"]
    assert any(
        rec.getMessage() == "FETCH_MINUTE_CLOSE_ALL_NAN_AFTER_FILTER"
        for rec in caplog.records
    )
    assert not any(
        "prepare_indicators produced empty dataframe" in rec.getMessage()
        for rec in caplog.records
    )
    assert not any("Error calculating EMA" in rec.getMessage() for rec in caplog.records)
