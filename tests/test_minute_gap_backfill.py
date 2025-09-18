import warnings
from datetime import UTC, datetime, timedelta

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core import bot_engine
from ai_trading.data import market_calendar
from ai_trading.data.fetch import _verify_minute_continuity
from ai_trading.guards import staleness
from ai_trading.utils import base as base_utils

def _make_gap_df():
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [ts[0], ts[2]],
            "open": [1.0, 1.2],
            "high": [1.1, 1.3],
            "low": [0.9, 1.1],
            "close": [1.05, 1.25],
            "volume": [100, 150],
        }
    )

def test_log_on_gaps(caplog):
    df = _make_gap_df()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with caplog.at_level("WARNING", logger="ai_trading.data.fetch"):
            out = _verify_minute_continuity(df, "TEST")
    assert not w
    assert out.equals(df)
    record = next(
        (rec for rec in caplog.records if rec.message == "MINUTE_GAPS_DETECTED"),
        None,
    )
    assert record is not None
    assert getattr(record, "symbol", None) == "TEST"
    assert getattr(record, "gap_count", None) == 1

def test_backfill_ffill():
    df = _make_gap_df()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        out = _verify_minute_continuity(df, "TEST", backfill="ffill")
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    assert len(out) == 3
    mid = out.loc[out["timestamp"] == ts[1]].iloc[0]
    assert mid["close"] == df["close"].iloc[0]
    assert mid["volume"] == 0

def test_backfill_interpolate():
    df = _make_gap_df()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        out = _verify_minute_continuity(df, "TEST", backfill="interpolate")
    ts = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    mid = out.loc[out["timestamp"] == ts[1]].iloc[0]
    assert mid["close"] == pytest.approx((df["close"].iloc[0] + df["close"].iloc[1]) / 2)


def test_fetch_minute_df_safe_gap_fill(monkeypatch, caplog):
    base_now = datetime(2024, 1, 3, 18, 30, tzinfo=UTC)
    session_start = base_now - timedelta(minutes=5)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return base_now.replace(tzinfo=None)
            return base_now.astimezone(tz)

    calls: list[dict[str, object]] = []

    def _make_df(ts_values: list[pd.Timestamp]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": ts_values,
                "open": [1.0 + 0.01 * i for i, _ in enumerate(ts_values)],
                "high": [1.01 + 0.01 * i for i, _ in enumerate(ts_values)],
                "low": [0.99 + 0.01 * i for i, _ in enumerate(ts_values)],
                "close": [1.0 + 0.01 * i for i, _ in enumerate(ts_values)],
                "volume": [100 + i for i, _ in enumerate(ts_values)],
            }
        )

    def _normalize_start(value: object) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    def fake_get_minute_df(symbol, start, end, feed=None, backfill=None, **_):
        calls.append({
            "symbol": symbol,
            "start": start,
            "end": end,
            "feed": feed,
            "backfill": backfill,
        })
        start_ts = _normalize_start(start)
        if feed in (None, "iex"):
            return _make_df([start_ts])
        if backfill:
            full_index = pd.date_range(start_ts, periods=5, freq="1min", tz="UTC")
            return _make_df(list(full_index))
        sparse_index = pd.date_range(start_ts, periods=5, freq="2min", tz="UTC")
        return _make_df(list(sparse_index))

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
    monkeypatch.setattr(bot_engine.CFG, "intraday_lookback_minutes", 5, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "alpaca_feed_failover", (), raising=False)
    monkeypatch.setattr(bot_engine.CFG, "longest_intraday_indicator_minutes", 5, raising=False)
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
    monkeypatch.delattr(bot_engine.CFG, "minute_gap_backfill", raising=False)

    with caplog.at_level("WARNING"):
        result = bot_engine.fetch_minute_df_safe("AAPL")

    assert len(result) == 5
    assert len(calls) >= 2
    assert calls[0]["backfill"] is None
    assert calls[1]["feed"] == "sip"
    assert calls[1]["backfill"] == "ffill"
    diffs = result["timestamp"].diff().dropna().dt.total_seconds()
    assert (diffs == 60).all()
    assert all(rec.message != "MINUTE_GAPS_DETECTED" for rec in caplog.records)
