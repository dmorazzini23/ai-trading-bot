from __future__ import annotations

import datetime as dt
import types
from enum import Enum
from types import MappingProxyType
from uuid import UUID

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.utils import base


class _Side(Enum):
    BUY = "buy"


def test_ensure_utc_index_localizes_and_converts_datetime_index():
    naive = pd.DataFrame({"close": [1]}, index=pd.DatetimeIndex(["2026-01-01 09:30"]))
    localized = base.ensure_utc_index(naive)
    assert str(localized.index.tz) == "UTC"
    assert naive.index.tz is None

    eastern = pd.DataFrame(
        {"close": [2]},
        index=pd.DatetimeIndex(["2026-01-01 09:30"], tz="America/New_York"),
    )
    converted = base.ensure_utc_index(eastern)
    assert str(converted.index.tz) == "UTC"
    assert converted.index[0].hour == 14

    not_datetime = pd.DataFrame({"close": [3]}, index=["row"])
    assert base.ensure_utc_index(not_datetime) is not_datetime


def test_log_warning_routes_health_stale_to_debug(monkeypatch):
    calls: list[tuple[str, str, object | None, bool]] = []

    class Logger:
        def debug(self, msg, *args, extra=None, exc_info=False):
            calls.append(("debug", msg, extra, bool(exc_info)))

        def warning(self, msg, *args, extra=None, exc_info=False):
            calls.append(("warning", msg, extra, bool(exc_info)))

    monkeypatch.setattr(base, "logger", Logger())

    base.log_warning("HEALTH_STALE_DATA", exc=RuntimeError("stale"), extra={"symbol": "SPY"})
    base.log_warning("OTHER_EVENT", exc=RuntimeError("boom"), extra={"symbol": "QQQ"})
    base.log_warning("HEALTH_STALE_DATA", extra={"symbol": "IWM"})
    base.log_warning("OTHER_EVENT", extra={"symbol": "DIA"})

    assert calls == [
        ("debug", "%s: %s", {"symbol": "SPY"}, True),
        ("warning", "%s: %s", {"symbol": "QQQ"}, True),
        ("debug", "HEALTH_STALE_DATA", {"symbol": "IWM"}, False),
        ("warning", "OTHER_EVENT", {"symbol": "DIA"}, False),
    ]


def test_should_log_stale_caches_by_symbol_timestamp(monkeypatch):
    base._STALE_CACHE.clear()
    clock = {"now": 1000.0}
    monkeypatch.setattr("time.time", lambda: clock["now"])
    ts = pd.Timestamp("2026-01-01T00:00:00Z")

    assert base.should_log_stale("SPY", ts, ttl=60) is True
    assert base.should_log_stale("SPY", ts, ttl=60) is False

    clock["now"] = 1061.0
    assert base.should_log_stale("SPY", ts, ttl=60) is True
    assert base.should_log_stale("QQQ", ts, ttl=60) is True


def test_backoff_delay_caps_and_applies_jitter(monkeypatch):
    monkeypatch.setattr(base.random, "uniform", lambda low, high: high)

    assert base.backoff_delay(4, base=2.0, cap=10.0, jitter=0.0) == 10.0
    assert base.backoff_delay(2, base=2.0, cap=10.0, jitter=0.25) == 5.0


def test_format_order_for_log_serializes_common_runtime_values():
    order = types.SimpleNamespace(
        symbol="SPY",
        submitted_at=dt.datetime(2026, 1, 2, 3, 4, tzinfo=dt.UTC),
        side=_Side.BUY,
        order_id=UUID("12345678-1234-5678-1234-567812345678"),
        qty=10,
        active=True,
        note={"nested": "value"},
    )

    formatted = base.format_order_for_log(order)

    assert "symbol=SPY" in formatted
    assert "submitted_at=2026-01-02T03:04:00+00:00" in formatted
    assert "side=buy" in formatted
    assert "order_id=12345678-1234-5678-1234-567812345678" in formatted
    assert "active=True" in formatted
    assert "note={'nested': 'value'}" in formatted
    assert base.format_order_for_log(None) == ""


def test_callable_lock_supports_context_and_callable_usage():
    lock = base._CallableLock()

    with lock() as acquired:
        assert acquired is lock
        assert lock.locked() is True

    assert lock.locked() is False


def test_get_latest_close_handles_aliases_invalid_and_nonfinite():
    assert base.get_latest_close(None) == 0.0
    assert base.get_latest_close(pd.DataFrame({"Close": ["bad", "7.5"]})) == 7.5
    assert base.get_latest_close(pd.DataFrame({"c": [float("inf")]})) == 0.0
    assert base.get_latest_close(pd.DataFrame({"open": [1.0]})) == 0.0


def test_health_row_logging_and_throttled_passed(monkeypatch):
    events: list[tuple[str, str, tuple[object, ...]]] = []

    class Logger:
        def info(self, msg, *args):
            events.append(("info", msg, args))

        def debug(self, msg, *args):
            events.append(("debug", msg, args))

    monkeypatch.setattr(base, "logger", Logger())
    monkeypatch.setattr(base, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(base, "monotonic_time", lambda: 25.0)
    base._LAST_HEALTH_ROW_LOG = 0.0
    base._LAST_HEALTH_ROWS_COUNT = -1
    base._LAST_HEALTH_STATUS = None
    base._last_health_log = 0.0

    base.log_health_row_check(10, True)
    base.log_health_row_check(10, True)
    base.log_health_row_check(4, False)
    rows = [1, 2, 3]
    assert base.health_rows_passed(rows) is rows
    assert base.health_rows_passed(rows) is rows

    assert ("debug", "HEALTH_ROWS_%s: received %d rows", ("PASSED", 10)) in events
    assert ("info", "HEALTH_ROWS_%s: received %d rows", ("FAILED", 4)) in events
    assert ("debug", "HEALTH_ROWS_PASSED: received %d rows", (3,)) in events
    assert ("debug", "HEALTH_ROWS_THROTTLED", ()) in events


def test_weekend_holiday_and_ensure_utc_conversions():
    saturday = dt.datetime(2026, 1, 3, 12, tzinfo=dt.UTC)
    friday = dt.datetime(2026, 1, 2, 12, tzinfo=dt.UTC)

    assert base.is_weekend(saturday) is True
    assert base.is_weekend(friday) is False
    assert base.is_market_holiday(dt.date(2026, 1, 1)) is True
    assert base.is_market_holiday(dt.datetime(2026, 1, 2, tzinfo=dt.UTC)) is False
    assert base.ensure_utc(dt.datetime(2026, 1, 2, 9, 30)).tzinfo is dt.UTC
    assert base.ensure_utc(dt.date(2026, 1, 2)) == dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    with pytest.raises(TypeError):
        base.ensure_utc("2026-01-02")  # type: ignore[arg-type]


def test_to_serializable_recurses_mapping_proxy_and_tuples():
    nested = MappingProxyType({"items": ({"value": 1}, MappingProxyType({"inner": 2}))})

    assert base.to_serializable(nested) == {"items": [{"value": 1}, {"inner": 2}]}


def test_warn_limited_suppresses_after_limit(monkeypatch):
    warnings: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        base.logger,
        "warning",
        lambda msg, *args, **_kwargs: warnings.append((msg, args)),
    )
    base._WARN_COUNTS.clear()

    for _ in range(4):
        base._warn_limited("demo", "problem %s", "x", limit=2)

    assert warnings == [
        ("problem %s", ("x",)),
        ("problem %s", ("x",)),
        ("Further '%s' warnings suppressed", ("demo",)),
    ]


def test_safe_to_datetime_handles_empty_none_scalar_and_naive_output():
    assert len(base.safe_to_datetime(None)) == 0
    assert len(base.safe_to_datetime([], utc=False)) == 0

    scalar = base.safe_to_datetime("2026-01-02 09:30", utc=False)
    assert scalar.tz is None
    assert scalar[0] == pd.Timestamp("2026-01-02 09:30")


def test_validate_ohlcv_rejects_invalid_timestamps_and_order():
    valid = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "open": [1, 2],
            "high": [2, 3],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100, 200],
        }
    )
    base.validate_ohlcv(valid)

    with pytest.raises(ValueError, match="timestamp contains"):
        bad_ts = valid.copy()
        bad_ts.loc[1, "timestamp"] = "not-a-date"
        base.validate_ohlcv(bad_ts)

    with pytest.raises(ValueError, match="not monotonic"):
        reversed_ts = valid.iloc[::-1].reset_index(drop=True)
        base.validate_ohlcv(reversed_ts)

    with pytest.raises(ValueError, match="no rows"):
        base.validate_ohlcv(valid.iloc[0:0], require_monotonic=False)


def test_health_check_handles_length_errors(monkeypatch):
    monkeypatch.setattr(base, "get_env", lambda *_args, **_kwargs: "2")

    assert base.health_check([1, 2]) is True
    assert base.health_check([1]) is False
    assert base.health_check(object()) is False


def test_column_helpers_validate_dtype_shape_and_aliases():
    df = pd.DataFrame(
        {
            "Datetime": pd.date_range("2026-01-01", periods=2, tz="UTC"),
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [100, 200],
            "symbol": ["SPY", "QQQ"],
            "Return": [0.1, None],
        }
    )

    assert base.get_datetime_column(df) == "Datetime"
    assert base.get_ohlcv_columns(df) == ["Open", "High", "Low", "Close", "Volume"]
    assert base.get_indicator_column(df, ["missing", "Close"]) == "Close"
    assert base.get_order_column(df, "Return") == "Return"
    assert base.get_return_column(df) == "Return"
    assert base.get_symbol_column(df) == "symbol"

    with pytest.raises(TypeError, match="not of dtype"):
        base.get_column(df, ["Open"], "open", dtype="int64")
    with pytest.raises(ValueError, match="not unique"):
        dupes = df.assign(symbol=["SPY", "SPY"])
        base.get_column(dupes, ["symbol"], "symbol", must_be_unique=True)
    with pytest.raises(ValueError, match="all null"):
        nulls = df.assign(Return=[None, None])
        base.get_column(nulls, ["Return"], "return", must_be_non_null=True)
    with pytest.raises(ValueError, match="not timezone-aware"):
        naive = df.assign(Datetime=pd.date_range("2026-01-01", periods=2))
        base.get_column(naive, ["Datetime"], "datetime", must_be_timezone_aware=True)
    with pytest.raises(ValueError, match="No recognized"):
        base.get_column(df, ["missing"], "missing")


def test_safe_column_helpers_and_basic_ohlcv_validation():
    assert base.get_open_column(pd.DataFrame()) is None
    assert base.get_ohlcv_columns({"not": "a dataframe"}) == []
    assert base.validate_ohlcv_basic(pd.DataFrame()) is False
    assert base.validate_ohlcv_basic(pd.DataFrame({"Open": [1]})) is False
    assert (
        base.validate_ohlcv_basic(
            pd.DataFrame(
                {
                    "Open": [1],
                    "High": [2],
                    "Low": [0],
                    "Close": [1.5],
                    "Volume": [100],
                }
            )
        )
        is True
    )


def test_price_feature_helpers_use_minute_data(monkeypatch):
    idx = pd.date_range(pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=29), periods=30, freq="min")
    data = pd.DataFrame(
        {
            "high": list(range(30, 60)),
            "low": list(range(20, 50)),
            "close": list(range(25, 55)),
            "volume": [100] * 29 + [500],
        },
        index=idx,
    )
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.fetch_minute_df_safe",
        lambda _symbol: data,
    )

    assert base.get_rolling_atr("SPY", window=3) > 0
    assert base.get_current_vwap("SPY") > 0
    assert base.get_volume_spike_factor("SPY") == pytest.approx(5.0)

    monkeypatch.setattr("ai_trading.core.bot_engine.fetch_minute_df_safe", lambda _symbol: None)
    assert base.get_rolling_atr("SPY") == 0.0
    assert base.get_current_vwap("SPY") == 0.0
    assert base.get_volume_spike_factor("SPY") == 1.0


def test_price_feature_helpers_fail_closed_on_stale_minute_data(monkeypatch):
    idx = pd.date_range("2020-01-02 14:30:00+00:00", periods=30, freq="min")
    stale = pd.DataFrame(
        {
            "high": list(range(30, 60)),
            "low": list(range(20, 50)),
            "close": list(range(25, 55)),
            "volume": [100] * 29 + [500],
        },
        index=idx,
    )
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.fetch_minute_df_safe",
        lambda _symbol: stale,
    )

    assert base.get_rolling_atr("SPY", window=3) == 0.0
    assert base.get_current_vwap("SPY") == 0.0
    assert base.get_volume_spike_factor("SPY") == 1.0


def test_pre_trade_health_check_collects_symbol_results(monkeypatch):
    monkeypatch.setattr(base, "check_symbol", lambda symbol, _api: symbol == "SPY")

    result = base.pre_trade_health_check(["SPY", "BAD"], api=object())

    assert result == {"SPY": True, "BAD": False}
