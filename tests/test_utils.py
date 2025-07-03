import os
import socket
import sys
import types
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

import utils

# Basic utils behaviour

def test_warn_limited(caplog):
    caplog.set_level("WARNING")
    utils._WARN_COUNTS.clear()
    for i in range(4):
        utils._warn_limited("k", "warn %d", i, limit=2)
    msgs = [rec.message for rec in caplog.records]
    assert "warn 0" in msgs[0]
    assert "warn 1" in msgs[1]
    assert "suppressed" in msgs[2]
    assert len(msgs) == 3


def test_to_serializable_mappingproxy():
    from types import MappingProxyType

    mp = MappingProxyType({"a": 1})
    res = utils.to_serializable({"x": mp, "l": [1, 2]})
    assert res == {"x": {"a": 1}, "l": [1, 2]}


def test_get_free_port():
    port = utils.get_free_port(start=9260, end=9260)
    assert port == 9260


def test_ensure_utc_variants():
    naive = datetime(2024, 1, 1, 12, 0)
    assert utils.ensure_utc(naive).tzinfo == timezone.utc
    aware = datetime(2024, 1, 1, 12, 0, tzinfo=utils.EASTERN_TZ)
    assert utils.ensure_utc(aware).tzinfo == timezone.utc
    d = date(2024, 1, 1)
    assert utils.ensure_utc(d).tzinfo == timezone.utc


def test_is_market_open_with_calendar(monkeypatch):
    mod = types.ModuleType("pandas_market_calendars")

    class Cal:
        def schedule(self, start_date=None, end_date=None):
            return pd.DataFrame({
                "market_open": [pd.Timestamp("2024-01-02 09:30", tz="US/Eastern")],
                "market_close": [pd.Timestamp("2024-01-02 16:00", tz="US/Eastern")],
            })

    mod.get_calendar = lambda name: Cal()
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mod)
    now = datetime(2024, 1, 2, 10, 0, tzinfo=utils.EASTERN_TZ)
    assert utils.is_market_open(now)


def test_is_market_open_fallback(monkeypatch):
    mod = types.ModuleType("pandas_market_calendars")
    mod.get_calendar = lambda name: (_ for _ in ()).throw(Exception("fail"))
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mod)
    weekend = datetime(2024, 1, 6, 10, 0, tzinfo=utils.EASTERN_TZ)
    assert not utils.is_market_open(weekend)


def test_safe_to_datetime_various(caplog):
    caplog.set_level("WARNING")
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    assert utils.safe_to_datetime(("SYM", ts)) == ts
    arr = [("A", ts), ("B", ts)]
    idx = utils.safe_to_datetime(arr)
    assert list(idx) == [ts, ts]
    res = utils.safe_to_datetime(["bad"])
    assert res.isna().all()
    assert "coercing" in caplog.text


def test_get_latest_close_cases():
    assert utils.get_latest_close(None) == 0.0
    df = pd.DataFrame()
    assert utils.get_latest_close(df) == 0.0
    df = pd.DataFrame({"close": [0, np.nan]})
    assert utils.get_latest_close(df) == 0.0
    df = pd.DataFrame({"close": [1.0, 2.0]})
    assert utils.get_latest_close(df) == 2.0


def test_health_check_paths(monkeypatch):
    assert not utils.health_check(None, "m")
    df = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setenv("HEALTH_MIN_ROWS", "5")
    assert not utils.health_check(df, "m")
    df = pd.DataFrame({"a": range(10)})
    assert utils.health_check(df, "m")


def test_safe_get_column_warning(caplog):
    caplog.set_level("WARNING")
    df = pd.DataFrame({"other": [1]})
    assert utils.get_open_column(df) is None
    assert "open price" in caplog.text


def test_log_warning(caplog):
    caplog.set_level("WARNING")
    utils.log_warning("msg", exc=ValueError("boom"), extra={"a": 1})
    assert "boom" in caplog.text


def test_get_column_validation_errors():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        utils.get_column(df, ["b"], "b")
    df = pd.DataFrame({"c": [1, 2]})
    with pytest.raises(TypeError):
        utils.get_column(df, ["c"], "c", dtype="datetime64[ns]")


def test_safe_get_column_and_ohlcv():
    df = pd.DataFrame({"Open": [1], "High": [2], "Low": [0], "Close": [1], "Volume": [10]})
    assert utils.get_open_column(df) == "Open"
    assert utils.get_ohlcv_columns(df) == ["Open", "High", "Low", "Close", "Volume"]


def test_pre_trade_health_check(monkeypatch):
    called = []
    monkeypatch.setattr(utils, "check_symbol", lambda s, a: called.append(s) or True)
    res = utils.pre_trade_health_check(["A", "B"], api=None)
    assert res == {"A": True, "B": True}
    assert called == ["A", "B"]


def test_column_helpers(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame({
        "Datetime": dates,
        "symbol": ["A", "B", "C"],
        "Return": [0.1, 0.2, 0.3],
        "indicator": [1, 2, 3],
        "OrderID": [1, 2, 3],
    })
    assert utils.get_datetime_column(df) == "Datetime"
    assert utils.get_symbol_column(df) == "symbol"
    assert utils.get_return_column(df) == "Return"
    assert utils.get_indicator_column(df, ["indicator"]) == "indicator"
    assert utils.get_order_column(df, "OrderID") == "OrderID"


def test_log_warning_no_exc(caplog):
    caplog.set_level("WARNING")
    utils.log_warning("plain")
    assert "plain" in caplog.text


def test_callable_lock_methods():
    lock = utils._CallableLock()
    assert not lock.locked()
    lock.acquire()
    assert lock.locked()
    lock.release()
    with lock():
        assert lock.locked()
    assert not lock.locked()


def test_get_latest_close_no_column():
    df = pd.DataFrame({"open": [1]})
    assert utils.get_latest_close(df) == 0.0


def test_is_market_open_holiday(monkeypatch):
    mod = types.ModuleType("pandas_market_calendars")

    class Cal:
        def schedule(self, start_date=None, end_date=None):
            return pd.DataFrame()

    mod.get_calendar = lambda name: Cal()
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mod)
    now = datetime(2024, 1, 1, 10, 0, tzinfo=utils.EASTERN_TZ)
    assert not utils.is_market_open(now)


def test_is_market_open_july3_2025(monkeypatch):
    """July 3, 2025 should be normal hours."""
    mod = types.ModuleType("pandas_market_calendars")

    class Cal:
        def schedule(self, start_date=None, end_date=None):
            return pd.DataFrame(
                {
                    "market_open": [pd.Timestamp("2025-07-03 09:30", tz="US/Eastern")],
                    "market_close": [pd.Timestamp("2025-07-03 13:00", tz="US/Eastern")],
                }
            )

    mod.get_calendar = lambda name: Cal()
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mod)
    now = datetime(2025, 7, 3, 11, 55, tzinfo=utils.EASTERN_TZ)
    assert utils.is_market_open(now)


def test_ensure_utc_type_error():
    with pytest.raises(AssertionError):
        utils.ensure_utc(1)


def test_get_free_port_none(monkeypatch):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    monkeypatch.setattr(socket, "socket", lambda *a, **k: socket.socket(*a, **k))
    try:
        res = utils.get_free_port(start=port, end=port - 1)
        assert res is None
    finally:
        sock.close()


def test_safe_to_datetime_error(monkeypatch, caplog):
    caplog.set_level("ERROR")

    class Bad:
        pass

    monkeypatch.setattr(pd, "to_datetime", lambda *a, **k: (_ for _ in ()).throw(TypeError("bad")))
    res = utils.safe_to_datetime([Bad()])
    assert res.isna().all()
    assert "failed" in caplog.text


def test_health_check_empty(caplog, monkeypatch):
    caplog.set_level("CRITICAL")
    monkeypatch.setenv("HEALTH_MIN_ROWS", "1")
    df = pd.DataFrame()
    assert not utils.health_check(df, "d")
    assert "empty dataset" in caplog.text


def test_get_column_errors():
    dates = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"])
    df = pd.DataFrame({"d": dates})
    with pytest.raises(ValueError):
        utils.get_column(df, ["d"], "lbl", must_be_monotonic=True)
    df = pd.DataFrame({"d": [pd.NaT, pd.NaT, pd.NaT]})
    with pytest.raises(ValueError):
        utils.get_column(df, ["d"], "lbl", must_be_non_null=True)
    df = pd.DataFrame({"d": [1, 1, 2]})
    with pytest.raises(ValueError):
        utils.get_column(df, ["d"], "lbl", must_be_unique=True)
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=3, tz="UTC")})
    df["d"] = df["d"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        utils.get_column(df, ["d"], "lbl", must_be_timezone_aware=True)


def test_safe_get_column_non_df():
    assert utils._safe_get_column([], ["x"], "lbl") is None


def test_get_ohlcv_columns_missing():
    df = pd.DataFrame({"Open": [1], "High": [2]})
    assert utils.get_ohlcv_columns(df) == []


def test_check_symbol_failure(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda path: (_ for _ in ()).throw(IOError("bad")))
    assert not utils.check_symbol("A", api=None)


def test_is_market_open_attr_error(monkeypatch):
    mod = types.ModuleType("pandas_market_calendars")
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mod)
    now = datetime(2024, 1, 6, 10, 0, tzinfo=utils.EASTERN_TZ)
    assert not utils.is_market_open(now)


def test_check_symbol_success(monkeypatch, tmp_path):
    file = tmp_path / "A.csv"
    pd.DataFrame({"close": [1, 2, 3]}).to_csv(file, index=False)
    monkeypatch.setattr(os.path, "join", lambda *a: str(file))
    monkeypatch.setenv("HEALTH_MIN_ROWS", "1")
    assert utils.check_symbol("A", api=None)
