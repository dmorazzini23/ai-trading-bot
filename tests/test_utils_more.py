import types
import sys
from types import MappingProxyType
from datetime import datetime, timezone
import pandas as pd
import socket
import pytest

import utils


def test_log_warning_with_exception(caplog):
    caplog.set_level('WARNING')
    utils.log_warning('hi', exc=ValueError('x'))
    assert 'hi: x' in caplog.text
    caplog.clear()
    utils.log_warning('hi')
    assert 'hi' in caplog.text


def test_callable_lock_methods():
    lock = utils._CallableLock()
    assert not lock.locked()
    lock.acquire()
    try:
        assert lock.locked()
    finally:
        lock.release()
    assert not lock.locked()


def test_is_market_open_weekday(monkeypatch):
    mod = types.ModuleType('pandas_market_calendars')
    mod.get_calendar = lambda name: (_ for _ in ()).throw(Exception('fail'))
    monkeypatch.setitem(sys.modules, 'pandas_market_calendars', mod)
    now = datetime(2024, 1, 2, 10, 0, tzinfo=utils.EASTERN_TZ)
    assert utils.is_market_open(now)


def test_ensure_utc_timezone_aware():
    aware = datetime(2024,1,1,12,0,tzinfo=timezone.utc)
    assert utils.ensure_utc(aware).tzinfo == timezone.utc
    with pytest.raises(AssertionError):
        utils.ensure_utc(123)


def test_get_free_port_none(monkeypatch):
    def fake_socket(*a, **k):
        class S:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def bind(self, *a): raise OSError
        return S()
    monkeypatch.setattr(socket, 'socket', fake_socket)
    assert utils.get_free_port(9200, 9200) is None


def test_to_serializable_mappingproxy():
    mp = MappingProxyType({'a': 1})
    assert utils.to_serializable(mp) == {'a': 1}
    assert utils.to_serializable([mp]) == [{'a': 1}]


def test_warn_limited(caplog):
    caplog.set_level('WARNING')
    for _ in range(4):
        utils._warn_limited('k', 'msg')
    assert "suppressed" in caplog.text


def test_safe_to_datetime_none_and_failure(monkeypatch):
    assert len(utils.safe_to_datetime(None)) == 0
    def bad(*a, **k):
        raise TypeError
    monkeypatch.setattr(utils.pd, 'to_datetime', bad)
    res = utils.safe_to_datetime(['x'])
    assert res.isna().all()


def test_get_ohlcv_columns_positive():
    df = pd.DataFrame({
        'Open':[1], 'High':[1], 'Low':[1], 'Close':[1], 'Volume':[1]
    })
    cols = utils.get_ohlcv_columns(df)
    assert cols == ['Open', 'High', 'Low', 'Close', 'Volume']
