import sys
from pathlib import Path
import types
import pandas as pd
from datetime import datetime, date, time, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils


def test_get_latest_close_normal():
    df = pd.DataFrame({'close':[1.0, 2.0]})
    assert utils.get_latest_close(df) == 2.0


def test_get_latest_close_missing():
    df = pd.DataFrame({'open':[1.0]})
    assert utils.get_latest_close(df) == 1.0


def test_is_market_open_with_calendar(monkeypatch):
    mod = types.ModuleType('pandas_market_calendars')
    class DummyCal:
        def schedule(self, start_date=None, end_date=None):
            return pd.DataFrame({
                'market_open':[pd.Timestamp('2024-01-02 09:30', tz='US/Eastern')],
                'market_close':[pd.Timestamp('2024-01-02 16:00', tz='US/Eastern')]
            })
    mod.get_calendar = lambda name: DummyCal()
    monkeypatch.setitem(sys.modules, 'pandas_market_calendars', mod)
    now = datetime(2024, 1, 2, 10, 0, tzinfo=utils.EASTERN_TZ)
    assert utils.is_market_open(now)


def test_is_market_open_fallback(monkeypatch):
    mod = types.ModuleType('pandas_market_calendars')
    mod.get_calendar = lambda name: (_ for _ in ()).throw(Exception('fail'))
    monkeypatch.setitem(sys.modules, 'pandas_market_calendars', mod)
    weekend = datetime(2024, 1, 6, 10, 0, tzinfo=utils.EASTERN_TZ)
    assert not utils.is_market_open(weekend)


def test_ensure_utc_and_convert():
    naive = datetime(2024,1,1,12,0)
    aware = utils.ensure_utc(naive)
    assert aware.tzinfo == timezone.utc
    d = date(2024,1,1)
    assert utils.ensure_utc(d).tzinfo == timezone.utc


def test_safe_to_datetime():
    vals = ['2024-01-01','2024-01-02']
    idx = utils.safe_to_datetime(vals)
    assert list(idx) == [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')]
    assert utils.safe_to_datetime(['abc']) is None
