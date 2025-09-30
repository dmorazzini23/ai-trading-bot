import pytest
pd = pytest.importorskip("pandas")

import sys
import types

sys.modules.setdefault("portalocker", types.ModuleType("portalocker"))
bs4_stub = types.ModuleType("bs4")
bs4_stub.BeautifulSoup = object
sys.modules.setdefault("bs4", bs4_stub)
flask_stub = types.ModuleType("flask")
class _Flask:
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):  # pragma: no cover - simple stub
        def _decorator(f):
            return f
        return _decorator

flask_stub.Flask = _Flask
sys.modules.setdefault("flask", flask_stub)
from ai_trading.core import bot_engine as be


def test_macd_insufficient_history_defers():
    df = pd.DataFrame({"close": range(10)})
    be._add_macd(df, "TST", None)
    assert df["macd"].isna().all()
    assert df["macds"].isna().all()


def test_signal_manager_skips_nan_indicators():
    sm = be.SignalManager()
    short_df = pd.DataFrame({"close": [1, 2, 3]})
    assert sm.signal_momentum(short_df) == (-1, 0.0, "momentum")
    df = pd.DataFrame({"close": [1, 2, 3, float("nan")]})
    s, w, _ = sm.signal_momentum(df)
    assert s == -1 and w == 0.0
    short_mean_rev = pd.DataFrame({"close": [1.0] * 10})
    assert sm.signal_mean_reversion(short_mean_rev) == (-1, 0.0, "mean_reversion")
    df2 = pd.DataFrame({"close": [1.0] * 20 + [float("nan")]})
    s2, w2, _ = sm.signal_mean_reversion(df2)
    assert s2 == -1 and w2 == 0.0
