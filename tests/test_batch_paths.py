import types
from typing import Any, cast

import ai_trading.core.bot_engine as be
import pytest
pd = pytest.importorskip("pandas")
def _mk_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "open": [1,2,3], "high":[1,2,3], "low":[1,2,3], "close":[1,2,3], "volume":[100,200,300]
        }
    )



def test_get_daily_df_for_feed_role_falls_back_without_keyword() -> None:
    class _Fetcher:
        def get_daily_df(self, ctx, symbol):
            assert symbol == "SPY"
            return _mk_df()

    ctx = types.SimpleNamespace(data_fetcher=_Fetcher())
    out = be._get_daily_df_for_feed_role(ctx, "SPY", feed_role="reference")
    assert out is not None
    assert not out.empty



def test_pretrade_batch(monkeypatch):
    ctx = types.SimpleNamespace(lookback_start="2024-01-01", lookback_end="2024-02-01", data_feed=None, min_rows=2)
    monkeypatch.setattr(be, "_fetch_universe_bars_chunked", lambda **kwargs: {"AAPL": _mk_df()})
    res = be.pre_trade_health_check(ctx, ["AAPL"])
    assert res["checked"] == 1



