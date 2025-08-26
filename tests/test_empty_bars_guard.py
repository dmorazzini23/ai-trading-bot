import pandas as pd
import pytest
from types import SimpleNamespace

from ai_trading.core import bot_engine


def _dummy_cfg():
    return SimpleNamespace(
        alpaca_api_key="key",
        alpaca_secret_key_plain="secret",
        alpaca_data_feed="iex",
        alpaca_adjustment=None,
        minute_cache_ttl=60,
        testing=True,
    )


def test_get_daily_df_empty_bars_raises(monkeypatch):
    cfg = _dummy_cfg()
    monkeypatch.setattr(bot_engine, "CFG", cfg)
    monkeypatch.setattr(bot_engine, "get_settings", lambda: cfg)
    fetcher = bot_engine.DataFetcher()
    ctx = SimpleNamespace()
    monkeypatch.setattr(bot_engine, "get_alpaca_secret_key_plain", lambda: "secret")
    monkeypatch.setattr(bot_engine, "safe_get_stock_bars", lambda *a, **k: pd.DataFrame())
    with pytest.raises(bot_engine.DataFetchError):
        fetcher.get_daily_df(ctx, "AAPL")


def test_compute_spy_vol_stats_handles_failure(monkeypatch):
    runtime = SimpleNamespace()
    runtime.data_fetcher = SimpleNamespace(
        get_daily_df=lambda *a, **k: (_ for _ in ()).throw(bot_engine.DataFetchError("boom"))
    )
    calls: list[str] = []
    runtime.halt_manager = SimpleNamespace(
        manual_halt_trading=lambda reason: calls.append(reason)
    )
    bot_engine._VOL_STATS.update({"mean": None, "std": None, "last": None, "last_update": None})
    bot_engine.compute_spy_vol_stats(runtime)
    assert bot_engine._VOL_STATS["mean"] == 0.0
    assert bot_engine._VOL_STATS["std"] == 0.0
    assert calls, "halt manager should be invoked"
