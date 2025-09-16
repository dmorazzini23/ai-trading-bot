import sys
import types

import pytest
from ai_trading.utils.lazy_imports import load_pandas

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

