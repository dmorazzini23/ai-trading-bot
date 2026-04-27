import types

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core import bot_engine as bot


def test_screen_candidates_empty(monkeypatch):
    """screen_candidates returns an empty list when none pass."""
    monkeypatch.setattr(bot, "screen_universe", lambda candidates, runtime: [])

    # Create a mock runtime object
    from unittest.mock import Mock
    mock_runtime = Mock()

    assert bot.screen_candidates(mock_runtime, ["AAA"]) == []


def test_screen_universe_atr_fallback(monkeypatch):
    """screen_universe falls back to internal ATR when pandas_ta is missing."""
    import ai_trading.indicators as indicators

    monkeypatch.setattr(bot, "ta", types.SimpleNamespace(_failed=True))
    monkeypatch.setattr(bot, "_SCREEN_CACHE", {})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {})
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)
    monkeypatch.setattr(bot, "is_valid_ohlcv", lambda df: True)
    monkeypatch.setattr(bot, "_validate_market_data_quality", lambda df, s: {"valid": True})

    called = {}

    def fake_atr(h, l, c, period=14):
        called["used"] = True
        return pd.Series([1.0] * len(h))

    monkeypatch.setattr(indicators, "atr", fake_atr)

    rows = bot.ATR_LENGTH + 1
    df = pd.DataFrame(
        {
            "high": range(2, 2 + rows),
            "low": range(1, 1 + rows),
            "close": [x + 0.5 for x in range(1, 1 + rows)],
            "volume": [200_000] * rows,
        }
    )

    class DummyFetcher:
        def get_daily_df(self, runtime, sym):
            return df

    runtime = types.SimpleNamespace(data_fetcher=DummyFetcher())

    result = bot.screen_universe(["AAA"], runtime)

    assert result == ["AAA"]
    assert called.get("used") is True


def test_screen_universe_refetches_for_missing_atr(monkeypatch):
    """screen_universe fetches more history when ATR cannot be computed."""
    import ai_trading.data.fetch as data_fetch

    monkeypatch.setattr(bot, "_SCREEN_CACHE", {})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {})
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)
    monkeypatch.setattr(bot, "is_valid_ohlcv", lambda df: True)
    monkeypatch.setattr(bot, "_validate_market_data_quality", lambda df, s: {"valid": True})

    rows_short = bot.ATR_LENGTH - 1
    df_short = pd.DataFrame(
        {
            "high": range(rows_short),
            "low": range(rows_short),
            "close": range(rows_short),
            "volume": [200_000] * rows_short,
        }
    )
    rows_long = bot.ATR_LENGTH + 5
    df_long = pd.DataFrame(
        {
            "high": range(rows_long),
            "low": range(rows_long),
            "close": range(rows_long),
            "volume": [200_000] * rows_long,
        }
    )

    class DummyFetcher:
        def get_daily_df(self, runtime, sym):
            return df_short

    runtime = types.SimpleNamespace(data_fetcher=DummyFetcher())

    called = {"extended": False}

    def fake_get_daily_df(symbol, start, end):
        called["extended"] = True
        return df_long

    monkeypatch.setattr(data_fetch, "get_daily_df", fake_get_daily_df)
    def fake_atr(h, l, c, length=bot.ATR_LENGTH):
        if len(h) < length:
            return pd.Series([])
        return pd.Series([1.0] * len(h))

    monkeypatch.setattr(bot, "ta", types.SimpleNamespace(atr=fake_atr))

    result = bot.screen_universe(["AAA"], runtime)

    assert result == ["AAA"]
    assert called["extended"] is True


def test_screen_universe_skips_when_atr_still_missing(monkeypatch):
    """screen_universe skips symbols when ATR remains unavailable."""
    import ai_trading.data.fetch as data_fetch

    monkeypatch.setattr(bot, "_SCREEN_CACHE", {})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {})
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)
    monkeypatch.setattr(bot, "is_valid_ohlcv", lambda df: True)
    monkeypatch.setattr(bot, "_validate_market_data_quality", lambda df, s: {"valid": True})

    rows_short = bot.ATR_LENGTH - 1
    df_short = pd.DataFrame(
        {
            "high": range(rows_short),
            "low": range(rows_short),
            "close": range(rows_short),
            "volume": [200_000] * rows_short,
        }
    )

    class DummyFetcher:
        def get_daily_df(self, runtime, sym):
            return df_short

    runtime = types.SimpleNamespace(data_fetcher=DummyFetcher())

    called = {"extended": False}

    def fake_get_daily_df(symbol, start, end):
        called["extended"] = True
        return df_short

    monkeypatch.setattr(data_fetch, "get_daily_df", fake_get_daily_df)
    monkeypatch.setattr(
        bot,
        "ta",
        types.SimpleNamespace(
            atr=lambda h, l, c, length=bot.ATR_LENGTH: pd.Series([])
        ),
    )

    result = bot.screen_universe(["AAA"], runtime)

    assert result == []
    assert called["extended"] is True


def test_screen_universe_reuses_cached_candidates_when_refetch_window_open(monkeypatch):
    monkeypatch.setattr(bot, "_SCREEN_CACHE", {"AAA": 1.25})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {"AAA": bot.time.time()})
    monkeypatch.setattr(bot, "_SCREEN_ROTATE_UNSEEN_ENABLED", False)
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)

    rows = bot.ATR_LENGTH + 3
    df_spy = pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [500_000] * rows,
        }
    )

    class DummyFetcher:
        def __init__(self):
            self.requested: list[str] = []

        def get_daily_df(self, runtime, sym):
            self.requested.append(sym)
            if sym == "SPY":
                return df_spy
            raise AssertionError(f"unexpected fetch for {sym}")

    fetcher = DummyFetcher()
    runtime = types.SimpleNamespace(data_fetcher=fetcher)

    selected = bot.screen_universe(["AAA"], runtime)

    assert selected == ["AAA"]
    assert fetcher.requested == ["SPY"]


def test_screen_universe_does_not_rotate_unseen_when_window_throttled(monkeypatch):
    monkeypatch.setattr(bot, "_SCREEN_CACHE", {})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {"AAA": bot.time.time()})
    monkeypatch.setattr(bot, "_SCREEN_ROTATE_UNSEEN_ENABLED", False)
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)

    class DummyFetcher:
        def get_daily_df(self, runtime, sym):
            raise AssertionError(f"unexpected fetch for {sym}")

    runtime = types.SimpleNamespace(data_fetcher=DummyFetcher())

    selected = bot.screen_universe(["AAA"], runtime)

    assert selected == []


def test_resolve_prepare_symbol_limit_prefers_explicit_setting(monkeypatch):
    monkeypatch.setenv("AI_TRADING_PREPARE_SYMBOL_LIMIT", "12")
    monkeypatch.setenv("MAX_SYMBOLS_PER_CYCLE", "3")
    assert bot._resolve_prepare_symbol_limit() == 12


def test_resolve_prepare_symbol_limit_falls_back_to_max_symbols(monkeypatch):
    monkeypatch.setenv("AI_TRADING_PREPARE_SYMBOL_LIMIT", "0")
    monkeypatch.setenv("MAX_SYMBOLS_PER_CYCLE", "4")
    assert bot._resolve_prepare_symbol_limit() == 4
