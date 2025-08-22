"""Tests for data_fetcher.build_fetcher initialization paths."""  # AI-AGENT-REF

from __future__ import annotations

import pandas as pd

import sys
import types

from ai_trading import data_fetcher as df_module


def test_build_fetcher_with_alpaca_keys(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    dummy_alpaca = types.SimpleNamespace(ALPACA_AVAILABLE=True)
    monkeypatch.setitem(sys.modules, "ai_trading.alpaca_api", dummy_alpaca)
    class DummyFetcher:
        def get_daily_df(self, ctx, sym):
            return pd.DataFrame()

        def get_minute_df(self, ctx, sym):
            return pd.DataFrame()

    dummy_core = types.SimpleNamespace(DataFetcher=DummyFetcher, DataFetchError=Exception)
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", dummy_core)
    fetcher = df_module.build_fetcher(object())
    assert fetcher is not None


def test_build_fetcher_fallback_requests(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    dummy_alpaca = types.SimpleNamespace(ALPACA_AVAILABLE=False)
    monkeypatch.setitem(sys.modules, "ai_trading.alpaca_api", dummy_alpaca)
    class DummyFetcher:
        def get_daily_df(self, ctx, sym):
            return pd.DataFrame()

        def get_minute_df(self, ctx, sym):
            return pd.DataFrame()

    dummy_core = types.SimpleNamespace(DataFetcher=DummyFetcher, DataFetchError=Exception)
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", dummy_core)
    monkeypatch.setattr(df_module, "yf", None, raising=False)
    monkeypatch.setattr(df_module, "requests", object(), raising=False)
    fetcher = df_module.build_fetcher(object())
    assert fetcher is not None


def test_build_fetcher_offline_returns_empty_df(monkeypatch):
    dummy_alpaca = types.SimpleNamespace(ALPACA_AVAILABLE=False)
    monkeypatch.setitem(sys.modules, "ai_trading.alpaca_api", dummy_alpaca)
    class DummyFetcher:
        def get_daily_df(self, ctx, sym):
            return pd.DataFrame()

        def get_minute_df(self, ctx, sym):
            return pd.DataFrame()

    dummy_core = types.SimpleNamespace(DataFetcher=DummyFetcher, DataFetchError=Exception)
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", dummy_core)
    monkeypatch.setattr(df_module, "yf", None, raising=False)
    monkeypatch.setattr(df_module, "requests", None, raising=False)
    fetcher = df_module.build_fetcher(object())
    df = fetcher.get_daily_df(object(), "SPY")
    assert isinstance(df, pd.DataFrame)
    assert df.empty

