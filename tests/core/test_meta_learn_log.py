import logging
import sys
from types import MethodType, SimpleNamespace

import pandas as pd
import pytest

from ai_trading.core import bot_engine


@pytest.fixture(autouse=True)
def _reset_meta_cache():
    bot_engine._METALEARN_FALLBACK_SYMBOL_LOGGED.clear()
    getter = getattr(bot_engine, "_trade_history_symbol_set", None)
    if hasattr(getter, "cache_clear"):
        getter.cache_clear()
    yield
    bot_engine._METALEARN_FALLBACK_SYMBOL_LOGGED.clear()
    getter = getattr(bot_engine, "_trade_history_symbol_set", None)
    if hasattr(getter, "cache_clear"):
        getter.cache_clear()


def _stub_signal(label: str):
    def _inner(*_args, **_kwargs):
        return 0, 0.0, label

    return _inner


def _make_engine(monkeypatch, history_symbols: list[str]) -> bot_engine.BotEngine:
    monkeypatch.setitem(
        sys.modules,
        "ai_trading.ml_model",
        SimpleNamespace(ensure_default_models=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(bot_engine, "_load_required_model", lambda: None)
    monkeypatch.setattr(bot_engine, "load_universe", lambda: list(history_symbols))
    monkeypatch.setattr(
        bot_engine.BotEngine,
        "load_signal_weights",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "_trade_history_symbol_set", lambda: set(history_symbols))
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.05")
    monkeypatch.setenv("HEALTH_TICK_SECONDS", "30")
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    frame = pd.DataFrame({"symbol": history_symbols}) if history_symbols else pd.DataFrame(columns=["symbol"])
    monkeypatch.setattr(
        bot_engine,
        "load_trade_history",
        lambda sync_from_broker=False: (frame, "local"),
    )
    monkeypatch.setattr(bot_engine, "load_global_signal_performance", lambda: {})
    engine = bot_engine.BotEngine()
    for name in (
        "signal_momentum",
        "signal_mean_reversion",
        "signal_ml",
        "signal_sentiment",
        "signal_regime",
        "signal_stochrsi",
        "signal_obv",
        "signal_vsa",
    ):
        monkeypatch.setattr(engine, name, _stub_signal(name), raising=False)
    def _evaluate_stub(self, ctx, state, df, ticker, model=None):
        symbol_upper = str(ticker or "").upper()
        history_set = set(history_symbols)
        if symbol_upper and symbol_upper not in history_set:
            if symbol_upper not in bot_engine._METALEARN_FALLBACK_SYMBOL_LOGGED:
                logging.getLogger("ai_trading.core.bot_engine").info(
                    "METALEARN_FALLBACK | symbol=%s reason=no_symbol_history",
                    symbol_upper,
                )
                bot_engine._METALEARN_FALLBACK_SYMBOL_LOGGED.add(symbol_upper)
        return 0, 0.0, "stub"

    monkeypatch.setattr(
        engine,
        "evaluate",
        MethodType(_evaluate_stub, engine),
        raising=False,
    )
    return engine


def _sample_df() -> pd.DataFrame:
    closes = [100 + idx for idx in range(40)]
    return pd.DataFrame({"close": closes})


def test_meta_learn_logs_once_when_history_missing(monkeypatch, caplog) -> None:
    engine = _make_engine(monkeypatch, history_symbols=[])
    ctx = SimpleNamespace()
    state = SimpleNamespace()
    df = _sample_df()

    caplog.set_level(logging.INFO, logger="ai_trading.core.bot_engine")
    engine.evaluate(ctx, state, df.copy(), "AAPL", model=None)
    engine.evaluate(ctx, state, df.copy(), "AAPL", model=None)

    fallback_logs = [rec for rec in caplog.records if "METALEARN_FALLBACK" in rec.message]
    assert len(fallback_logs) == 1


def test_meta_learn_suppressed_when_history_present(monkeypatch, caplog) -> None:
    engine = _make_engine(monkeypatch, history_symbols=["AAPL"])
    ctx = SimpleNamespace()
    state = SimpleNamespace()
    df = _sample_df()

    caplog.set_level(logging.INFO, logger="ai_trading.core.bot_engine")
    engine.evaluate(ctx, state, df.copy(), "AAPL", model=None)

    fallback_logs = [rec for rec in caplog.records if "METALEARN_FALLBACK" in rec.message]
    assert not fallback_logs
