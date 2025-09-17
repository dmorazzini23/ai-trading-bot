from types import SimpleNamespace
import pandas as pd

from ai_trading.core import bot_engine
from ai_trading.core.bot_engine import BotState, SignalManager, _evaluate_trade_signal


def test_signal_manager_hold_when_indicators_drop(monkeypatch):
    sm = SignalManager()
    ctx = SimpleNamespace(signal_manager=sm)
    state = BotState()

    monkeypatch.setattr(bot_engine, "load_global_signal_performance", lambda: {})
    monkeypatch.setattr(SignalManager, "load_signal_weights", lambda self: {})
    monkeypatch.setattr(bot_engine, "signals_evaluated", None, raising=False)

    monkeypatch.setattr(
        SignalManager,
        "signal_momentum",
        lambda self, df, model=None: (1, 0.4, "momentum"),
        raising=False,
    )
    monkeypatch.setattr(
        SignalManager,
        "signal_mean_reversion",
        lambda self, df, model=None: (1, 0.3, "mean_reversion"),
        raising=False,
    )
    monkeypatch.setattr(
        SignalManager,
        "signal_ml",
        lambda self, df, model=None, symbol=None: (1, 0.2, "ml"),
        raising=False,
    )
    monkeypatch.setattr(
        SignalManager,
        "signal_sentiment",
        lambda self, runtime, ticker, df=None, model=None: (1, 0.1, "sentiment"),
        raising=False,
    )
    monkeypatch.setattr(
        SignalManager,
        "signal_regime",
        lambda self, runtime, state, df, model=None: (1, 0.05, "regime"),
        raising=False,
    )
    monkeypatch.setattr(
        SignalManager,
        "signal_stochrsi",
        lambda self, df, model=None: (1, 0.05, "stochrsi"),
        raising=False,
    )
    monkeypatch.setattr(
        SignalManager,
        "signal_obv",
        lambda self, df, model=None: (1, 0.05, "obv"),
        raising=False,
    )
    monkeypatch.setattr(
        SignalManager,
        "signal_vsa",
        lambda self, df, model=None: (1, 0.05, "vsa"),
        raising=False,
    )

    long_df = pd.DataFrame({"close": list(range(1, 251))})
    score, confidence, label = _evaluate_trade_signal(ctx, state, long_df, "AAA", None)
    assert score > 0
    assert confidence > 0
    assert "momentum" in label
    assert ctx.signal_manager.last_components

    short_df = pd.DataFrame({"close": list(range(1, 11))})
    hold_score, hold_confidence, hold_label = _evaluate_trade_signal(
        ctx, state, short_df, "BBB", None
    )

    assert hold_score == 0.0
    assert hold_confidence == 0.0
    assert hold_label == "HOLD"
    assert ctx.signal_manager.last_components == []
