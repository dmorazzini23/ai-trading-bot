from types import SimpleNamespace

import pandas as pd

from ai_trading.core import bot_engine


def _stub_signal(result):
    def _inner(*_args, **_kwargs):
        return result

    return _inner


def test_signal_manager_evaluate_updates_last_components(monkeypatch):
    manager = bot_engine.SignalManager()

    monkeypatch.setattr(
        bot_engine,
        "load_global_signal_performance",
        lambda: {},
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "_trade_history_symbol_set",
        lambda: {"TST"},
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "composite_signal_confidence",
        lambda _conf: 0.42,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "_METALEARN_FALLBACK_SYMBOL_LOGGED",
        set(),
        raising=False,
    )

    stubbed = {
        "signal_momentum": (1, 0.1, "momentum"),
        "signal_mean_reversion": (1, 0.1, "mean_reversion"),
        "signal_ml": (1, 0.1, "ml"),
        "signal_sentiment": (1, 0.1, "sentiment"),
        "signal_regime": (1, 0.1, "regime"),
        "signal_stochrsi": (1, 0.1, "stochrsi"),
        "signal_obv": (1, 0.1, "obv"),
        "signal_vsa": (1, 0.1, "vsa"),
    }
    for name, result in stubbed.items():
        monkeypatch.setattr(manager, name, _stub_signal(result), raising=False)

    closes = pd.Series(range(250), dtype="float64")
    df = pd.DataFrame({"close": closes})
    ctx = SimpleNamespace()
    state = SimpleNamespace()

    signal, confidence, label = manager.evaluate(ctx, state, df, "TST", None)

    assert (signal, confidence, label) == (1, 0.42, "+".join(r[2] for r in stubbed.values()))
    assert manager.last_components == list(stubbed.values())
