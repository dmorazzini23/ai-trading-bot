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


def test_signal_manager_regime_routing_filters_components(monkeypatch):
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
        lambda _conf: 0.50,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "_METALEARN_FALLBACK_SYMBOL_LOGGED",
        set(),
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "get_regime_signal_routing_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "get_regime_signal_profile",
        lambda: "conservative",
        raising=False,
    )

    call_counts: dict[str, int] = {}

    def _track(result):
        def _inner(*_args, **_kwargs):
            label = result[2]
            call_counts[label] = call_counts.get(label, 0) + 1
            return result

        return _inner

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
        monkeypatch.setattr(manager, name, _track(result), raising=False)

    closes = pd.Series(range(250), dtype="float64")
    df = pd.DataFrame({"close": closes})
    ctx = SimpleNamespace()
    state = SimpleNamespace(current_regime="trending")

    signal, confidence, label = manager.evaluate(ctx, state, df, "TST", None)

    assert signal == 1
    assert confidence == 0.50
    assert label == "momentum+ml+regime+obv"
    assert call_counts.get("momentum", 0) == 1
    assert call_counts.get("ml", 0) == 1
    assert call_counts.get("regime", 0) == 1
    assert call_counts.get("obv", 0) == 1
    assert call_counts.get("mean_reversion", 0) == 0
    assert call_counts.get("stochrsi", 0) == 0
    assert call_counts.get("sentiment", 0) == 0
    assert call_counts.get("vsa", 0) == 0
