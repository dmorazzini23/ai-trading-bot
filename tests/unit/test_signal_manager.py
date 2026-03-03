from types import SimpleNamespace

import pandas as pd
import pytest

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


def test_signal_manager_applies_meta_component_weights(monkeypatch):
    manager = bot_engine.SignalManager()
    monkeypatch.setattr(
        bot_engine,
        "load_global_signal_performance",
        lambda: {"momentum": 1.0, "ml": 0.2},
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "_trade_history_symbol_set",
        lambda: {"TST"},
        raising=False,
    )
    monkeypatch.setattr(
        manager,
        "load_signal_weights",
        lambda: {"momentum": 0.8, "ml": 0.1},
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "_METALEARN_FALLBACK_SYMBOL_LOGGED",
        set(),
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "signals_evaluated", None, raising=False)
    monkeypatch.setenv("AI_TRADING_META_COMPONENT_WEIGHTING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_META_COMPONENT_GATE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_META_WEIGHT_STRENGTH", "1.0")
    monkeypatch.setenv("AI_TRADING_META_PERFORMANCE_STRENGTH", "1.0")
    monkeypatch.setenv("AI_TRADING_META_COMPONENT_MAX_WEIGHT", "1.5")

    monkeypatch.setattr(
        manager,
        "signal_momentum",
        _stub_signal((1, 0.5, "momentum")),
        raising=False,
    )
    monkeypatch.setattr(
        manager,
        "signal_ml",
        _stub_signal((-1, 0.5, "ml")),
        raising=False,
    )
    monkeypatch.setattr(manager, "signal_mean_reversion", _stub_signal(None), raising=False)
    monkeypatch.setattr(manager, "signal_sentiment", _stub_signal(None), raising=False)
    monkeypatch.setattr(manager, "signal_regime", _stub_signal(None), raising=False)
    monkeypatch.setattr(manager, "signal_stochrsi", _stub_signal(None), raising=False)
    monkeypatch.setattr(manager, "signal_obv", _stub_signal(None), raising=False)
    monkeypatch.setattr(manager, "signal_vsa", _stub_signal(None), raising=False)

    df = pd.DataFrame({"close": pd.Series(range(250), dtype="float64")})
    ctx = SimpleNamespace()
    state = SimpleNamespace(current_regime="trending")
    _signal, _confidence, _label = manager.evaluate(ctx, state, df, "TST", None)
    by_label = {label: weight for _side, weight, label in manager.last_components}
    assert by_label["momentum"] == pytest.approx(1.5)
    assert by_label["ml"] < 0.1


def test_signal_manager_gate_downweights_non_qualified_tags(monkeypatch):
    manager = bot_engine.SignalManager()
    monkeypatch.setattr(
        bot_engine,
        "load_global_signal_performance",
        lambda: {"momentum": 0.6},
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "_trade_history_symbol_set",
        lambda: {"TST"},
        raising=False,
    )
    monkeypatch.setattr(manager, "load_signal_weights", lambda: {}, raising=False)
    monkeypatch.setattr(
        bot_engine,
        "_METALEARN_FALLBACK_SYMBOL_LOGGED",
        set(),
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "signals_evaluated", None, raising=False)
    monkeypatch.setenv("AI_TRADING_META_COMPONENT_WEIGHTING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_META_COMPONENT_GATE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_META_PERFORMANCE_STRENGTH", "0")
    monkeypatch.setenv("AI_TRADING_META_UNSEEN_TAG_MULTIPLIER", "1.0")
    monkeypatch.setenv("AI_TRADING_META_COMPONENT_OUT_OF_SET_MULTIPLIER", "0.2")

    monkeypatch.setattr(
        manager,
        "signal_momentum",
        _stub_signal((1, 0.4, "momentum")),
        raising=False,
    )
    monkeypatch.setattr(
        manager,
        "signal_ml",
        _stub_signal((1, 0.4, "ml")),
        raising=False,
    )
    monkeypatch.setattr(manager, "signal_mean_reversion", _stub_signal(None), raising=False)
    monkeypatch.setattr(manager, "signal_sentiment", _stub_signal(None), raising=False)
    monkeypatch.setattr(manager, "signal_regime", _stub_signal(None), raising=False)
    monkeypatch.setattr(manager, "signal_stochrsi", _stub_signal(None), raising=False)
    monkeypatch.setattr(manager, "signal_obv", _stub_signal(None), raising=False)
    monkeypatch.setattr(manager, "signal_vsa", _stub_signal(None), raising=False)

    df = pd.DataFrame({"close": pd.Series(range(250), dtype="float64")})
    ctx = SimpleNamespace()
    state = SimpleNamespace(current_regime="trending")
    _signal, _confidence, _label = manager.evaluate(ctx, state, df, "TST", None)
    by_label = {label: weight for _side, weight, label in manager.last_components}
    assert by_label["momentum"] == pytest.approx(0.4)
    assert by_label["ml"] == pytest.approx(0.08)
