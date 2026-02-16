from __future__ import annotations

from types import SimpleNamespace

from ai_trading.core import bot_engine


def test_regime_profile_aggressive_present() -> None:
    assert "aggressive" in bot_engine._REGIME_SIGNAL_COMPONENTS
    components = bot_engine._REGIME_SIGNAL_COMPONENTS["aggressive"]["trending"]
    assert "momentum" in components


def test_regime_profile_aggressive_resolves(monkeypatch) -> None:
    monkeypatch.setattr(bot_engine, "get_regime_signal_routing_enabled", lambda: True)
    monkeypatch.setattr(bot_engine, "get_regime_signal_profile", lambda: "aggressive")
    state = SimpleNamespace(current_regime="trending")
    resolved = bot_engine._resolve_regime_signal_components(state)
    assert "momentum" in resolved
