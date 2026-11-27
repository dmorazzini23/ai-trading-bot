from __future__ import annotations

from types import SimpleNamespace

from ai_trading.core import bot_engine


def test_regime_profile_aggressive_present() -> None:
    assert "aggressive" in bot_engine._REGIME_SIGNAL_COMPONENTS
    components = bot_engine._REGIME_SIGNAL_COMPONENTS["aggressive"]["trending"]
    assert "momentum" in components

