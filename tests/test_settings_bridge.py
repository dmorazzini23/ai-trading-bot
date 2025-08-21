"""Ensure legacy settings module re-exports modern config."""

from ai_trading.settings import get_settings as legacy_get
from ai_trading.config.settings import get_settings as modern_get


def test_settings_bridge_alias():
    """Legacy get_settings should reference modern config."""  # AI-AGENT-REF
    assert legacy_get is modern_get

