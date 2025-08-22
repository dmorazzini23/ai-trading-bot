"""Ensure legacy settings module re-exports modern config."""

from ai_trading.config.settings import get_settings as modern_get
from ai_trading.settings import get_settings as legacy_get


def test_settings_bridge_alias():
    """Legacy and modern helpers return the same instance."""  # AI-AGENT-REF
    assert legacy_get() is modern_get()

