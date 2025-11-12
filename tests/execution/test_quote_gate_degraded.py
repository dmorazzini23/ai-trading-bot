from ai_trading.config.runtime import reload_trading_config
from ai_trading.core import bot_engine


def test_last_close_allowed_when_degraded(monkeypatch):
    monkeypatch.setenv("EXECUTION_ALLOW_LAST_CLOSE", "0")
    monkeypatch.setenv("EXECUTION_MARKET_ON_DEGRADED", "0")
    monkeypatch.setenv("TRADING__DEGRADED_FEED_MODE", "block")
    reload_trading_config()
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: True)

    assert bot_engine._safe_mode_blocks_trading() is True
    assert bot_engine._allow_last_close_execution() is False

    monkeypatch.setenv("TRADING__DEGRADED_FEED_MODE", "widen")
    reload_trading_config()

    assert bot_engine._allow_last_close_execution() is True

    monkeypatch.delenv("TRADING__DEGRADED_FEED_MODE", raising=False)
    monkeypatch.delenv("EXECUTION_MARKET_ON_DEGRADED", raising=False)
    monkeypatch.delenv("EXECUTION_ALLOW_LAST_CLOSE", raising=False)
    reload_trading_config()
