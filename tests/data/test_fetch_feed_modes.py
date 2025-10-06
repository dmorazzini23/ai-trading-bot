import importlib

import ai_trading.core.bot_engine as bot_engine_module
import ai_trading.data.fetch as fetch_module


def test_iex_ignores_sip_unauthorized(monkeypatch):
    with monkeypatch.context() as patch:
        patch.delenv("PYTEST_RUNNING", raising=False)
        patch.delenv("PYTEST_CURRENT_TEST", raising=False)
        patch.delenv("ALPACA_DATA_FEED", raising=False)
        patch.delenv("ALPACA_FEED_FAILOVER", raising=False)
        patch.delenv("ALPACA_ALLOW_SIP", raising=False)
        patch.delenv("ALPACA_HAS_SIP", raising=False)
        patch.setenv("DATA_FEED_INTRADAY", "iex")
        patch.setenv("ALPACA_SIP_UNAUTHORIZED", "1")

        fetch = importlib.reload(fetch_module)
        bot_engine = importlib.reload(bot_engine_module)

        assert fetch._sip_lock_relevant() is False
        assert fetch.is_primary_provider_enabled() is True
        assert bot_engine._sip_lockout_active() is False

    importlib.reload(fetch_module)
    importlib.reload(bot_engine_module)
