import importlib
import logging

import ai_trading.core.bot_engine as bot_engine


def test_config_loaded_logs_once_and_on_reload(caplog):
    """Config load log fires once and repeats after reload."""
    with caplog.at_level(logging.INFO):
        importlib.reload(bot_engine)
    assert caplog.text.count("Config settings loaded, validation deferred to runtime") == 1

    caplog.clear()
    with caplog.at_level(logging.INFO):
        bot_engine.BotMode()
        bot_engine.BotMode()
    assert "Config settings loaded, validation deferred to runtime" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        bot_engine._reload_env()
        bot_engine.BotMode()
    assert caplog.text.count("Config settings loaded, validation deferred to runtime") == 1
