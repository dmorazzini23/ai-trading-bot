import logging

from ai_trading.logging.emit_once import emit_once


def test_emit_once_emits_only_first_time(caplog):
    logger = logging.getLogger("ai_trading.test")
    caplog.set_level(logging.INFO)

    assert emit_once(logger, "UNIQUE_KEY", "info", "Hello") is True
    assert emit_once(logger, "UNIQUE_KEY", "info", "Hello") is False

    msgs = [r.message for r in caplog.records]
    assert msgs.count("Hello") == 1

