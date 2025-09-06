import logging
from datetime import date as real_date

import ai_trading.logging.emit_once as emit_once_mod


def test_emit_once_emits_once_per_day(caplog, monkeypatch):
    logger = logging.getLogger("ai_trading.test")
    caplog.set_level(logging.INFO)

    class Day1(real_date):
        @classmethod
        def today(cls) -> real_date:
            return real_date(2024, 1, 1)

    monkeypatch.setattr(emit_once_mod, "date", Day1)

    assert emit_once_mod.emit_once(logger, "UNIQUE_KEY", "info", "Hello") is True
    assert emit_once_mod.emit_once(logger, "UNIQUE_KEY", "info", "Hello") is False

    class Day2(real_date):
        @classmethod
        def today(cls) -> real_date:
            return real_date(2024, 1, 2)

    monkeypatch.setattr(emit_once_mod, "date", Day2)

    assert emit_once_mod.emit_once(logger, "UNIQUE_KEY", "info", "Hello") is True

    msgs = [r.message for r in caplog.records]
    assert msgs.count("Hello") == 2

