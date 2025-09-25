import logging
from datetime import date as real_date

import ai_trading.logging.emit_once as emit_once_mod


def test_emit_once_emits_once_per_day(caplog, monkeypatch):
    logger = logging.getLogger("ai_trading.test")
    caplog.set_level(logging.INFO)

    monkeypatch.setattr(emit_once_mod, "_utc_today", lambda: real_date(2024, 1, 1))

    assert emit_once_mod.emit_once(logger, "UNIQUE_KEY", "info", "Hello") is True
    assert emit_once_mod.emit_once(logger, "UNIQUE_KEY", "info", "Hello") is False

    monkeypatch.setattr(emit_once_mod, "_utc_today", lambda: real_date(2024, 1, 2))

    assert emit_once_mod.emit_once(logger, "UNIQUE_KEY", "info", "Hello") is True

    msgs = [r.message for r in caplog.records]
    assert msgs.count("Hello") == 2


def test_emit_once_accepts_logger_adapter(caplog, monkeypatch):
    base_logger = logging.getLogger("ai_trading.test.adapter")
    adapter = logging.LoggerAdapter(base_logger, {})
    caplog.set_level(logging.INFO)

    monkeypatch.setattr(emit_once_mod, "_utc_today", lambda: real_date(2024, 1, 1))

    assert emit_once_mod.emit_once(adapter, "ADAPTER_KEY", "info", "Adapter Hello") is True
    assert emit_once_mod.emit_once(adapter, "ADAPTER_KEY", "info", "Adapter Hello") is False

    msgs = [r.message for r in caplog.records]
    assert msgs.count("Adapter Hello") == 1

