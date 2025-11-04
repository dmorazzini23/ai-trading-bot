from __future__ import annotations

import logging

from ai_trading.core import bot_engine


def test_data_retry_settings_clamped_and_logged(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="ai_trading.core.bot_engine")
    monkeypatch.setenv("DATA_SOURCE_RETRY_ATTEMPTS", "9")
    monkeypatch.setenv("DATA_SOURCE_RETRY_DELAY_SECONDS", "9.25")
    monkeypatch.setattr(bot_engine, "_DATA_RETRY_SETTINGS_CACHE", None, raising=False)
    monkeypatch.setattr(bot_engine, "_DATA_RETRY_SETTINGS_LOGGED", False, raising=False)

    attempts, delay = bot_engine._resolve_data_retry_settings()

    assert attempts == 5
    assert delay == 5.0
    matching = [
        record
        for record in caplog.records
        if record.getMessage() == "DATA_RETRY_SETTINGS"
    ]
    assert matching, "expected DATA_RETRY_SETTINGS log"
    assert matching[0].attempts == 5
    assert matching[0].delay_seconds == 5.0
