from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import ai_trading.main as main


@pytest.fixture(autouse=True)
def _reset_trading_config(monkeypatch):
    """Restore get_trading_config after each test."""

    original = main.get_trading_config
    yield
    monkeypatch.setattr(main, "get_trading_config", original, raising=False)


def test_emit_data_config_log_prefers_trading_config_values(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=main.logger.name)
    fallback_cfg = SimpleNamespace(
        data_provider="finnhub",
        alpaca_data_feed="sip",
        alpaca_adjustment="split",
    )
    monkeypatch.setattr(main, "get_trading_config", lambda: fallback_cfg)
    settings = SimpleNamespace(alpaca_data_feed="iex", alpaca_adjustment="raw")
    config = SimpleNamespace(data_provider="alpaca")

    main._emit_data_config_log(settings, config)

    record = next(rec for rec in caplog.records if rec.getMessage().startswith("DATA_CONFIG"))
    message = record.getMessage()
    assert "feed=sip" in message
    assert "adjustment=split" in message
    assert "provider=finnhub" in message
