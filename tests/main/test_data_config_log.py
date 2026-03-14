from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import ai_trading.main as main
from ai_trading.telemetry import runtime_state


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
        alpaca_feed_ignored=False,
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


def test_emit_data_config_log_appends_note_when_feed_ignored(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=main.logger.name)
    fallback_cfg = SimpleNamespace(
        data_provider="finnhub",
        alpaca_data_feed="sip",
        alpaca_adjustment="split",
        alpaca_feed_ignored=True,
    )
    monkeypatch.setattr(main, "get_trading_config", lambda: fallback_cfg)
    settings = SimpleNamespace(alpaca_data_feed="sip", alpaca_adjustment="split")
    config = SimpleNamespace(data_provider="finnhub")

    main._emit_data_config_log(settings, config)

    record = next(rec for rec in caplog.records if rec.getMessage().startswith("DATA_CONFIG"))
    assert "note=feed only used with alpaca* providers" in record.getMessage()
    assert getattr(record, "note", None) == "feed only used with alpaca* providers"


def test_emit_data_config_log_initializes_provider_runtime_state(monkeypatch):
    runtime_state.reset_data_provider_state()
    fallback_cfg = SimpleNamespace(
        data_provider="alpaca_iex",
        alpaca_data_feed="iex",
        alpaca_adjustment="all",
        alpaca_feed_ignored=False,
    )
    monkeypatch.setattr(main, "get_trading_config", lambda: fallback_cfg)
    settings = SimpleNamespace(alpaca_data_feed="iex", alpaca_adjustment="raw")
    config = SimpleNamespace(data_provider="alpaca_iex")

    main._emit_data_config_log(settings, config)

    provider_state = runtime_state.observe_data_provider_state()
    assert provider_state["primary"] == "alpaca"
    assert provider_state["active"] == "alpaca"
    assert provider_state["using_backup"] is False
    assert provider_state["reason"] == "startup_config_resolved"
    assert provider_state["status"] == "warming_up"
    assert provider_state["data_status"] == "warming_up"
    assert provider_state["updated"] is not None
