from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace

import ai_trading.main as main


def test_preflight_logs_context_fields_on_fallback(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=main.logger.name)
    stub_cfg = SimpleNamespace(
        paper=True,
        alpaca_base_url="https://paper-api.alpaca.markets",
        alpaca_has_sip=False,
        alpaca_allow_sip=False,
    )
    monkeypatch.setattr(main, "get_trading_config", lambda: stub_cfg)
    monkeypatch.setattr(main, "get_settings", lambda: stub_cfg)
    monkeypatch.setattr(main, "enforce_alpaca_feed_policy", lambda: {"status": "fallback", "provider": "alpaca", "feed": "iex"})
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace(__name__=name))
    monkeypatch.setattr(main, "ensure_trade_log_path", lambda: None)

    assert main.preflight_import_health() is True

    records = [rec for rec in caplog.records if rec.getMessage() == "ALPACA_PROVIDER_PREFLIGHT"]
    assert records, "expected ALPACA_PROVIDER_PREFLIGHT log"
    record = records[0]
    assert record.status == "fallback"
    assert record.paper is True
    assert record.alpaca_base_url == "https://paper-api.alpaca.markets"
    assert record.alpaca_has_sip is False
    assert record.alpaca_allow_sip is False
