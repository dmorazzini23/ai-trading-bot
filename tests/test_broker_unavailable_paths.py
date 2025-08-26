import logging
from types import SimpleNamespace

from ai_trading.core import bot_engine
from ai_trading.core.bot_engine import check_pdt_rule, safe_alpaca_get_account
from ai_trading.logging import logger_once


def test_safe_account_none():
    # AI-AGENT-REF: ensure None is returned when Alpaca client missing
    ctx = SimpleNamespace(api=None)
    assert safe_alpaca_get_account(ctx) is None


def test_pdt_rule_skips_without_false_fail(caplog):
    # AI-AGENT-REF: verify PDT check logs skip and not failure
    ctx = SimpleNamespace(api=None)
    with caplog.at_level(logging.INFO):
        assert check_pdt_rule(ctx) is False
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PDT" in m and "PPED" in m for m in msgs)
    assert not any("PDT_CHECK_FAILED" in m for m in msgs)


def test_run_all_trades_aborts_without_api(monkeypatch, caplog):
    """run_all_trades_worker should abort early when Alpaca client missing."""
    logger_once._emitted_keys.clear()
    state = bot_engine.BotState()
    runtime = bot_engine.get_ctx()
    runtime.api = None
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot_engine, "_ALPACA_IMPORT_ERROR", None)
    heartbeat = {}
    monkeypatch.setattr(bot_engine, "_send_heartbeat", lambda: heartbeat.setdefault("called", True))
    def fail(*a, **k):
        raise AssertionError("PDT should not be called")
    monkeypatch.setattr(bot_engine, "check_pdt_rule", fail)
    with caplog.at_level("ERROR"):
        bot_engine.run_all_trades_worker(state, runtime)
    assert heartbeat.get("called")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("ALPACA_CLIENT_MISSING" in m for m in msgs)
