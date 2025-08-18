import logging
from types import SimpleNamespace

from ai_trading.core.bot_engine import check_pdt_rule, safe_alpaca_get_account


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
