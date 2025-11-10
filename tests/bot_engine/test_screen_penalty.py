from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine


def test_truncate_degraded_candidates_logs_penalty(caplog):
    runtime = SimpleNamespace(cfg=None)
    symbols = ["AAPL", "MSFT", "TSLA", "AMZN"]
    with caplog.at_level("WARNING"):
        trimmed = bot_engine._truncate_degraded_candidates(symbols, runtime, reason="provider_disabled")
    assert len(trimmed) <= len(symbols)
    penalty_logs = [record for record in caplog.records if record.message == "SCREEN_DEGRADED_INPUTS"]
    assert penalty_logs, "SCREEN_DEGRADED_INPUTS log not emitted"
    payload = penalty_logs[0].__dict__
    assert payload.get("penalty_factor") is not None
    assert payload.get("reason") == "provider_disabled"
