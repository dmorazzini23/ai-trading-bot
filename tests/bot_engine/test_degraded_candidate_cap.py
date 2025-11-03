from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("TRADING__DEGRADED_MAX_CANDIDATES", raising=False)
    yield
    monkeypatch.delenv("TRADING__DEGRADED_MAX_CANDIDATES", raising=False)


def test_truncate_degraded_candidates_respects_config(caplog):
    runtime = SimpleNamespace(cfg=SimpleNamespace(degraded_max_candidates=2))
    symbols = ["AAPL", "MSFT", "GOOG"]
    caplog.set_level(logging.WARNING, logger=bot_engine.logger.name)

    truncated = bot_engine._truncate_degraded_candidates(list(symbols), runtime)

    assert truncated == ["AAPL", "MSFT"]
    assert any(record.msg == "DEGRADED_CANDIDATES_TRUNCATED" for record in caplog.records)


def test_truncate_degraded_candidates_env_fallback(monkeypatch, caplog):
    runtime = SimpleNamespace(cfg=SimpleNamespace(degraded_max_candidates=None))
    symbols = ["AAPL", "MSFT"]
    monkeypatch.setenv("TRADING__DEGRADED_MAX_CANDIDATES", "1")
    caplog.set_level(logging.WARNING, logger=bot_engine.logger.name)

    truncated = bot_engine._truncate_degraded_candidates(list(symbols), runtime)

    assert truncated == ["AAPL"]
    assert any(record.msg == "DEGRADED_CANDIDATES_TRUNCATED" for record in caplog.records)
