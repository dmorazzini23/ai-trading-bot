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


def test_resolve_data_provider_degraded_uses_runtime_state(monkeypatch):
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: {"using_backup": True, "reason": "using_backup", "status": "degraded"},
    )
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: None)
    monkeypatch.setattr(
        bot_engine.provider_monitor,
        "is_disabled",
        lambda *_a, **_k: False,
    )

    degraded, reason = bot_engine._resolve_data_provider_degraded()

    assert degraded is True
    assert reason == "using_backup"


def test_process_symbols_skips_when_degraded(monkeypatch, caplog):
    runtime = SimpleNamespace(
        _data_degraded=True,
        _data_degraded_reason="using_backup",
        execution_engine=None,
    )
    monkeypatch.setattr(bot_engine, "get_ctx", lambda: runtime)
    monkeypatch.setattr(
        bot_engine,
        "state",
        SimpleNamespace(
            position_cache={},
            trade_cooldowns={},
            last_trade_direction={},
        ),
    )
    caplog.set_level(logging.WARNING, logger=bot_engine.logger.name)

    processed, row_counts = bot_engine._process_symbols(
        ["AAPL", "MSFT"],
        current_cash=100000.0,
        model=None,
        regime_ok=True,
    )

    assert processed == ["AAPL", "MSFT"]
    assert row_counts == {}
    assert any(record.msg == "DEGRADED_FEED_SKIP_SYMBOL" for record in caplog.records)
