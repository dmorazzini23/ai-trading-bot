from __future__ import annotations

import logging
from types import SimpleNamespace

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trading.core import bot_engine


class DummyExecutor:
    def submit(self, fn, symbol):  # noqa: ANN001 - test helper
        return SimpleNamespace(result=lambda: fn(symbol))


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("TRADING__DEGRADED_MAX_CANDIDATES", raising=False)
    yield
    monkeypatch.delenv("TRADING__DEGRADED_MAX_CANDIDATES", raising=False)


@pytest.fixture
def symbol_processing_env(monkeypatch):
    runtime = SimpleNamespace(
        _data_degraded=False,
        _data_degraded_reason=None,
        _data_degraded_fatal=False,
        execution_engine=None,
    )
    state = SimpleNamespace(
        position_cache={},
        trade_cooldowns={},
        last_trade_direction={},
        trade_history=[],
    )
    monkeypatch.setattr(bot_engine, "get_ctx", lambda: runtime)
    monkeypatch.setattr(bot_engine, "state", state, raising=False)
    monkeypatch.setattr(bot_engine, "ensure_final_bar", lambda *_, **__: True)
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "log_skip_cooldown", lambda *_, **__: None)
    monkeypatch.setattr(bot_engine, "_trade_limit_reached", lambda *_a: False)
    monkeypatch.setattr(
        bot_engine,
        "skipped_duplicates",
        SimpleNamespace(inc=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "skipped_cooldown",
        SimpleNamespace(inc=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(bot_engine.executors, "_ensure_executors", lambda: None)
    monkeypatch.setattr(bot_engine, "prediction_executor", DummyExecutor(), raising=False)
    monkeypatch.setattr(bot_engine, "_safe_trade", lambda *_, **__: None)
    monkeypatch.setattr(bot_engine, "get_cycle_budget_context", lambda: None)
    return runtime, state


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
        lambda: {
            "using_backup": True,
            "reason": "using_backup",
            "status": "degraded",
            "timeframes": {"1Min": True},
        },
    )
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: None)
    monkeypatch.setattr(
        bot_engine.provider_monitor,
        "is_disabled",
        lambda *_a, **_k: False,
    )

    degraded, reason, fatal = bot_engine._resolve_data_provider_degraded()

    assert degraded is True
    assert reason == "using_backup"
    assert fatal is False


def test_resolve_data_provider_degraded_ignores_daily_fallback(monkeypatch):
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "using_backup": True,
            "reason": "daily_fallback",
            "status": "healthy",
            "timeframes": {"1Day": True},
        },
    )
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: None)
    monkeypatch.setattr(
        bot_engine.provider_monitor,
        "is_disabled",
        lambda *_a, **_k: False,
    )

    degraded, reason, fatal = bot_engine._resolve_data_provider_degraded()

    assert degraded is False
    assert reason == "daily_fallback"
    assert fatal is False


def test_process_symbols_skips_when_degraded(symbol_processing_env, caplog):
    runtime, _state = symbol_processing_env
    runtime._data_degraded = True
    runtime._data_degraded_reason = "provider_disabled"
    runtime._data_degraded_fatal = True
    caplog.set_level(logging.WARNING, logger=bot_engine.logger.name)

    processed, row_counts, _ = bot_engine._process_symbols(
        ["AAPL", "MSFT"],
        current_cash=100000.0,
        model=None,
        regime_ok=True,
    )

    assert processed == []
    assert row_counts == {}
    assert any(record.msg == "DEGRADED_FEED_SKIP_SYMBOL" for record in caplog.records)


def test_process_symbols_detects_degrade_mid_cycle(symbol_processing_env, monkeypatch, caplog):
    runtime, _state = symbol_processing_env
    runtime._data_degraded = False
    runtime._data_degraded_reason = None
    runtime._data_degraded_fatal = False

    detections = iter(
        [
            (True, "using_backup_provider", False),
            (True, "using_backup_provider", False),
        ]
    )

    monkeypatch.setattr(
        bot_engine,
        "_resolve_data_provider_degraded",
        lambda: next(detections, (True, "using_backup_provider", False)),
    )

    frame = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [100],
        }
    )

    monkeypatch.setattr(
        bot_engine,
        "fetch_minute_df_safe",
        lambda _symbol: frame.copy(),
    )

    caplog.set_level(logging.WARNING, logger=bot_engine.logger.name)

    processed, row_counts, _ = bot_engine._process_symbols(
        ["AAPL", "MSFT"],
        current_cash=100000.0,
        model=None,
        regime_ok=True,
    )

    assert processed == ["AAPL", "MSFT"]
    assert row_counts == {"AAPL": 1, "MSFT": 1}
    assert any(record.msg == "DEGRADED_FEED_ACTIVE" for record in caplog.records)
    assert not any(record.msg == "DEGRADED_FEED_SKIP_SYMBOL" for record in caplog.records)


def test_process_symbols_processes_when_backup_active(symbol_processing_env, monkeypatch, caplog):
    runtime, _state = symbol_processing_env
    runtime._data_degraded = False
    runtime._data_degraded_reason = None
    runtime._data_degraded_fatal = False

    monkeypatch.setattr(
        bot_engine,
        "_resolve_data_provider_degraded",
        lambda: (True, "using_backup_provider", False),
    )

    frame = pd.DataFrame(
        {
            "open": [2.0],
            "high": [2.0],
            "low": [2.0],
            "close": [2.0],
            "volume": [200],
        }
    )

    monkeypatch.setattr(
        bot_engine,
        "fetch_minute_df_safe",
        lambda _symbol: frame.copy(),
    )

    caplog.set_level(logging.INFO, logger=bot_engine.logger.name)

    processed, row_counts, _ = bot_engine._process_symbols(
        ["AAPL"],
        current_cash=50000.0,
        model=None,
        regime_ok=True,
    )

    assert processed == ["AAPL"]
    assert row_counts == {"AAPL": 1}
    assert not any(record.msg == "DATA_SOURCE_EMPTY" for record in caplog.records)
