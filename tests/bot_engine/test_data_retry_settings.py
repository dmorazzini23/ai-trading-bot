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


def test_data_retry_settings_flatten_mode(monkeypatch):
    monkeypatch.setenv("DATA_SOURCE_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("DATA_SOURCE_RETRY_DELAY_SECONDS", "2.5")
    monkeypatch.setenv("AI_TRADING_DATA_RETRY_FLATTEN_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DATA_RETRY_FLATTEN_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("AI_TRADING_DATA_RETRY_FLATTEN_MAX_DELAY_SECONDS", "0")
    monkeypatch.setattr(bot_engine, "_DATA_RETRY_SETTINGS_CACHE", None, raising=False)
    monkeypatch.setattr(bot_engine, "_DATA_RETRY_SETTINGS_LOGGED", False, raising=False)

    attempts, delay = bot_engine._resolve_data_retry_settings()

    assert attempts == 1
    assert delay == 0.0


def test_pre_rank_execution_candidates_preserves_input_order_without_weights(monkeypatch):
    monkeypatch.delenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", raising=False)

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG"],
        runtime=None,
    )

    assert ranked == ["MSFT", "AAPL", "GOOG"]


def test_pre_rank_execution_candidates_dedupes_symbols(monkeypatch):
    monkeypatch.delenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", raising=False)

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "MSFT", "aapl", "GOOG"],
        runtime=None,
    )

    assert ranked == ["MSFT", "AAPL", "GOOG"]


def test_pre_rank_execution_candidates_uses_weights_with_top_n(monkeypatch):
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "2")
    runtime = type(
        "_Runtime",
        (),
        {"portfolio_weights": {"MSFT": 0.2, "AAPL": 0.4, "GOOG": 0.9}},
    )()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG"],
        runtime=runtime,
    )

    assert ranked == ["GOOG", "AAPL"]
