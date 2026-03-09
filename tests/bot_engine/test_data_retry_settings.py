from __future__ import annotations

import logging

import pytest

from ai_trading.core import bot_engine


@pytest.fixture(autouse=True)
def _disable_shadow_snapshot_by_default(monkeypatch):
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_PRERANK_ENABLED", "1")


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


def test_pre_rank_execution_candidates_prefers_runtime_rank(monkeypatch):
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "2")
    runtime = type(
        "_Runtime",
        (),
        {
            "portfolio_weights": {"MSFT": 0.8, "AAPL": 0.1, "GOOG": 0.2},
            "execution_candidate_rank": {"MSFT": -5.0, "AAPL": 3.2, "GOOG": 2.1},
        },
    )()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG"],
        runtime=runtime,
    )

    assert ranked == ["AAPL", "GOOG"]


def test_pre_rank_execution_candidates_records_shadow_snapshot_when_enabled(monkeypatch):
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_EXEC_CANDIDATE_TOP_N", "2")
    payloads: list[dict] = []
    monkeypatch.setattr(
        bot_engine,
        "_record_shadow_prediction",
        lambda payload: payloads.append(dict(payload)),
    )
    runtime = type(
        "_Runtime",
        (),
        {
            "portfolio_weights": {"MSFT": 0.8, "AAPL": 0.1, "GOOG": 0.2},
            "execution_candidate_rank": {"MSFT": -5.0, "AAPL": 3.2, "GOOG": 2.1},
        },
    )()

    ranked = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "AAPL", "GOOG"],
        runtime=runtime,
    )

    assert ranked == ["AAPL", "GOOG"]
    assert payloads, "expected prerank shadow snapshot"
    latest = payloads[-1]
    assert latest["mode"] == "execution_candidate_prerank"
    assert latest["rank_source"] == "runtime_rank"
    assert latest["requested"] == 3
    assert latest["selected"] == 2
    assert latest["top_n"] == 2
    assert [entry["symbol"] for entry in latest["ranked"]] == ["AAPL", "GOOG"]


def test_merge_managed_position_symbols_includes_nonzero_positions() -> None:
    merged = bot_engine._merge_managed_position_symbols(
        ["AAPL", "ABBV"],
        {
            "AAPL": 10,
            "MSFT": 5,
            "TSLA": 0,
            "NVDA": float("nan"),
            "AMZN": "3",
        },
    )

    assert merged == ["AAPL", "ABBV", "MSFT", "AMZN"]


def test_load_tickers_falls_back_to_packaged_universe_when_path_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bot_engine, "load_universe", lambda: ["AAPL", "MSFT"])
    symbols = bot_engine.load_tickers("/tmp/does-not-exist.csv")
    assert symbols == ["AAPL", "MSFT"]
