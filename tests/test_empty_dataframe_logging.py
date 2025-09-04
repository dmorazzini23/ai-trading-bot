"""Tests for contextual empty DataFrame logging."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from ai_trading.core import bot_engine


def _write_empty_csv(path: Path, header: list[str]) -> None:
    path.write_text(",".join(header) + "\n")


def test_parse_local_positions_logs_info_on_empty(caplog, tmp_path, monkeypatch):
    """_parse_local_positions should log info when the log is empty."""

    trade_log = tmp_path / "trades.csv"
    _write_empty_csv(trade_log, ["symbol", "qty", "side", "exit_time"])
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(trade_log))
    monkeypatch.setattr(bot_engine, "_EMPTY_TRADE_LOG_INFO_EMITTED", False, raising=False)

    with caplog.at_level(logging.INFO):
        bot_engine._parse_local_positions()
        bot_engine._parse_local_positions()

    records = [r for r in caplog.records if r.levelno == logging.INFO and str(trade_log) in r.getMessage()]
    assert len(records) == 1


def test_parse_local_positions_warns_when_missing(caplog, tmp_path, monkeypatch):
    """_parse_local_positions should warn when the log file is missing."""

    trade_log = tmp_path / "trades.csv"
    # Intentionally do not create the file
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(trade_log))
    monkeypatch.setattr(bot_engine, "_EMPTY_TRADE_LOG_INFO_EMITTED", False, raising=False)

    with caplog.at_level(logging.WARNING):
        bot_engine._parse_local_positions()

    assert any(r.levelno == logging.WARNING and str(trade_log) in r.getMessage() for r in caplog.records)


def test_load_signal_weights_warning(caplog, tmp_path, monkeypatch):
    """load_signal_weights should warn when the CSV is empty."""

    weights_file = tmp_path / "weights.csv"
    _write_empty_csv(weights_file, ["signal_name", "weight"])
    monkeypatch.setattr(bot_engine, "SIGNAL_WEIGHTS_FILE", str(weights_file))

    manager = bot_engine.SignalManager()

    with caplog.at_level(logging.WARNING):
        manager.load_signal_weights()

    assert any(r.levelno == logging.WARNING and str(weights_file) in r.getMessage() for r in caplog.records)


def test_meta_learning_weight_optimizer_warning(caplog, tmp_path):
    """Meta-learning optimizer should warn on empty trade log."""

    trade_log = tmp_path / "trades.csv"
    _write_empty_csv(
        trade_log,
        ["entry_price", "exit_price", "signal_tags", "side", "confidence"],
    )

    with caplog.at_level(logging.WARNING):
        bot_engine.run_meta_learning_weight_optimizer(
            trade_log_path=str(trade_log),
            output_path=str(tmp_path / "out.csv"),
        )

    assert any(r.levelno == logging.WARNING and str(trade_log) in r.getMessage() for r in caplog.records)


def test_average_reward_debug(caplog, tmp_path, monkeypatch):
    """_average_reward should log debug with file context when empty."""

    reward_file = tmp_path / "rewards.csv"
    _write_empty_csv(reward_file, ["reward"])
    monkeypatch.setattr(bot_engine, "REWARD_LOG_FILE", str(reward_file))
    monkeypatch.setattr(bot_engine, "_is_market_open_now", lambda: True)

    with caplog.at_level(logging.DEBUG):
        bot_engine._average_reward()

    assert any(r.levelno == logging.DEBUG and str(reward_file) in r.getMessage() for r in caplog.records)


def test_too_correlated_uses_cached_trade_log(monkeypatch):
    calls: list[None] = []

    def fake_read(*_a, **_k):
        calls.append(None)
        import pandas as pd  # type: ignore

        return pd.DataFrame(columns=["symbol", "exit_time"])

    monkeypatch.setattr(bot_engine, "_read_trade_log", fake_read)
    monkeypatch.setattr(bot_engine, "_TRADE_LOG_CACHE", None, raising=False)
    monkeypatch.setattr(bot_engine, "_TRADE_LOG_CACHE_LOADED", False, raising=False)

    bot_engine._load_trade_log_cache()

    ctx = SimpleNamespace(data_fetcher=SimpleNamespace(get_daily_df=lambda *_: None))
    bot_engine.too_correlated(ctx, "AAA")
    bot_engine.too_correlated(ctx, "BBB")

    assert len(calls) == 1
