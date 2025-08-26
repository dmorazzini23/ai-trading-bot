"""Tests for contextual empty DataFrame logging."""

from __future__ import annotations

import logging
from pathlib import Path

from ai_trading.core import bot_engine


def _write_empty_csv(path: Path, header: list[str]) -> None:
    path.write_text(",".join(header) + "\n")


def test_parse_local_positions_debug(caplog, tmp_path, monkeypatch):
    """_parse_local_positions should log debug with file context."""

    trade_log = tmp_path / "trades.csv"
    _write_empty_csv(trade_log, ["symbol", "qty", "side", "exit_time"])
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(trade_log))

    with caplog.at_level(logging.DEBUG):
        bot_engine._parse_local_positions()

    assert any(
        r.levelno == logging.DEBUG
        and str(trade_log) in r.getMessage()
        for r in caplog.records
    )


def test_load_signal_weights_warning(caplog, tmp_path, monkeypatch):
    """load_signal_weights should warn when the CSV is empty."""

    weights_file = tmp_path / "weights.csv"
    _write_empty_csv(weights_file, ["signal_name", "weight"])
    monkeypatch.setattr(bot_engine, "SIGNAL_WEIGHTS_FILE", str(weights_file))

    manager = bot_engine.SignalManager()

    with caplog.at_level(logging.WARNING):
        manager.load_signal_weights()

    assert any(
        r.levelno == logging.WARNING
        and str(weights_file) in r.getMessage()
        for r in caplog.records
    )


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

    assert any(
        r.levelno == logging.WARNING
        and str(trade_log) in r.getMessage()
        for r in caplog.records
    )


def test_average_reward_debug(caplog, tmp_path, monkeypatch):
    """_average_reward should log debug with file context when empty."""

    reward_file = tmp_path / "rewards.csv"
    _write_empty_csv(reward_file, ["reward"])
    monkeypatch.setattr(bot_engine, "REWARD_LOG_FILE", str(reward_file))
    monkeypatch.setattr(bot_engine, "_is_market_open_now", lambda: True)

    with caplog.at_level(logging.DEBUG):
        bot_engine._average_reward()

    assert any(
        r.levelno == logging.DEBUG
        and str(reward_file) in r.getMessage()
        for r in caplog.records
    )

