from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock
import logging

from ai_trading.core.run_all_trades import run_all_trades


def test_api_exception_logs_warning(tmp_path, caplog):
    """API errors should be logged as warnings and skipped."""

    api_mock = Mock(side_effect=RuntimeError("boom"))

    caplog.set_level(logging.WARNING)
    log = tmp_path / "trades.csv"
    run_all_trades(["AAPL"], api_mock, trade_log=log)

    assert any("API_ERROR" in r.message for r in caplog.records)
    api_mock.assert_called_once_with("AAPL")
    assert log.exists()


def test_empty_symbols_calls_sleep_once(tmp_path):
    """When no symbols are supplied the sleep callback is invoked once."""

    sleep_mock = Mock()
    api_mock = Mock()
    log = tmp_path / "trades.csv"

    run_all_trades([], api_mock, trade_log=log, sleep=sleep_mock)

    sleep_mock.assert_called_once_with(1.0)
    api_mock.assert_not_called()
    assert log.exists()


def test_creates_trade_log(tmp_path):
    """A trade log CSV is created with the expected header."""

    api_mock = Mock(return_value={})
    log = tmp_path / "trades.csv"
    run_all_trades(["AAPL"], api_mock, trade_log=log)

    api_mock.assert_called_once_with("AAPL")
    assert log.exists()
    first_line = log.read_text().splitlines()[0]
    assert (
        first_line
        == "symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward"
    )
