import logging

from ai_trading import logging as app_logging


def test_log_fetch_attempt_info_is_rate_limited(caplog):
    app_logging.reset_rate_limit_tracker()
    caplog.set_level(logging.INFO, logger="ai_trading.logging")

    for _ in range(6):
        app_logging.log_fetch_attempt("alpaca", status=200, symbol="AAPL")

    info_logs = [
        record
        for record in caplog.records
        if record.name == "ai_trading.logging" and record.getMessage() == "FETCH_ATTEMPT"
    ]
    assert len(info_logs) == 1


def test_log_fetch_attempt_warning_is_rate_limited(caplog):
    app_logging.reset_rate_limit_tracker()
    caplog.set_level(logging.WARNING, logger="ai_trading.logging")

    for _ in range(5):
        app_logging.log_fetch_attempt("alpaca", status=429, error="rate_limited", symbol="AAPL")

    warn_logs = [
        record
        for record in caplog.records
        if record.name == "ai_trading.logging" and record.getMessage() == "FETCH_ATTEMPT"
    ]
    assert len(warn_logs) == 1
