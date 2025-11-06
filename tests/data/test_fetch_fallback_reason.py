import pytest

from ai_trading.data.fetch import _build_backup_usage_extra


def test_timeout_reason_includes_detail() -> None:
    payload = _build_backup_usage_extra(
        "yahoo",
        "aapl",
        "1Min",
        "request timeout",
        {"error_type": "TimeoutError", "error": "timeout"},
    )
    assert payload["reason"] == "timeout"
    assert payload["provider"] == "yahoo"
    assert payload["symbol"] == "AAPL"
    assert payload["error_type"] == "TimeoutError"


def test_rate_limit_reason_from_status_code() -> None:
    payload = _build_backup_usage_extra(
        "finnhub",
        "msft",
        "1Min",
        None,
        {"status_code": 429},
    )
    assert payload["reason"] == "rate_limited"
    assert payload["http_status"] == 429


def test_gap_ratio_reason_detected() -> None:
    payload = _build_backup_usage_extra(
        "alpaca_sip",
        "tsla",
        "1Min",
        None,
        {"gap_ratio": 0.25},
    )
    assert payload["reason"] == "gap_ratio_exceeded"
    assert payload["gap_ratio"] == pytest.approx(0.25)


def test_timestamp_missing_reason_from_hint() -> None:
    payload = _build_backup_usage_extra(
        "alpaca",
        "spy",
        "1Min",
        "Quote timestamp missing",
        {},
    )
    assert payload["reason"] == "quote_timestamp_missing"
    assert payload["detail"] == "Quote timestamp missing"

