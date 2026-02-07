import types

import pytest

from ai_trading.data import fetch as data_fetcher


@pytest.mark.parametrize(
    "metadata,http_status,expected_reason",
    [
        ({"http_status": 429}, 429, "rate_limited"),
        ({"http_status": 401}, 401, "auth_error"),
        ({"http_status": 503}, 503, "server_error"),
        ({"http_status": 418}, 418, "bad_request"),
    ],
)
def test_classify_http_failure_reasons(metadata, http_status, expected_reason):
    reason, details = data_fetcher._classify_fallback_reason("ignored", metadata)
    assert reason == expected_reason
    assert details["http_status"] == http_status


def test_classify_gap_ratio_reason():
    metadata = {"gap_ratio": 0.12}
    reason, details = data_fetcher._classify_fallback_reason(None, metadata)
    assert reason == "gap_ratio_exceeded"
    assert details["gap_ratio"] == pytest.approx(0.12)
    assert details["gap_ratio_pct"] == pytest.approx(12.0)


def test_classify_timeout_reason_from_hint():
    reason, details = data_fetcher._classify_fallback_reason("timeout waiting for bars", {})
    assert reason == "timeout"
    assert "detail" in details or details == {}


def test_classify_empty_bars_reason():
    metadata = {"fallback_reason": "empty_bars"}
    reason, _ = data_fetcher._classify_fallback_reason(None, metadata)
    assert reason == "empty_bars"


def test_classify_backup_cooldown_reason():
    reason, _ = data_fetcher._classify_fallback_reason("backup_cooldown_active", {})
    assert reason == "backup_cooldown_active"
