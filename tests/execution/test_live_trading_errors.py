from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.execution import live_trading


def test_extract_api_error_metadata_handles_sparse_exception():
    class CustomError(Exception):
        pass

    err = CustomError("broken")

    metadata = live_trading._extract_api_error_metadata(err)

    assert metadata["detail"] == "broken"
    assert metadata["error_type"] == "CustomError"


def test_fallback_api_error_parses_json_payload_and_http_status():
    http_error = SimpleNamespace(response=SimpleNamespace(status_code=418))

    err = live_trading.APIError(
        '{"message": "teapot", "code": "short_and_stout"}',
        http_error=http_error,
    )

    assert err.message == "teapot"
    assert err.code == "short_and_stout"
    assert err.status_code == 418


def test_fallback_api_error_keeps_plain_message_when_metadata_is_unreadable():
    class BrokenHTTPError:
        @property
        def response(self):
            raise RuntimeError("response unavailable")

    err = live_trading.APIError(
        "plain broker failure",
        http_error=BrokenHTTPError(),
        code="plain",
        status_code=429,
    )

    assert err.message == "plain broker failure"
    assert err.code == "plain"
    assert err.status_code == 429


def test_lookup_and_duplicate_error_classifiers_use_metadata():
    missing = live_trading.APIError(
        "order disappeared",
        code="40410000",
        status_code=500,
    )
    duplicate = live_trading.APIError(
        "client_order_id must be unique",
        code="other",
        status_code=422,
    )
    wrong_status = live_trading.APIError(
        "client_order_id must be unique",
        status_code=400,
    )

    assert live_trading._is_missing_order_lookup_error(missing)
    assert live_trading._is_duplicate_client_order_id_error(duplicate)
    assert not live_trading._is_duplicate_client_order_id_error(wrong_status)


def test_retry_after_parser_accepts_seconds_dates_and_header_objects():
    future = datetime.now(UTC) + timedelta(seconds=60)

    class HeaderBag:
        def get(self, key):
            if key == "Retry-After":
                return "3.5"
            return None

    header_error = live_trading.APIError(
        "rate limited",
        http_error=SimpleNamespace(response=SimpleNamespace(headers=HeaderBag())),
    )

    assert live_trading._parse_retry_after_seconds("-2") == 0.0
    assert live_trading._parse_retry_after_seconds("nan") is None
    assert live_trading._parse_retry_after_seconds(future.strftime("%a, %d %b %Y %H:%M:%S GMT")) > 0.0
    assert live_trading._extract_retry_after_seconds(header_error) == pytest.approx(3.5)


def test_retry_after_header_get_failure_degrades_to_none():
    class BrokenHeaders:
        def get(self, _key):
            raise RuntimeError("header lookup failed")

    err = live_trading.APIError(
        "rate limited",
        http_error=SimpleNamespace(response=SimpleNamespace(headers=BrokenHeaders())),
    )

    assert live_trading._extract_retry_after_seconds(err) is None


@pytest.mark.parametrize(
    "detail, expected",
    [
        ("insufficient buying power", "insufficient_buying_power"),
        ("outside price band", "limit_up_down"),
        ("minimum price increment rejected", "price_increment"),
        ("market is closed", "market_closed"),
    ],
)
def test_classify_rejection_reason_common_broker_details(detail, expected):
    assert live_trading._classify_rejection_reason(detail) == expected


def test_market_retry_requires_fallback_price_and_price_like_metadata():
    metadata = {"detail": "limit price outside NBBO", "status_code": "422"}

    assert live_trading._should_retry_limit_as_market(metadata, using_fallback_price=True)
    assert not live_trading._should_retry_limit_as_market(metadata, using_fallback_price=False)
    assert live_trading._should_retry_limit_as_market(
        {"code": "40010003"},
        using_fallback_price=True,
    )


def test_submit_no_result_error_fingerprint_normalizes_details():
    assert live_trading._submit_no_result_error_fingerprint(
        reason="submit_no_result",
        detail="Missing_Client_Order_ID from broker",
        status_code="422",
    ) == "422:missing_client_order_id"
    assert live_trading._submit_no_result_error_fingerprint(
        reason="submit_no_result_timeout",
        detail="connection reset by peer",
    ) == "connection"
    assert live_trading._submit_no_result_error_fingerprint(reason="broker_reject") is None


def test_submit_limit_order_handles_timeout_without_unboundlocalerror(monkeypatch, caplog):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(live_trading, "get_alpaca_creds", lambda: ("key", "secret"))
    monkeypatch.setattr(live_trading, "get_tick_size", lambda symbol: Decimal("0.01"))
    monkeypatch.setattr(
        live_trading,
        "_call_preflight_capacity",
        lambda *args, **kwargs: SimpleNamespace(can_submit=True, suggested_qty=args[3]),
    )
    monkeypatch.setattr(live_trading, "_safe_mode_guard", lambda *args, **kwargs: False)
    monkeypatch.setattr(live_trading.ExecutionEngine, "_refresh_settings", lambda self: None)
    monkeypatch.setattr(live_trading.ExecutionEngine, "_get_account_snapshot", lambda self: {})
    engine: Any = live_trading.ExecutionEngine()
    engine.is_initialized = True
    engine.trading_client = object()
    engine._ensure_initialized = lambda: True
    engine._pre_execution_checks = lambda: True

    def _raise_timeout(*args, **kwargs):
        raise TimeoutError("provider timeout")

    engine._execute_with_retry = _raise_timeout

    with caplog.at_level("ERROR"):
        result = engine.submit_limit_order("AAPL", "buy", 1, limit_price=123.45)

    assert result is None
    assert any(record.message == "ORDER_SUBMIT_RETRIES_EXHAUSTED" for record in caplog.records)


def test_normalize_order_payload_preserves_fractional_fill_qty():
    payload = {"id": "order-1", "status": "partially_filled", "filled_qty": "0.6", "qty": "1"}

    _, status, filled_qty, requested_qty, order_id, client_order_id = live_trading._normalize_order_payload(
        payload,
        qty_fallback=1,
    )

    assert status == "partially_filled"
    assert filled_qty == pytest.approx(0.6)
    assert requested_qty == 1
    assert order_id == "order-1"
    assert client_order_id is None


def test_normalize_order_payload_preserves_fractional_requested_qty():
    payload = {
        "id": "order-2",
        "status": "accepted",
        "filled_qty": "0",
        "qty": "0.6",
    }

    _, status, filled_qty, requested_qty, order_id, client_order_id = live_trading._normalize_order_payload(
        payload,
        qty_fallback=1,
    )

    assert status == "accepted"
    assert filled_qty == pytest.approx(0.0)
    assert requested_qty == pytest.approx(0.6)
    assert order_id == "order-2"
    assert client_order_id is None
