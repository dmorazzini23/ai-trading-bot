from __future__ import annotations

from datetime import UTC, datetime, timedelta
from importlib import reload
from types import SimpleNamespace
from typing import Any

import pytest

import ai_trading.execution.live_trading as lt


class _ErrorWithMetadata(Exception):
    def __init__(self, message: str = "broker error", **attrs: Any) -> None:
        super().__init__(message)
        for key, value in attrs.items():
            setattr(self, key, value)


def test_backup_quote_acceptance_uses_fallback_age_and_gap_metadata() -> None:
    accepted, details = lt._maybe_accept_backup_quote(
        {
            "fallback_quote_age": "1.25",
            "fallback_quote_limit": "3",
            "fallback_quote_ok": True,
            "fallback_quote_timestamp": "2026-04-25T12:00:00+00:00",
            "gap_limit": "0.05",
        },
        provider_hint="yahoo",
        gap_ratio_value=0.01,
        min_quote_fresh_ms=2500.0,
        quote_age_ms=10_000.0,
        quote_timestamp_present=False,
    )

    assert accepted is True
    assert details["provider"] == "yahoo"
    assert details["age_ms"] == 1250.0
    assert details["age_limit_ms"] == 2500.0
    assert details["gap_ratio"] == 0.01
    assert details["quote_source"] == "backup"
    assert details["timestamp"] == datetime(2026, 4, 25, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("annotations", "provider_hint", "gap_ratio", "quote_age", "timestamp_present"),
    [
        (None, "yahoo", 0.01, 100.0, True),
        ({"fallback_quote_age": "1"}, "", 0.01, 100.0, True),
        ({"fallback_quote_age": "1"}, "alpaca_iex", 0.01, 100.0, True),
        ({"fallback_quote_age": "10", "fallback_quote_limit": "1"}, "yahoo", 0.01, 100.0, True),
        ({}, "yahoo", 0.01, 100.0, False),
        ({"fallback_quote_age": "1", "gap_limit": "0.02"}, "yahoo", 0.03, 100.0, True),
        ({"fallback_quote_age": "1", "fallback_quote_ok": False, "fallback_quote_error": "bad"}, "yahoo", 0.01, 100.0, True),
    ],
)
def test_backup_quote_acceptance_rejects_unhealthy_backup_context(
    annotations: dict[str, Any] | None,
    provider_hint: str,
    gap_ratio: float,
    quote_age: float,
    timestamp_present: bool,
) -> None:
    accepted, details = lt._maybe_accept_backup_quote(
        annotations,
        provider_hint=provider_hint,
        gap_ratio_value=gap_ratio,
        min_quote_fresh_ms=2000.0,
        quote_age_ms=quote_age,
        quote_timestamp_present=timestamp_present,
    )

    assert accepted is False
    assert details == {}


def test_order_payload_normalization_handles_dicts_and_objects() -> None:
    order_obj, status, filled, requested, order_id, client_id = lt._normalize_order_payload(
        {
            "order_id": "broker-1",
            "client_order_id": "client-1",
            "status": "accepted",
            "filled_qty": "2.5",
            "quantity": "10",
            "symbol": "AAPL",
            "side": "buy",
        },
        qty_fallback=7,
    )

    assert order_obj.id == "broker-1"
    assert order_obj.client_order_id == "client-1"
    assert status == "accepted"
    assert filled == 2.5
    assert requested == 10.0
    assert order_id == "broker-1"
    assert client_id == "client-1"

    payload = SimpleNamespace(
        client_order_id="client-2",
        status=None,
        filled_quantity="bad",
        requested_quantity=None,
    )
    order_obj, status, filled, requested, order_id, client_id = lt._normalize_order_payload(
        payload,
        qty_fallback=4,
    )

    assert order_obj is payload
    assert status == "submitted"
    assert filled == 0.0
    assert requested == 4.0
    assert order_id == "client-2"
    assert client_id == "client-2"


def test_error_metadata_extracts_detail_code_status_and_retry_after() -> None:
    future_retry = datetime.now(UTC) + timedelta(seconds=60)
    err = _ErrorWithMetadata(
        "request failed",
        detail="client_order_id already exists",
        code="409",
        response=SimpleNamespace(
            status_code=409,
            headers={"Retry-After": future_retry.strftime("%a, %d %b %Y %H:%M:%S GMT")},
        ),
    )

    metadata = lt._extract_api_error_metadata(err)

    assert metadata["detail"] == "client_order_id already exists"
    assert metadata["code"] == "409"
    assert metadata["status_code"] == 409
    assert metadata["error_type"] == "_ErrorWithMetadata"
    assert lt._is_duplicate_client_order_id_error(err) is True
    assert lt._extract_retry_after_seconds(err) == pytest.approx(60.0, abs=2.0)


def test_missing_order_and_retry_after_helpers_cover_text_and_numeric_paths() -> None:
    missing = _ErrorWithMetadata("order not found", status_code="404")
    retry = _ErrorWithMetadata("rate limited", response=SimpleNamespace(headers={"Retry-After": "2.5"}))

    assert lt._is_missing_order_lookup_error(missing) is True
    assert lt._is_duplicate_client_order_id_error(missing) is False
    assert lt._extract_retry_after_seconds(retry) == 2.5
    assert lt._parse_retry_after_seconds(float("inf")) is None
    assert lt._parse_retry_after_seconds("not-a-date") is None


def test_short_sale_precheck_blocks_account_and_asset_restrictions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lt, "_allow_shorts_configured", lambda: True)
    monkeypatch.setattr(lt, "_long_only_state", lambda: (False, None))

    ok, extras, reason = lt._short_sale_precheck(
        None,
        SimpleNamespace(get_asset=lambda symbol: SimpleNamespace(shortable=True, easy_to_borrow=True, marginable=True)),
        symbol="AAPL",
        side="sell",
        quantity=1,
        closing_position=False,
        account_snapshot=SimpleNamespace(shorting_enabled=False, margin_enabled=True),
    )

    assert ok is False
    assert extras is not None
    assert extras["reason"] == "account_shorting_disabled"
    assert reason == "long_only"

    ok, extras, reason = lt._short_sale_precheck(
        None,
        SimpleNamespace(get_asset=lambda symbol: SimpleNamespace(shortable=False, easy_to_borrow=True, marginable=True)),
        symbol="AAPL",
        side="sell",
        quantity=1,
        closing_position=False,
        account_snapshot=SimpleNamespace(shorting_enabled=True, margin_enabled=True),
    )

    assert ok is False
    assert extras is not None
    assert extras["reason"] == "asset_not_shortable"
    assert reason == "shortability"


def test_capacity_helper_adapts_account_kwarg_and_side_semantics() -> None:
    reload(lt)
    calls: list[tuple[Any, Any, Any, Any, Any, Any]] = []

    def _preflight(symbol: str, side: str, price: float, qty: int, broker: object, *, account: object | None = None) -> lt.CapacityCheck:
        calls.append((symbol, side, price, qty, broker, account))
        return lt.CapacityCheck(True, qty)

    broker = object()
    account = object()
    result = lt._call_preflight_capacity("AAPL", "buy", 100.0, 3, broker, account, _preflight)

    assert result == lt.CapacityCheck(True, 3)
    assert calls == [("AAPL", "buy", 100.0, 3, broker, account)]
    assert lt._capacity_precheck_side("sell", closing_position=False) == "sell_short"
    assert lt._capacity_precheck_side("sell", closing_position=True) == "sell"
    assert lt._order_consumes_capacity("sell") is False
    assert lt._order_consumes_capacity("sell_short") is True
    assert lt._is_capacity_exhaustion_reason("insufficient_day_trading_buying_power") is True
