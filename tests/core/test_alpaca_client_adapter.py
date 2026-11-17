"""Tests for native Alpaca trading client integration."""

import os
import sys
from datetime import UTC, datetime, timedelta
from types import ModuleType, SimpleNamespace
from typing import Any, cast

os.environ.setdefault("PYTEST_RUNNING", "1")

import pytest

import ai_trading.util.env_check as env_check

env_check.assert_dotenv_not_shadowed = lambda: None  # type: ignore[assignment]

from ai_trading.config.management import reload_trading_config
from ai_trading.core import alpaca_client

class _StubLogger:
    def __init__(self) -> None:
        self.warning_calls: list[tuple[Any, dict[str, Any]]] = []
        self.info_calls: list[tuple[Any, dict[str, Any]]] = []
        self.error_calls: list[tuple[Any, dict[str, Any]]] = []

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self.warning_calls.append((message, kwargs))

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self.info_calls.append((message, kwargs))

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self.error_calls.append((message, kwargs))


class _FakeTradingClient:
    """Mimic the modern alpaca-py TradingClient surface."""

    def get_orders(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def get_all_positions(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def get_order_by_id(self, order_id: Any) -> tuple[str, Any]:
        return ("order", order_id)

    def cancel_order_by_id(self, order_id: Any) -> tuple[str, Any]:
        return ("cancelled", order_id)

    def submit_order(self, order_data: Any) -> Any:
        return order_data


@pytest.fixture
def stub_logger(monkeypatch: pytest.MonkeyPatch) -> _StubLogger:
    logger = _StubLogger()
    monkeypatch.setattr(alpaca_client, "_get_bot_logger_once", lambda: logger)
    return logger


def test_validate_trading_api_accepts_native_trading_client(
    stub_logger: _StubLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native TradingClient instances should validate without adapter warnings."""

    monkeypatch.setattr(alpaca_client, "ALPACA_AVAILABLE", True)
    monkeypatch.setattr(alpaca_client, "get_trading_client_cls", lambda: _FakeTradingClient)

    client = _FakeTradingClient()

    result = alpaca_client._validate_trading_api(client)

    assert result is True
    assert ("ALPACA_API_ADAPTER", {"key": "alpaca_api_adapter"}) not in stub_logger.warning_calls
    assert client.cancel_order_by_id("abc123") == ("cancelled", "abc123")


def test_validate_trading_api_does_not_inject_legacy_methods(
    stub_logger: _StubLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation checks native methods without mutating the client object."""

    monkeypatch.setattr(alpaca_client, "ALPACA_AVAILABLE", True)
    monkeypatch.setattr(alpaca_client, "get_trading_client_cls", lambda: _FakeTradingClient)

    client = _FakeTradingClient()
    result = alpaca_client._validate_trading_api(client)

    assert result is True
    assert not hasattr(client, "list_orders")
    assert not hasattr(client, "list_positions")
    assert not hasattr(client, "cancel_order")
    assert not hasattr(client, "get_order")

    error_keys = {kwargs.get("key") for _, kwargs in stub_logger.error_calls}
    assert "alpaca_list_orders_patch_failed" not in error_keys
    assert "alpaca_list_positions_patch_failed" not in error_keys


def test_validate_trading_api_fails_closed_when_class_resolution_fails(
    stub_logger: _StubLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production validation should not accept an unresolvable TradingClient class."""

    monkeypatch.setattr(alpaca_client, "ALPACA_AVAILABLE", True)
    monkeypatch.setattr(alpaca_client, "is_shadow_mode", lambda: False)

    def fake_get_env(name: str, default: Any = None, **_kwargs: Any) -> Any:
        if name == "PYTEST_RUNNING":
            return None
        return default

    monkeypatch.setattr(alpaca_client, "get_env", fake_get_env)
    monkeypatch.setattr(
        alpaca_client,
        "get_trading_client_cls",
        lambda: (_ for _ in ()).throw(RuntimeError("TradingClient unavailable")),
    )

    client = _FakeTradingClient()

    result = alpaca_client._validate_trading_api(client)

    assert result is False
    error_keys = {kwargs.get("key") for _, kwargs in stub_logger.error_calls}
    assert "alpaca_trading_client_class_missing" in error_keys






def test_list_open_orders_merges_pending_with_nonempty_open_query() -> None:
    class _Api:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_orders(self, *, filter: Any) -> list[Any]:
            status = getattr(getattr(filter, "status", None), "value", filter.status)
            self.calls.append(status)
            if status == "open":
                return [SimpleNamespace(id="o1", status="open")]
            return [
                SimpleNamespace(id="o1", status="open"),
                SimpleNamespace(id="p1", status="pending_new"),
            ]

    api = _Api()
    orders = alpaca_client.list_open_orders(api)

    assert [order.id for order in orders] == ["o1", "p1"]
    assert api.calls == ["open", "all"]


def test_list_open_orders_reports_slow_complete_scan_without_truncating(monkeypatch) -> None:
    class Api:
        def get_orders(self, *, filter: Any) -> list[Any]:
            status = getattr(filter.status, "value", filter.status)
            if status == "open":
                return []
            return [SimpleNamespace(id="pending-1", status="accepted")]

    observed: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(alpaca_client, "perf_counter", iter((0.0, 11.0)).__next__)
    monkeypatch.setattr(
        alpaca_client.logger,
        "warning",
        lambda event, **kwargs: observed.append((event, kwargs["extra"])),
    )

    assert [order.id for order in alpaca_client.list_open_orders(Api())] == ["pending-1"]
    assert observed == [
        (
            "BROKER_OPEN_ORDER_SNAPSHOT_SLOW",
            {
                "elapsed_ms": 11_000,
                "all_pages": 1,
                "orders_scanned": 1,
                "active_orders": 1,
                "open_pages": 1,
                "history_bounded": True,
            },
        )
    ]


def test_list_open_orders_falls_back_to_all_and_keeps_active_statuses() -> None:
    class _Status:
        def __init__(self, value: str) -> None:
            self.value = value

    class _Api:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_orders(self, *, filter: Any) -> list[Any]:
            status = getattr(getattr(filter, "status", None), "value", filter.status)
            self.calls.append(status)
            if status == "open":
                return []
            if status == "all":
                return [
                    SimpleNamespace(id="p1", status=_Status("pending_new")),
                    SimpleNamespace(id="n1", status="new"),
                    SimpleNamespace(id="f1", status="filled"),
                ]
            return []

    api = _Api()
    orders = alpaca_client.list_open_orders(api)

    assert [order.id for order in orders] == ["p1", "n1"]
    assert api.calls == ["open", "all"]


def test_list_open_orders_rejects_unsupported_all_query() -> None:
    class _Api:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_orders(self, *, filter: Any) -> list[Any]:
            status = getattr(getattr(filter, "status", None), "value", filter.status)
            self.calls.append(status)
            if status == "open":
                return []
            raise TypeError("status=all unsupported")

    api = _Api()
    with pytest.raises(TypeError, match="status=all unsupported"):
        alpaca_client.list_open_orders(api)
    assert api.calls == ["open", "all"]


def test_list_open_orders_paginates_past_fifty_recent_closed_orders() -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    history = [
        SimpleNamespace(
            id=f"order-{index}",
            status="accepted" if index == 500 else "filled",
            submitted_at=start + timedelta(seconds=index),
        )
        for index in range(501)
    ]
    calls: list[tuple[str, datetime | None, int | None]] = []

    class Api:
        def get_orders(self, *, filter: Any) -> list[Any]:
            status = getattr(filter.status, "value", filter.status)
            calls.append((status, filter.after, filter.limit))
            if status == "open":
                return []
            return [
                order for order in history
                if filter.after is None or order.submitted_at > filter.after
            ][: filter.limit]

    orders = alpaca_client.list_open_orders(Api())

    assert [order.id for order in orders] == ["order-500"]
    assert [status for status, _, _ in calls] == ["open", "all", "all"]
    assert all(limit == 500 for _, _, limit in calls)


def test_list_open_orders_rejects_unadvanceable_timestamp_page() -> None:
    submitted_at = datetime(2026, 9, 1, tzinfo=UTC)
    history = [
        SimpleNamespace(id=f"order-{index}", status="filled", submitted_at=submitted_at)
        for index in range(501)
    ]

    class Api:
        def get_orders(self, *, filter: Any) -> list[Any]:
            if getattr(filter.status, "value", filter.status) == "open":
                return []
            return [
                order for order in history
                if filter.after is None or order.submitted_at > filter.after
            ][: filter.limit]

    with pytest.raises(RuntimeError, match="pagination did not advance"):
        alpaca_client.list_open_orders(Api())


def test_get_orders_network_error_does_not_fallback_to_invalid_status_kwarg() -> None:
    class _NetworkError(RuntimeError):
        pass

    class _Api:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def get_orders(self, *args: Any, **kwargs: Any) -> list[Any]:
            self.calls.append(dict(kwargs))
            if "status" in kwargs:
                raise AssertionError("unexpected status kwarg fallback")
            raise _NetworkError("connection aborted")

        def get_all_positions(self) -> list[Any]:
            return []

        def cancel_order_by_id(self, order_id: Any) -> Any:
            return order_id

    api = _Api()
    with pytest.raises(_NetworkError):
        alpaca_client.list_open_orders(api)
    assert api.calls
