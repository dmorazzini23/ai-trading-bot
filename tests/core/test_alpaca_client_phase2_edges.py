from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import ANY

import pytest

from ai_trading.core import alpaca_client


class _StubLogger:
    def __init__(self) -> None:
        self.warning_calls: list[tuple[Any, dict[str, Any]]] = []
        self.info_calls: list[tuple[Any, dict[str, Any]]] = []
        self.error_calls: list[tuple[Any, dict[str, Any]]] = []

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        del args
        self.warning_calls.append((message, kwargs))

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        del args
        self.info_calls.append((message, kwargs))

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        del args
        self.error_calls.append((message, kwargs))


@pytest.fixture
def stub_logger(monkeypatch: pytest.MonkeyPatch) -> _StubLogger:
    logger = _StubLogger()
    monkeypatch.setattr(alpaca_client, "_get_bot_logger_once", lambda: logger)
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    return logger


def test_validate_trading_api_handles_none_and_missing_methods(
    stub_logger: _StubLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(alpaca_client._alpaca_api, "ALPACA_AVAILABLE", False)

    assert alpaca_client._validate_trading_api(None) is False
    assert alpaca_client._validate_trading_api(object()) is False

    warning_keys = {kwargs.get("key") for _message, kwargs in stub_logger.warning_calls}
    error_keys = {kwargs.get("key") for _message, kwargs in stub_logger.error_calls}
    assert "alpaca_client_missing" in warning_keys
    assert "alpaca_cancel_order_missing" not in error_keys


def test_validate_trading_api_maps_get_order_and_cancel_batch(
    stub_logger: _StubLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch

    class Api:
        def get_orders(self, **_kwargs: Any) -> list[Any]:
            return []

        def get_all_positions(self) -> list[Any]:
            return []

        def get_order_by_id(self, order_id: str) -> tuple[str, str]:
            return ("got", order_id)

        def cancel_order_by_id(self, order_id: str) -> tuple[str, str]:
            return ("cancelled", order_id)

        def submit_order(self, order_data: Any) -> Any:
            return order_data

    api = Api()

    assert alpaca_client._validate_trading_api(api) is True
    assert not hasattr(api, "get_order")
    assert not hasattr(api, "cancel_order")
    warning_keys = {kwargs.get("key") for _message, kwargs in stub_logger.warning_calls}
    assert "alpaca_native_methods_missing" not in warning_keys


def test_validate_trading_api_list_orders_status_falls_back_when_filter_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch

    calls: list[dict[str, Any]] = []

    class Api:
        def get_orders(self, *args: Any, **kwargs: Any) -> list[str]:
            calls.append(dict(kwargs))
            if "filter" in kwargs:
                raise TypeError("filter unsupported")
            return ["order"]

        def get_all_positions(self) -> list[Any]:
            return []

        def get_order_by_id(self, order_id: str) -> str:
            return order_id

        def cancel_order_by_id(self, order_id: str) -> str:
            return order_id

        def submit_order(self, order_data: Any) -> Any:
            return order_data

    api = Api()

    assert alpaca_client._validate_trading_api(api) is True
    assert not hasattr(api, "list_orders")
    with pytest.raises(TypeError):
        alpaca_client.list_open_orders(api)
    assert calls == [{"filter": ANY}]


def test_list_open_orders_filters_active_statuses_after_empty_open_snapshot() -> None:
    class Status:
        def __init__(self, value: str) -> None:
            self.value = value

    orders = [
        SimpleNamespace(status=Status("accepted")),
        SimpleNamespace(status="filled"),
        SimpleNamespace(status="pending_cancel"),
        SimpleNamespace(status="canceled"),
    ]
    calls: list[str] = []

    class Api:
        def get_orders(self, *, filter: Any) -> list[Any]:
            status = getattr(getattr(filter, "status", None), "value", filter.status)
            calls.append(status)
            return [] if status == "open" else orders

    filtered = alpaca_client.list_open_orders(Api())

    assert calls == ["open", "all"]
    assert [order.status.value if hasattr(order.status, "value") else order.status for order in filtered] == [
        "accepted",
        "pending_cancel",
    ]


def test_ensure_alpaca_attached_sets_context_api_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = SimpleNamespace(
        get_orders=lambda **_kwargs: [],
        get_all_positions=lambda: [],
        get_order_by_id=lambda _order_id: None,
        cancel_order_by_id=lambda _order_id: None,
        submit_order=lambda _order_data: None,
    )
    bot_engine = ModuleType("ai_trading.core.bot_engine")
    bot_engine.trading_client = api  # type: ignore[attr-defined]
    bot_engine.data_client = None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", bot_engine)
    monkeypatch.setattr(alpaca_client, "_initialize_alpaca_clients", lambda: True)
    validated: list[Any] = []
    def fake_validate(candidate: Any) -> bool:
        validated.append(candidate)
        return True

    monkeypatch.setattr(alpaca_client, "_validate_trading_api", fake_validate)

    class Context:
        def __init__(self) -> None:
            self.api = None

    ctx = Context()

    alpaca_client.ensure_alpaca_attached(ctx)

    assert ctx.api is api
    assert validated == [api]


def test_initialize_clients_skips_disabled_execution_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    bot_engine = ModuleType("ai_trading.core.bot_engine")
    bot_engine.trading_client = None  # type: ignore[attr-defined]
    bot_engine.data_client = None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", bot_engine)
    monkeypatch.setenv("EXECUTION_MODE", "disabled")
    monkeypatch.setattr(alpaca_client, "ALPACA_AVAILABLE", True)
    monkeypatch.setattr(alpaca_client._alpaca_api, "ALPACA_AVAILABLE", True)

    assert alpaca_client._initialize_alpaca_clients() is False
    assert bot_engine.trading_client is None  # type: ignore[attr-defined]
    assert bot_engine.data_client is None  # type: ignore[attr-defined]
