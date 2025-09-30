"""Tests for the Alpaca trading client adapter integration."""

import os
from typing import Any

os.environ.setdefault("PYTEST_RUNNING", "1")

import pytest

import ai_trading.util.env_check as env_check

env_check.assert_dotenv_not_shadowed = lambda: None  # type: ignore[assignment]

from ai_trading.alpaca_api import TradingClientAdapter
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

    def list_orders(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def list_positions(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def cancel_order_by_id(self, order_id: Any) -> tuple[str, Any]:
        return ("cancelled", order_id)


class _GetOnlyTradingClient:
    """Expose only get_* methods to exercise shim injection logic."""

    def __init__(self) -> None:
        self.orders_called = 0
        self.positions_called = 0

    def get_orders(self, *args: Any, **kwargs: Any) -> list[str]:
        self.orders_called += 1
        return ["order"]

    def get_all_positions(self) -> list[str]:
        self.positions_called += 1
        return ["position"]


@pytest.fixture
def stub_logger(monkeypatch: pytest.MonkeyPatch) -> _StubLogger:
    logger = _StubLogger()
    monkeypatch.setattr(alpaca_client, "_get_bot_logger_once", lambda: logger)
    return logger


def test_validate_trading_api_uses_adapter_without_warning(stub_logger: _StubLogger, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrapped TradingClient instances should not trigger adapter warnings."""

    monkeypatch.setattr(alpaca_client, "ALPACA_AVAILABLE", True)
    monkeypatch.setattr(alpaca_client, "get_trading_client_cls", lambda: _FakeTradingClient)

    adapter = TradingClientAdapter(_FakeTradingClient())

    result = alpaca_client._validate_trading_api(adapter)

    assert result is True
    assert ("ALPACA_API_ADAPTER", {"key": "alpaca_api_adapter"}) not in stub_logger.warning_calls
    assert adapter.cancel_order("abc123") == ("cancelled", "abc123")


def test_validate_trading_api_injects_shims_for_get_only_client(
    stub_logger: _StubLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adapters with __slots__ accept shim injection without AttributeError."""

    monkeypatch.setattr(alpaca_client, "ALPACA_AVAILABLE", True)

    client = _GetOnlyTradingClient()
    adapter = TradingClientAdapter(client)

    result = alpaca_client._validate_trading_api(adapter)

    assert result is True

    # Shimmed methods are now exposed and delegate to the wrapped client.
    assert callable(getattr(adapter, "list_orders"))
    assert adapter.list_orders() == ["order"]
    assert adapter.list_positions() == ["position"]

    assert client.orders_called == 1
    assert client.positions_called == 1

    error_keys = {kwargs.get("key") for _, kwargs in stub_logger.error_calls}
    assert "alpaca_list_orders_patch_failed" not in error_keys
    assert "alpaca_list_positions_patch_failed" not in error_keys
