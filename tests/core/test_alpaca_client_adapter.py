"""Tests for the Alpaca trading client adapter integration."""

import os
import sys
from types import ModuleType
from typing import Any

os.environ.setdefault("PYTEST_RUNNING", "1")

import pytest

import ai_trading.util.env_check as env_check

env_check.assert_dotenv_not_shadowed = lambda: None  # type: ignore[assignment]

from ai_trading.alpaca_api import TradingClientAdapter
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


def test_initialize_production_config_avoids_adapter_warning(
    stub_logger: _StubLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production configuration should initialise native TradingClient without warnings."""

    class _ProdTradingClient:
        __module__ = "alpaca.trading.client"

        def __init__(self, *_, **__):
            self._orders: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def get_orders(self, *args: Any, **kwargs: Any) -> list[str]:
            self._orders.append((args, dict(kwargs)))
            return ["order"]

        def get_all_positions(self) -> list[str]:
            return ["position"]

        def cancel_order_by_id(self, order_id: Any) -> dict[str, Any]:
            return {"id": order_id}

    class _ProdDataClient:
        def __init__(self, *_, **__):
            self.initialized = True

    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_API_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("EXECUTION_MODE", "live")

    reload_trading_config()

    monkeypatch.setattr(alpaca_client, "ALPACA_AVAILABLE", True)
    monkeypatch.setattr(alpaca_client, "get_trading_client_cls", lambda: _ProdTradingClient)
    monkeypatch.setattr(alpaca_client, "get_data_client_cls", lambda: _ProdDataClient)
    monkeypatch.setattr(alpaca_client, "get_api_error_cls", lambda: Exception)

    stub_be = ModuleType("ai_trading.core.bot_engine")
    stub_be.trading_client = None
    stub_be.data_client = None
    stub_be._ensure_alpaca_env_or_raise = lambda: (
        "test-key",
        "test-secret",
        "https://api.alpaca.markets",
    )
    stub_be._alpaca_diag_info = lambda: {"env": "prod"}

    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", stub_be)

    try:
        initialised = alpaca_client._initialize_alpaca_clients()
        assert initialised is True

        trading_client = getattr(stub_be, "trading_client", None)
        assert isinstance(trading_client, TradingClientAdapter)
        assert trading_client.list_orders(status="open") == ["order"]
        assert trading_client.list_positions() == ["position"]

        result = alpaca_client._validate_trading_api(trading_client)
        assert result is True
        assert ("ALPACA_API_ADAPTER", {"key": "alpaca_api_adapter"}) not in stub_logger.warning_calls
    finally:
        monkeypatch.delenv("APP_ENV", raising=False)
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        monkeypatch.delenv("ALPACA_API_URL", raising=False)
        monkeypatch.delenv("EXECUTION_MODE", raising=False)
        reload_trading_config()
        stub_be.trading_client = None
        stub_be.data_client = None
