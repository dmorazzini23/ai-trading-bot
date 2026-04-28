"""Tests for native Alpaca trading client integration."""

import os
import sys
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

        def get_order_by_id(self, order_id: Any) -> dict[str, Any]:
            return {"id": order_id}

        def cancel_order_by_id(self, order_id: Any) -> dict[str, Any]:
            return {"id": order_id}

        def submit_order(self, order_data: Any) -> Any:
            return order_data

    class _ProdDataClient:
        def __init__(self, *_, **__):
            self.initialized = True

    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("EXECUTION_MODE", "live")

    reload_trading_config()

    monkeypatch.setattr(alpaca_client, "ALPACA_AVAILABLE", True)
    monkeypatch.setattr(alpaca_client, "get_trading_client_cls", lambda: _ProdTradingClient)
    monkeypatch.setattr(alpaca_client, "get_data_client_cls", lambda: _ProdDataClient)
    monkeypatch.setattr(alpaca_client, "get_api_error_cls", lambda: Exception)

    stub_be = cast(Any, ModuleType("ai_trading.core.bot_engine"))
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
        assert isinstance(trading_client, _ProdTradingClient)
        assert not hasattr(trading_client, "list_orders")
        assert not hasattr(trading_client, "list_positions")
        assert trading_client.get_all_positions() == ["position"]

        result = alpaca_client._validate_trading_api(trading_client)
        assert result is True
        assert ("ALPACA_API_ADAPTER", {"key": "alpaca_api_adapter"}) not in stub_logger.warning_calls
    finally:
        monkeypatch.delenv("APP_ENV", raising=False)
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
        monkeypatch.delenv("EXECUTION_MODE", raising=False)
        reload_trading_config()
        stub_be.trading_client = None
        stub_be.data_client = None


def test_initialize_uses_active_bot_engine_stub_after_real_import(
    stub_logger: _StubLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initialization should target the active bot_engine module even after imports."""

    class _StubTradingClient:
        def __init__(self, *_, **__):
            self._orders: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def get_orders(self, *args: Any, **kwargs: Any) -> list[str]:
            self._orders.append((args, dict(kwargs)))
            return ["order"]

        def get_all_positions(self) -> list[str]:
            return ["position"]

        def get_order_by_id(self, order_id: Any) -> dict[str, Any]:
            return {"id": order_id}

        def cancel_order_by_id(self, order_id: Any) -> dict[str, Any]:
            return {"id": order_id}

        def submit_order(self, order_data: Any) -> Any:
            return order_data

    class _StubDataClient:
        def __init__(self, *_, **__):
            self.initialized = True

    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("ALPACA_API_KEY", "stub-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "stub-secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("EXECUTION_MODE", "live")

    reload_trading_config()

    import ai_trading.core as core_pkg

    original_bot_engine = sys.modules.get("ai_trading.core.bot_engine")

    real_bot_engine = cast(Any, ModuleType("ai_trading.core.bot_engine"))
    real_bot_engine.trading_client = "real-client"  # type: ignore[attr-defined]
    real_bot_engine.data_client = "real-data"  # type: ignore[attr-defined]
    real_bot_engine.logger_once = stub_logger

    sys.modules["ai_trading.core.bot_engine"] = real_bot_engine
    monkeypatch.setattr(core_pkg, "bot_engine", real_bot_engine, raising=False)

    monkeypatch.setattr(alpaca_client, "ALPACA_AVAILABLE", True)
    monkeypatch.setattr(alpaca_client, "get_trading_client_cls", lambda: _StubTradingClient)
    monkeypatch.setattr(alpaca_client, "get_data_client_cls", lambda: _StubDataClient)
    monkeypatch.setattr(alpaca_client, "get_api_error_cls", lambda: Exception)
    monkeypatch.setattr(alpaca_client, "_shared_logger_once", None)

    stub_be = cast(Any, ModuleType("ai_trading.core.bot_engine"))
    stub_be.trading_client = None  # type: ignore[attr-defined]
    stub_be.data_client = None  # type: ignore[attr-defined]
    stub_be._ensure_alpaca_env_or_raise = lambda: (
        "stub-key",
        "stub-secret",
        "https://api.alpaca.markets",
    )
    stub_be._alpaca_diag_info = lambda: {"env": "prod"}
    stub_be.logger_once = stub_logger

    try:
        sys.modules["ai_trading.core.bot_engine"] = stub_be

        initialised = alpaca_client._initialize_alpaca_clients()

        assert initialised is True

        trading_client = getattr(stub_be, "trading_client", None)
        data_client = getattr(stub_be, "data_client", None)

        assert isinstance(trading_client, _StubTradingClient)
        assert isinstance(data_client, _StubDataClient)
        assert trading_client.get_orders() == ["order"]
        assert trading_client.get_all_positions() == ["position"]
        assert real_bot_engine.trading_client == "real-client"  # type: ignore[attr-defined]
        assert real_bot_engine.data_client == "real-data"  # type: ignore[attr-defined]
    finally:
        monkeypatch.delenv("APP_ENV", raising=False)
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
        monkeypatch.delenv("EXECUTION_MODE", raising=False)
        reload_trading_config()
        if original_bot_engine is None:
            sys.modules.pop("ai_trading.core.bot_engine", None)
        else:
            sys.modules["ai_trading.core.bot_engine"] = original_bot_engine


def test_list_open_orders_uses_open_query_when_available() -> None:
    class _Api:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_orders(self, *, filter: Any) -> list[Any]:
            status = getattr(getattr(filter, "status", None), "value", filter.status)
            self.calls.append(status)
            if status == "open":
                return [SimpleNamespace(id="o1", status="open")]
            return []

    api = _Api()
    orders = alpaca_client.list_open_orders(api)

    assert len(orders) == 1
    assert orders[0].id == "o1"
    assert api.calls == ["open"]


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


def test_list_open_orders_returns_open_when_all_not_supported() -> None:
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
    orders = alpaca_client.list_open_orders(api)

    assert orders == []
    assert api.calls == ["open", "all"]


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
