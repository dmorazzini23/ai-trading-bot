from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

"""Canonical Alpaca client helpers shared by runtime modules."""

from typing import Any
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta
from importlib import import_module
from time import perf_counter
from types import ModuleType

from ai_trading.logging import get_logger, logger_once
import ai_trading.alpaca_api as _alpaca_api
from ai_trading.alpaca_api import (
    get_trading_client_cls,
    get_data_client_cls,
    get_api_error_cls,
    _set_alpaca_service_available,
)
from ai_trading.config.management import get_env, is_shadow_mode
from ai_trading.diagnostics.env_diag import log_env_diag
from ai_trading.exc import COMMON_EXC

logger = get_logger(__name__)

ALPACA_AVAILABLE = getattr(_alpaca_api, "ALPACA_AVAILABLE", False)


_shared_logger_once: Any | None = None


def _get_active_alpaca_api_module() -> ModuleType:
    """Return the currently active alpaca_api module object."""

    module = sys.modules.get("ai_trading.alpaca_api")
    if isinstance(module, ModuleType):
        return module
    return _alpaca_api


def _get_bot_engine_module() -> ModuleType:
    """Return the active bot_engine module, honouring monkeypatched stubs."""

    module = sys.modules.get("ai_trading.core.bot_engine")
    if isinstance(module, ModuleType):
        return module
    return import_module("ai_trading.core.bot_engine")


def _get_bot_logger_once() -> Any:
    """Return the shared logger_once exposed by bot_engine when available."""

    global _shared_logger_once
    try:
        bot_engine_module = _get_bot_engine_module()
    except COMMON_EXC:
        return _shared_logger_once or logger_once
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("BOT_ENGINE_MODULE_LOOKUP_FAILED", exc_info=True)
        return _shared_logger_once or logger_once

    shared = getattr(bot_engine_module, "logger_once", None)
    if shared is None:
        return _shared_logger_once or logger_once

    if _shared_logger_once is not shared:
        _shared_logger_once = shared
    return shared


def _validate_trading_api(api: Any) -> bool:
    """Ensure the trading API exposes the native alpaca-py TradingClient surface."""
    log_once = _get_bot_logger_once()

    if api is None:
        if _alpaca_api.ALPACA_AVAILABLE and not is_shadow_mode():
            log_once.error("ALPACA_CLIENT_MISSING", key="alpaca_client_missing")
        else:
            log_once.warning("ALPACA_CLIENT_MISSING", key="alpaca_client_missing")
        return False

    required_methods = (
        "get_orders",
        "get_all_positions",
        "get_order_by_id",
        "cancel_order_by_id",
        "submit_order",
    )
    missing_methods = [
        name for name in required_methods if not callable(getattr(api, name, None))
    ]
    if missing_methods:
        log_method = log_once.warning if (
            is_shadow_mode() or get_env("PYTEST_RUNNING", None, resolve_aliases=False)
        ) else log_once.error
        log_method(
            "ALPACA_NATIVE_METHODS_MISSING",
            key="alpaca_native_methods_missing",
            extra={"missing_methods": missing_methods},
        )
        if not is_shadow_mode() and not get_env(
            "PYTEST_RUNNING", None, resolve_aliases=False
        ):
            raise RuntimeError(
                "Alpaca client missing native alpaca-py methods: "
                + ", ".join(missing_methods)
            )
        return False

    try:
        TradingClient = get_trading_client_cls()
    except RuntimeError as exc:
        explicit_test_path = bool(get_env("PYTEST_RUNNING", None, resolve_aliases=False))
        if is_shadow_mode() or explicit_test_path:
            log_once.warning(
                "ALPACA_TRADING_CLIENT_CLASS_MISSING",
                key="alpaca_trading_client_class_missing",
                extra={"error": str(exc)},
            )
            return True
        log_once.error(
            "ALPACA_TRADING_CLIENT_CLASS_MISSING",
            key="alpaca_trading_client_class_missing",
            extra={"error": str(exc)},
        )
        return False
    if isinstance(api, TradingClient):
        pass
    elif not get_env("PYTEST_RUNNING", None, resolve_aliases=False):
        log_once.warning("ALPACA_CLIENT_NOT_NATIVE", key="alpaca_client_not_native")
    return True


def _orders_request(status: str, *, after: datetime | None = None) -> Any:
    """Build a native alpaca-py GetOrdersRequest for a query status."""

    try:
        from alpaca.common.enums import Sort
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        raise RuntimeError("alpaca-py trading order request classes unavailable") from exc

    status_value: Any = getattr(QueryOrderStatus, str(status).upper(), status)
    return GetOrdersRequest(
        status=status_value,
        limit=500,
        direction=Sort.ASC,
        after=after,
    )


def _get_orders_by_status(
    api: Any, status: str, *, after: datetime | None = None
) -> Any:
    get_orders = getattr(api, "get_orders", None)
    if not callable(get_orders):
        raise RuntimeError("Alpaca client missing native get_orders method")
    return get_orders(filter=_orders_request(status, after=after))


def list_open_orders(api: Any):
    """Return a complete active-order snapshot or raise on an incomplete page.

    The broker's ``open`` filter can omit early active statuses. Query ``all``
    even when ``open`` is nonempty, then page by submission time. A full page
    without an advancing timestamp cannot be safely treated as complete.
    """

    started_at = perf_counter()
    all_pages = 0
    orders_scanned = 0
    active_statuses = {
        "open",
        "new",
        "pending_new",
        "accepted",
        "accepted_for_bidding",
        "partially_filled",
        "pending_replace",
        "pending_cancel",
        "held",
    }

    filtered: list[Any] = []
    seen_ids: set[str] = set()

    def add_active(order: Any, *, from_open_query: bool) -> None:
        status_raw = order.get("status", "") if isinstance(order, Mapping) else getattr(order, "status", "")
        status_value = getattr(status_raw, "value", status_raw)
        try:
            status_text = str(status_value).strip().lower()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            status_text = ""
        if status_text not in active_statuses and not (from_open_query and not status_text):
            return
        order_id = order.get("id") if isinstance(order, Mapping) else getattr(order, "id", None)
        client_order_id = order.get("client_order_id") if isinstance(order, Mapping) else getattr(order, "client_order_id", None)
        keys = {
            f"id:{order_id}" if order_id not in (None, "") else "",
            f"client:{client_order_id}" if client_order_id not in (None, "") else "",
        } - {""}
        if seen_ids.intersection(keys):
            return
        seen_ids.update(keys)
        filtered.append(order)

    for order in _get_orders_by_status(api, "open") or []:
        add_active(order, from_open_query=True)

    after: datetime | None = None
    previous_last: datetime | None = None
    while True:
        page = list(_get_orders_by_status(api, "all", after=after) or [])
        all_pages += 1
        orders_scanned += len(page)
        if len(page) > 500:
            raise RuntimeError("Alpaca all-orders page exceeds requested limit")
        for order in page:
            add_active(order, from_open_query=False)
        if len(page) < 500:
            break
        last = (
            page[-1].get("submitted_at")
            if isinstance(page[-1], Mapping)
            else getattr(page[-1], "submitted_at", None)
        )
        if not isinstance(last, datetime) or last.tzinfo is None:
            raise RuntimeError("Cannot paginate Alpaca orders without submitted_at")
        if previous_last is not None and last <= previous_last:
            raise RuntimeError("Alpaca all-orders pagination did not advance")
        if any(
            (order.get("id") if isinstance(order, Mapping) else getattr(order, "id", None))
            in (None, "")
            for order in page
        ):
            raise RuntimeError("Cannot paginate Alpaca orders without broker IDs")
        previous_last = last
        # Overlap the boundary timestamp. A tied 500-order page fails closed
        # on the next iteration instead of silently dropping tied orders.
        after = last - timedelta(microseconds=1)
    elapsed_ms = round((perf_counter() - started_at) * 1000)
    if elapsed_ms >= 10_000:
        logger.warning(
            "BROKER_OPEN_ORDER_SNAPSHOT_SLOW",
            extra={
                "elapsed_ms": elapsed_ms,
                "all_pages": all_pages,
                "orders_scanned": orders_scanned,
                "active_orders": len(filtered),
            },
        )
    return filtered


def ensure_alpaca_attached(ctx) -> None:
    """Attach global trading client to the context if it's missing."""
    log_once = _get_bot_logger_once()
    if get_env("PYTEST_RUNNING", None, resolve_aliases=False) and not (
        get_env("ALPACA_API_KEY", None, resolve_aliases=False)
        and get_env("ALPACA_SECRET_KEY", None, resolve_aliases=False)
    ):
        raise RuntimeError("Missing Alpaca API credentials")
    if getattr(ctx, "api", None) is not None:
        return
    if not _alpaca_api.ALPACA_AVAILABLE:
        try:
            # Refresh TradingClient availability from the active module state.
            # This supports test stubs that provide trading-only classes.
            get_trading_client_cls()
        except COMMON_EXC:
            return
    if not _initialize_alpaca_clients():
        return
    # Keep bot_engine runtime state synchronized with the initialized client.
    be: Any = _get_bot_engine_module()
    api = getattr(be, "trading_client", None)
    if api is None:
        if _alpaca_api.ALPACA_AVAILABLE and not is_shadow_mode():
            log_once.error(
                "ALPACA_CLIENT_MISSING after initialization", key="alpaca_client_missing"
            )
        else:
            log_once.warning(
                "ALPACA_CLIENT_MISSING after initialization", key="alpaca_client_missing"
            )
        if not is_shadow_mode():
            raise RuntimeError("Alpaca client missing after initialization")
        return
    if hasattr(ctx, "_ensure_initialized"):
        try:
            ctx._ensure_initialized()  # type: ignore[attr-defined]
        except COMMON_EXC:
            pass
    try:
        setattr(ctx, "api", api)
    except COMMON_EXC:
        inner = getattr(ctx, "_context", None)
        if inner is not None and getattr(inner, "api", None) is None:
            try:
                setattr(inner, "api", api)
            except COMMON_EXC:
                pass
    api = getattr(ctx, "api", None)
    if api is None:
        log_once.error("FAILED_TO_ATTACH_ALPACA_CLIENT", key="alpaca_attach_failed")
        if not is_shadow_mode():
            raise RuntimeError("Failed to attach Alpaca client to context")
        return
    if not _validate_trading_api(api):
        return


def _initialize_alpaca_clients() -> bool:
    """Initialize Alpaca trading clients lazily to avoid import delays."""
    # Defer imports to avoid cycles
    import time

    be: Any = _get_bot_engine_module()

    log_once = _get_bot_logger_once()

    if getattr(be, "trading_client", None) is not None:
        return True
    alpaca_available = bool(
        getattr(_get_active_alpaca_api_module(), "ALPACA_AVAILABLE", False)
        or ALPACA_AVAILABLE
    )
    if not alpaca_available:
        try:
            # Re-evaluate availability lazily so trading-only stubs can initialize.
            get_trading_client_cls()
        except COMMON_EXC:
            pass
        alpaca_available = bool(
            getattr(_get_active_alpaca_api_module(), "ALPACA_AVAILABLE", False)
            or ALPACA_AVAILABLE
        )
    if not alpaca_available:
        be.trading_client = None
        be.data_client = None
        return False
    execution_mode = str(get_env("EXECUTION_MODE", "sim", cast=str) or "sim").lower()
    if execution_mode == "disabled":
        be.trading_client = None
        be.data_client = None
        _set_alpaca_service_available(False)
        logger.info(
            "ALPACA_CLIENT_INIT_SKIPPED",
            extra={"reason": "execution_mode_disabled", "execution_mode": execution_mode},
        )
        return False
    try:
        APIError = get_api_error_cls()
    except ImportError as e:  # pragma: no cover - defensive
        logger.error("ALPACA_CLIENT_IMPORT_FAILED", extra={"error": str(e)})
        be.trading_client = None
        be.data_client = None
        return False
    for attempt in (1, 2):
        try:
            from ai_trading.core.bot_engine import _ensure_alpaca_env_or_raise
            key, secret, base_url = _ensure_alpaca_env_or_raise()
        except COMMON_EXC as e:
            logger.error("ALPACA_ENV_RESOLUTION_FAILED", extra={"error": str(e)})
            if attempt == 1:
                try:
                    from ai_trading.config.management import reload_env
                    reload_env(override=False)
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    logger.debug("RELOAD_ENV_AFTER_RESOLUTION_FAILURE_FAILED", exc_info=True)
                continue
            log_once.error("ALPACA_CLIENT_INIT_FAILED - env", key="alpaca_client_init_failed")
            be.trading_client = None
            be.data_client = None
            raise
        if not (key and secret):
            logger.info("Shadow mode or missing credentials: skipping Alpaca client initialization")
            log_once.warning("ALPACA_INIT_SKIPPED - shadow mode or missing credentials", key="alpaca_init_skipped")
            _set_alpaca_service_available(False)
            return False
        try:
            trading_client_cls = get_trading_client_cls()
            stock_client_cls = get_data_client_cls()
        except COMMON_EXC as e:
            logger.error("ALPACA_CLIENT_IMPORT_FAILED", extra={"error": str(e)})
            if attempt == 1:
                time.sleep(1)
                continue
            log_once.error("ALPACA_CLIENT_INIT_FAILED - import", key="alpaca_client_init_failed")
            be.trading_client = None
            be.data_client = None
            return False
        try:
            be.trading_client = trading_client_cls(
                api_key=key,
                secret_key=secret,
                paper="paper" in str(base_url).lower(),
                url_override=base_url,
            )
            if not _validate_trading_api(be.trading_client):
                be.trading_client = None
                be.data_client = None
                return False
            be.data_client = stock_client_cls(api_key=key, secret_key=secret)
        except (APIError, TypeError, ValueError, OSError) as e:
            if is_shadow_mode() or get_env("PYTEST_RUNNING", None, resolve_aliases=False):
                logger.warning("ALPACA_CLIENT_INIT_FAILED", extra={"error": str(e)})
                log_once.warning("ALPACA_CLIENT_INIT_FAILED - client", key="alpaca_client_init_failed")
            else:
                logger.error("ALPACA_CLIENT_INIT_FAILED", extra={"error": str(e)})
                log_once.error("ALPACA_CLIENT_INIT_FAILED - client", key="alpaca_client_init_failed")
            be.trading_client = None
            be.data_client = None
            return False
        logger.info("ALPACA_CLIENT_INIT_SUCCESS")
        log_env_diag(logger, extra={"initialized": True})
        return True
    return False
