from __future__ import annotations

"""Alpaca helpers moved from core.bot_engine with façade re-export.

The canonical implementations live here. core.bot_engine imports these names
so existing call sites continue to work. This reduces size and coupling of
bot_engine while keeping runtime behavior identical.
"""

from typing import Any

from ai_trading.logging import get_logger, logger_once
from ai_trading.alpaca_api import (
    ALPACA_AVAILABLE,
    get_trading_client_cls,
    get_data_client_cls,
    get_api_error_cls,
)
from ai_trading.config.management import is_shadow_mode
from ai_trading.exc import COMMON_EXC

logger = get_logger(__name__)


def _validate_trading_api(api: Any) -> bool:
    """Ensure trading api exposes required methods across SDK variants.

    - Guarantees `list_orders` exists, mapping to `get_orders(...)` if needed.
    - Guarantees `list_positions` exists, mapping to `get_all_positions()` if needed.
    """
    if api is None:
        logger_once.error("ALPACA_CLIENT_MISSING", key="alpaca_client_missing")
        return False
    if not hasattr(api, "list_orders"):
        if hasattr(api, "get_orders"):
            import inspect

            try:
                sig = inspect.signature(api.get_orders)  # type: ignore[attr-defined]
            except (TypeError, ValueError):  # pragma: no cover - defensive
                sig = None
            accepts_status = bool(sig and "status" in sig.parameters)
            accepts_filter = bool(sig and "filter" in sig.parameters)

            def _list_orders_wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
                status = kwargs.pop("status", None)
                if status is None:
                    return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]

                enum_val: Any = status
                try:  # optional import paths for SDK enums
                    enums_mod = __import__("alpaca.trading.enums", fromlist=[""])
                    enum_cls = getattr(enums_mod, "QueryOrderStatus", None) or getattr(
                        enums_mod, "OrderStatus", None
                    )
                    if enum_cls is not None:
                        enum_val = getattr(enum_cls, str(status).upper(), status)
                except Exception:
                    pass

                # Prefer building a filter object when available to be compatible
                # with newer alpaca-py signatures and tests that assert this path.
                try:
                    requests_mod = __import__(
                        "alpaca.trading.requests", fromlist=["GetOrdersRequest"]
                    )
                    GetOrdersRequest = getattr(requests_mod, "GetOrdersRequest")
                    filter_obj = GetOrdersRequest(statuses=[enum_val])
                    return api.get_orders(
                        *args, filter=filter_obj, **kwargs
                    )  # type: ignore[attr-defined]
                except Exception:
                    if accepts_status:
                        kwargs["status"] = enum_val
                        return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]
                    # Last resort: pass through status kwarg
                    kwargs["status"] = enum_val
                    return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]

            setattr(api, "list_orders", _list_orders_wrapper)  # type: ignore[attr-defined]
            logger_once.info(
                "API_GET_ORDERS_MAPPED", key="alpaca_get_orders_mapped"
            )
        else:
            logger_once.error("ALPACA_LIST_ORDERS_MISSING", key="alpaca_list_orders_missing")
            if not is_shadow_mode():
                raise RuntimeError("Alpaca client missing list_orders method")
            return False

    # Map positions accessor for alpaca-py `TradingClient`
    if not hasattr(api, "list_positions") and hasattr(api, "get_all_positions"):
        try:
            def _list_positions_wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
                # TradingClient.get_all_positions() takes no filters; ignore args
                return api.get_all_positions()  # type: ignore[attr-defined]

            setattr(api, "list_positions", _list_positions_wrapper)  # type: ignore[attr-defined]
            logger_once.info(
                "API_GET_POSITIONS_MAPPED", key="alpaca_get_positions_mapped"
            )
        except Exception:
            # Non-fatal; caller may handle attribute absence
            pass
    TradingClient = get_trading_client_cls()
    if not isinstance(api, TradingClient):
        logger_once.warning("ALPACA_API_ADAPTER", key="alpaca_api_adapter")
    return True


def list_open_orders(api: Any):
    """Return open orders from api, compatible across SDK versions."""
    return api.list_orders(status="open")


def ensure_alpaca_attached(ctx) -> None:
    """Attach global trading client to the context if it's missing."""
    if getattr(ctx, "api", None) is not None:
        return
    if not ALPACA_AVAILABLE:
        return
    if not _initialize_alpaca_clients():
        return
    # Mirror the global singleton in bot_engine for compatibility
    import ai_trading.core.bot_engine as be
    api = getattr(be, "trading_client", None)
    if api is None:
        logger_once.error("ALPACA_CLIENT_MISSING after initialization", key="alpaca_client_missing")
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
        logger_once.error("FAILED_TO_ATTACH_ALPACA_CLIENT", key="alpaca_attach_failed")
        if not is_shadow_mode():
            raise RuntimeError("Failed to attach Alpaca client to context")
        return
    _validate_trading_api(api)


def _initialize_alpaca_clients() -> bool:
    """Initialize Alpaca trading clients lazily to avoid import delays."""
    # Defer imports to avoid cycles
    import time
    import ai_trading.core.bot_engine as be

    if getattr(be, "trading_client", None) is not None:
        return True
    if not ALPACA_AVAILABLE:
        be.trading_client = None
        be.data_client = None
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
                except Exception:
                    pass
                continue
            logger_once.error("ALPACA_CLIENT_INIT_FAILED - env", key="alpaca_client_init_failed")
            be.trading_client = None
            be.data_client = None
            raise
        if not (key and secret):
            diag = {}
            try:
                from ai_trading.core.bot_engine import _alpaca_diag_info
                diag = _alpaca_diag_info()
            except Exception:
                pass
            logger.info("Shadow mode or missing credentials: skipping Alpaca client initialization")
            logger_once.warning("ALPACA_INIT_SKIPPED - shadow mode or missing credentials", key="alpaca_init_skipped")
            logger.info("ALPACA_DIAG", extra=diag)
            return False
        try:
            AlpacaREST = get_trading_client_cls()
            stock_client_cls = get_data_client_cls()
        except COMMON_EXC as e:
            logger.error("ALPACA_CLIENT_IMPORT_FAILED", extra={"error": str(e)})
            if attempt == 1:
                time.sleep(1)
                continue
            logger_once.error("ALPACA_CLIENT_INIT_FAILED - import", key="alpaca_client_init_failed")
            be.trading_client = None
            be.data_client = None
            return False
        try:
            be.trading_client = AlpacaREST(api_key=key, secret_key=secret, url_override=base_url)
            be.data_client = stock_client_cls(api_key=key, secret_key=secret)
        except (APIError, TypeError, ValueError, OSError) as e:
            logger.error("ALPACA_CLIENT_INIT_FAILED", extra={"error": str(e)})
            logger_once.error("ALPACA_CLIENT_INIT_FAILED - client", key="alpaca_client_init_failed")
            be.trading_client = None
            be.data_client = None
            return False
        try:
            from ai_trading.core.bot_engine import _alpaca_diag_info
            logger.info("ALPACA_DIAG", extra={"initialized": True, **_alpaca_diag_info()})
        except Exception:
            pass
        return True
    return False
