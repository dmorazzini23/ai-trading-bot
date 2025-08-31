from __future__ import annotations

"""Alpaca helper implementations decoupled from bot_engine.

Provides canonical implementations for client validation, attaching the
client to a context, and lazy initialization. `bot_engine` delegates to
these to reduce size and coupling.
"""

from typing import Any

from ai_trading.logging import get_logger, logger_once
from ai_trading.alpaca_api import get_trading_client_cls, get_data_client_cls, get_api_error_cls
from ai_trading.config.management import is_shadow_mode, reload_env
from ai_trading.exc import COMMON_EXC

logger = get_logger(__name__)


def _validate_trading_api(api: Any) -> bool:
    """Ensure trading api exposes list_orders; adapt legacy get_orders if present."""
    if api is None:
        logger_once.error("ALPACA_CLIENT_MISSING", key="alpaca_client_missing")
        return False
    if not hasattr(api, "list_orders"):
        if hasattr(api, "get_orders"):
            def _list_orders_wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
                status = kwargs.pop("status", None)
                if status is None:
                    return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]
                import inspect
                params = inspect.signature(api.get_orders).parameters
                enum_val: Any = status
                try:
                    enums_mod = __import__("alpaca.trading.enums", fromlist=[""])
                    enum_cls = getattr(enums_mod, "QueryOrderStatus", None) or getattr(enums_mod, "OrderStatus", None)
                    if enum_cls is not None:
                        enum_val = getattr(enum_cls, str(status).upper(), status)
                except Exception:
                    pass
                if "status" in params:
                    kwargs["status"] = enum_val
                    return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]
                req = None
                try:
                    requests_mod = __import__("alpaca.trading.requests", fromlist=[""])
                    enums_mod = __import__("alpaca.trading.enums", fromlist=[""])
                    enum_cls = getattr(enums_mod, "QueryOrderStatus", None) or getattr(enums_mod, "OrderStatus", None)
                    if enum_cls is not None:
                        enum_val = getattr(enum_cls, str(status).upper(), status)
                    req_cls = getattr(requests_mod, "GetOrdersRequest")
                    req = req_cls(statuses=[enum_val])
                except Exception:
                    pass
                if req is not None:
                    try:
                        return api.get_orders(*args, filter=req, **kwargs)  # type: ignore[attr-defined]
                    except TypeError:
                        return api.get_orders(req, *args, **kwargs)  # type: ignore[attr-defined]
                kwargs["status"] = status
                return api.get_orders(*args, **kwargs)  # type: ignore[attr-defined]
            setattr(api, "list_orders", _list_orders_wrapper)  # type: ignore[attr-defined]
            logger_once.warning("API_GET_ORDERS_MAPPED", key="alpaca_get_orders_mapped")
        else:
            logger_once.error("ALPACA_LIST_ORDERS_MISSING", key="alpaca_list_orders_missing")
            if not is_shadow_mode():
                raise RuntimeError("Alpaca client missing list_orders method")
            return False
    TradingClient = get_trading_client_cls()
    if not isinstance(api, TradingClient):
        logger_once.warning("ALPACA_API_ADAPTER", key="alpaca_api_adapter")
    return True


def list_open_orders(api: Any):
    return api.list_orders(status="open")


def ensure_alpaca_attached(ctx) -> None:
    """Attach global trading client to the context if it's missing."""
    if getattr(ctx, "api", None) is not None:
        return
    if not _initialize_alpaca_clients():
        return
    # Pull singleton from bot_engine to avoid duplication
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
    """Initialize Alpaca trading clients lazily; return True on success."""
    import time
    import ai_trading.core.bot_engine as be
    APIError = get_api_error_cls()
    if getattr(be, "trading_client", None) is not None:
        return True
    for attempt in (1, 2):
        try:
            from ai_trading.core.bot_engine import _ensure_alpaca_env_or_raise
            key, secret, base_url = _ensure_alpaca_env_or_raise()
        except COMMON_EXC as e:
            logger.error("ALPACA_ENV_RESOLUTION_FAILED", extra={"error": str(e)})
            if attempt == 1:
                try:
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

