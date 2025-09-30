from __future__ import annotations

"""Alpaca helpers moved from core.bot_engine with façade re-export.

The canonical implementations live here. core.bot_engine imports these names
so existing call sites continue to work. This reduces size and coupling of
bot_engine while keeping runtime behavior identical.
"""

from typing import Any
import os
import sys

from ai_trading.logging import get_logger, logger_once
from ai_trading.alpaca_api import (
    ALPACA_AVAILABLE,
    TradingClientAdapter,
    get_trading_client_cls,
    get_data_client_cls,
    get_api_error_cls,
    _set_alpaca_service_available,
)
from ai_trading.config.management import get_env, is_shadow_mode
from ai_trading.exc import COMMON_EXC

logger = get_logger(__name__)


_shared_logger_once: Any | None = None


def _get_bot_logger_once() -> Any:
    """Return the shared logger_once exposed by bot_engine when available."""

    global _shared_logger_once
    if _shared_logger_once is not None:
        return _shared_logger_once

    bot_engine_module = sys.modules.get("ai_trading.core.bot_engine")
    if bot_engine_module is not None:
        shared = getattr(bot_engine_module, "logger_once", None)
        if shared is not None:
            _shared_logger_once = shared
            return shared

    try:
        from ai_trading.core import bot_engine as bot_engine_module  # noqa: WPS433 - deferred import
    except COMMON_EXC:
        return logger_once
    except Exception:
        return logger_once

    shared = getattr(bot_engine_module, "logger_once", None)
    if shared is None:
        return logger_once

    _shared_logger_once = shared
    return shared


def _validate_trading_api(api: Any) -> bool:
    """Ensure trading api exposes required methods across SDK variants.

    - Guarantees `list_orders` exists, mapping to `get_orders(...)` if needed.
    - Guarantees `list_positions` exists, mapping to `get_all_positions()` if needed.
    """
    log_once = _get_bot_logger_once()
    warned_adapter = False
    adapted_api = False

    def _try_setattr(target: Any, name: str, value: Any) -> bool:
        """Set attribute on ``target`` or its wrapped client when possible."""

        try:
            setattr(target, name, value)
            return True
        except AttributeError:
            wrapped = getattr(target, "_ai_trading_wrapped_client", None)
            if wrapped is not None and wrapped is not target:
                try:
                    setattr(wrapped, name, value)
                    return True
                except AttributeError:
                    return False
            return False

    if api is None:
        if ALPACA_AVAILABLE and not is_shadow_mode():
            log_once.error("ALPACA_CLIENT_MISSING", key="alpaca_client_missing")
        else:
            log_once.warning("ALPACA_CLIENT_MISSING", key="alpaca_client_missing")
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

            if _try_setattr(api, "list_orders", _list_orders_wrapper):
                log_once.info(
                    "API_GET_ORDERS_MAPPED", key="alpaca_get_orders_mapped"
                )
                adapted_api = True
            else:  # pragma: no cover - defensive fallback
                log_once.error(
                    "ALPACA_LIST_ORDERS_PATCH_FAILED",
                    key="alpaca_list_orders_patch_failed",
                )
        else:
            log_once.error("ALPACA_LIST_ORDERS_MISSING", key="alpaca_list_orders_missing")
            if not is_shadow_mode() and not os.getenv("PYTEST_RUNNING"):
                raise RuntimeError("Alpaca client missing list_orders method")
            return False

    # Map positions accessor for alpaca-py `TradingClient`
    if not hasattr(api, "list_positions") and hasattr(api, "get_all_positions"):
        try:
            def _list_positions_wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
                # TradingClient.get_all_positions() takes no filters; ignore args
                return api.get_all_positions()  # type: ignore[attr-defined]

            if _try_setattr(api, "list_positions", _list_positions_wrapper):
                log_once.info(
                    "API_GET_POSITIONS_MAPPED", key="alpaca_get_positions_mapped"
                )
            else:  # pragma: no cover - defensive fallback
                log_once.error(
                    "ALPACA_LIST_POSITIONS_PATCH_FAILED",
                    key="alpaca_list_positions_patch_failed",
                )
        except Exception:
            # Non-fatal; caller may handle attribute absence
            pass

    if not hasattr(api, "cancel_order"):
        cancel_by_id = getattr(api, "cancel_order_by_id", None)
        cancel_orders = getattr(api, "cancel_orders", None)

        if callable(cancel_by_id):
            def _cancel_order_wrapper(order_id: Any):
                return cancel_by_id(order_id)

            if _try_setattr(api, "cancel_order", _cancel_order_wrapper):
                log_once.info(
                    "API_CANCEL_ORDER_MAPPED", key="alpaca_cancel_order_mapped"
                )
                adapted_api = True
            else:  # pragma: no cover - defensive fallback
                log_once.error(
                    "ALPACA_CANCEL_ORDER_PATCH_FAILED",
                    key="alpaca_cancel_order_patch_failed",
                )
        elif callable(cancel_orders):
            CancelOrdersRequest = None
            try:
                requests_mod = __import__(
                    "alpaca.trading.requests", fromlist=["CancelOrdersRequest"]
                )
                CancelOrdersRequest = getattr(requests_mod, "CancelOrdersRequest", None)
            except Exception as exc:  # pragma: no cover - defensive fallback
                log_once.error(
                    "ALPACA_CANCEL_ORDERS_REQUEST_IMPORT_FAILED",
                    key="alpaca_cancel_orders_request_import_failed",
                    extra={"error": str(exc)},
                )

            if CancelOrdersRequest is None:
                class _FallbackCancelOrdersRequest:
                    def __init__(self, **kwargs):
                        if not kwargs:
                            raise TypeError("payload required")
                        self.payload = kwargs

                CancelOrdersRequest = _FallbackCancelOrdersRequest

            def _cancel_order_via_batch(order_id: Any):

                last_error: Exception | None = None
                init_variants = (
                    {"order_id": order_id},
                    {"order_ids": [order_id]},
                    {"client_order_id": order_id},
                )

                for init_kwargs in init_variants:
                    try:
                        request_obj = CancelOrdersRequest(**init_kwargs)
                    except TypeError as exc:
                        last_error = exc
                        continue

                    for caller in (
                        lambda ro=request_obj: cancel_orders(ro),
                        lambda ro=request_obj: cancel_orders(request=ro),
                        lambda ro=request_obj: cancel_orders(
                            cancel_orders_request=ro
                        ),
                    ):
                        try:
                            return caller()
                        except TypeError as exc:
                            last_error = exc
                            continue

                raise RuntimeError(
                    "Alpaca client cancel_orders shim could not adapt provided API"
                ) from last_error

            if _try_setattr(api, "cancel_order", _cancel_order_via_batch):
                log_once.info(
                    "API_CANCEL_ORDERS_MAPPED", key="alpaca_cancel_orders_mapped"
                )
                adapted_api = True
            else:  # pragma: no cover - defensive fallback
                log_once.error(
                    "ALPACA_CANCEL_ORDER_PATCH_FAILED",
                    key="alpaca_cancel_order_patch_failed",
                )
        else:
            log_once.error(
                "ALPACA_CANCEL_ORDER_MISSING", key="alpaca_cancel_order_missing"
            )
            return False
    if adapted_api:
        log_once.warning("ALPACA_API_ADAPTER", key="alpaca_api_adapter")
        warned_adapter = True

    try:
        TradingClient = get_trading_client_cls()
    except RuntimeError:
        log_once.warning(
            "ALPACA_TRADING_CLIENT_CLASS_MISSING",
            key="alpaca_trading_client_class_missing",
        )
        return True
    wrapped_client = getattr(api, "_ai_trading_wrapped_client", None)
    if isinstance(api, TradingClient):
        pass
    elif wrapped_client is not None and isinstance(wrapped_client, TradingClient):
        pass
    elif not warned_adapter:
        log_once.warning("ALPACA_API_ADAPTER", key="alpaca_api_adapter")
        warned_adapter = True
    return True


def list_open_orders(api: Any):
    """Return open orders from api, compatible across SDK versions."""
    return api.list_orders(status="open")


def ensure_alpaca_attached(ctx) -> None:
    """Attach global trading client to the context if it's missing."""
    log_once = _get_bot_logger_once()
    if os.getenv("PYTEST_RUNNING") and not (
        os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY")
    ):
        raise RuntimeError("Missing Alpaca API credentials")
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
        if ALPACA_AVAILABLE and not is_shadow_mode():
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
    import ai_trading.core.bot_engine as be

    log_once = _get_bot_logger_once()

    if getattr(be, "trading_client", None) is not None:
        return True
    if not ALPACA_AVAILABLE:
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
                except Exception:
                    pass
                continue
            log_once.error("ALPACA_CLIENT_INIT_FAILED - env", key="alpaca_client_init_failed")
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
            log_once.warning("ALPACA_INIT_SKIPPED - shadow mode or missing credentials", key="alpaca_init_skipped")
            logger.info("ALPACA_DIAG", extra=diag)
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
        factory_cls = trading_client_cls
        legacy_module = str(getattr(trading_client_cls, "__module__", ""))
        if legacy_module.startswith("alpaca_trade_api"):
            try:
                from alpaca.trading.client import TradingClient as _NativeTradingClient
            except Exception:
                _NativeTradingClient = None
            if _NativeTradingClient is not None:
                factory_cls = _NativeTradingClient
        try:
            raw_trading_client = factory_cls(
                api_key=key,
                secret_key=secret,
                paper="paper" in str(base_url).lower(),
                url_override=base_url,
            )
            trading_client_obj: Any
            if isinstance(raw_trading_client, TradingClientAdapter):
                trading_client_obj = raw_trading_client
            else:
                trading_client_obj = TradingClientAdapter(raw_trading_client)
            be.trading_client = trading_client_obj
            be.data_client = stock_client_cls(api_key=key, secret_key=secret)
        except (APIError, TypeError, ValueError, OSError) as e:
            logger.error("ALPACA_CLIENT_INIT_FAILED", extra={"error": str(e)})
            log_once.error("ALPACA_CLIENT_INIT_FAILED - client", key="alpaca_client_init_failed")
            be.trading_client = None
            be.data_client = None
            return False
        logger.info("ALPACA_CLIENT_INIT_SUCCESS")
        try:
            from ai_trading.core.bot_engine import _alpaca_diag_info
            logger.info("ALPACA_DIAG", extra={"initialized": True, **_alpaca_diag_info()})
        except Exception:
            pass
        return True
    return False
