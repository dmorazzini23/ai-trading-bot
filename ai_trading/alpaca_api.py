from __future__ import annotations

import importlib.util
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

__all__ = [
    "ALPACA_AVAILABLE",
    "SHADOW_MODE",
    "HTTP_RETRY_CODES",
    "make_client_order_id",
    "get_client",
    "submit_order",
    "alpaca_get",
    "start_trade_updates_stream",
]


def _module_available(name: str) -> bool:
    # If tests insert a None sentinel, treat as unavailable.
    if name in sys.modules and sys.modules[name] is None:
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


ALPACA_AVAILABLE: bool = all(
    _module_available(n) for n in ("alpaca_trade_api", "alpaca.trading", "alpaca.data", "alpaca")
)

SHADOW_MODE: bool = os.getenv("AI_TRADING_SHADOW_MODE", "false").lower() in ("1", "true", "yes")
HTTP_RETRY_CODES = {429, 500, 502, 503, 504}


def make_client_order_id(prefix: str = "bot") -> str:
    import random
    import time

    return f"{prefix}-{int(time.time()*1000)}-{random.randint(1000,9999)}"


_client = None


def get_client(
    api_key: str | None = None, api_secret: str | None = None, base_url: str | None = None
):
    """
    Lazy client getter; callers may pass explicit credentials in tests.
    """
    global _client
    if _client is not None:
        return _client
    if not ALPACA_AVAILABLE:
        return None
    import alpaca_trade_api as tradeapi  # import here to avoid import-time side effects

    _client = (
        tradeapi.REST(api_key, api_secret, base_url)
        if any([api_key, api_secret, base_url])
        else tradeapi.REST()
    )
    return _client


class AttrDict(dict):
    """Dict with attribute access; supports both obj['k'] and obj.k."""

    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


def _coerce(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _wrap_order_result(d: Mapping[str, Any]) -> AttrDict:
    ad = AttrDict(d)
    # Normalize common fields for tests
    if "id" not in ad and "order_id" in ad:
        ad["id"] = ad["order_id"]
    return ad


def submit_order(  # noqa: C901
    api=None,
    req: Any | None = None,
    *,
    symbol: str | None = None,
    qty: int | float | None = None,
    side: str | None = None,
    type: str = "market",
    time_in_force: str = "day",
    client_order_id: str | None = None,
    dry_run: bool = False,
    **kwargs: Any,
):
    """
    Unified submit helper used by tests. Returns attribute-accessible mapping.
    In SHADOW_MODE or dry_run, does not hit network.
    Retries on transient HTTP status codes if using Alpaca client.
    """
    if req is not None:
        if isinstance(req, str) and symbol is None:
            symbol = req
        elif not isinstance(req, str):
            symbol = symbol or _coerce(req, "symbol")
            qty = qty if qty is not None else _coerce(req, "qty", _coerce(req, "quantity"))
            side = side or _coerce(req, "side")
            time_in_force = _coerce(req, "time_in_force", time_in_force)
            client_order_id = client_order_id or _coerce(req, "client_order_id")
    if symbol is None or qty is None or side is None:
        raise TypeError("symbol, qty and side required")

    client = api or get_client()
    coid = client_order_id or make_client_order_id()

    if dry_run or SHADOW_MODE or client is None or not hasattr(client, "submit_order"):
        return _wrap_order_result(
            {
                "id": "dry-run",
                "client_order_id": coid,
                "status": "accepted",
                "symbol": symbol,
                "qty": qty,
            }
        )

    for attempt in range(3):
        try:
            resp = client.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force,
                client_order_id=coid,
                **kwargs,
            )
            if hasattr(resp, "__dict__"):
                data = {**getattr(resp, "__dict__", {})}
            else:
                data = dict(resp)
            if "client_order_id" not in data:
                data["client_order_id"] = coid
            return _wrap_order_result(data)
        except Exception as e:
            status = getattr(e, "status", None)
            code = getattr(e, "code", None)
            if (status or code) in HTTP_RETRY_CODES and attempt < 2:
                time.sleep(0.25 * (attempt + 1))
                continue
            raise


def alpaca_get(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - legacy stub
    return None


def start_trade_updates_stream(
    *_args: Any, **_kwargs: Any
) -> None:  # pragma: no cover - legacy stub
    return None
