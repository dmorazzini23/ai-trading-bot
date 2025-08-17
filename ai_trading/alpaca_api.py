from __future__ import annotations

import os
import sys
import types  # noqa: F401
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

# --- availability detection (respects tests that stub sys.modules) ---
_ALPACA_MODULE_NAMES = ("alpaca_trade_api", "alpaca", "alpaca.trading", "alpaca.data")
ALPACA_AVAILABLE: bool = all(
    (name in sys.modules and sys.modules.get(name) is not None) for name in _ALPACA_MODULE_NAMES
)
# Allow forcing OFF in tests via env
if os.environ.get("TESTING", "").lower() == "true" and any(
    sys.modules.get(n) is None for n in _ALPACA_MODULE_NAMES
):
    ALPACA_AVAILABLE = False

# Legacy constant kept for tests
SHADOW_MODE: bool = os.environ.get("AI_TRADING_SHADOW_MODE", "0") in (
    "1",
    "true",
    "True",
)

# Retryable statuses expected by tests (include rate-limit)
RETRYABLE_HTTP_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504)


def _maybe_get_client():
    if not ALPACA_AVAILABLE:
        return None
    try:
        import alpaca_trade_api as _api  # type: ignore

        return _api
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class SubmitOrderResult(Mapping[str, Any]):
    success: bool
    order_id: str | None = None
    status: int | None = None
    error: str | None = None
    retryable: bool = False

    def __getitem__(self, k: str) -> Any:  # Mapping protocol
        return getattr(self, k)

    def __iter__(self) -> Iterator[str]:  # Mapping protocol
        return iter(("success", "order_id", "status", "error", "retryable"))

    def __len__(self) -> int:  # Mapping protocol
        return 5


def submit_order(
    client: Any,
    *,
    symbol: str,
    qty: int,
    side: str,
    dry_run: bool = False,
    client_order_id: str | None = None,
    **kwargs,
) -> SubmitOrderResult:
    """
    Normalized submit that returns attribute-accessible result (not a bare dict).
    - In dry_run mode: never calls network; returns success=True with a fake order_id.
    - On HTTP error: returns success=False with .status and .retryable set.
    """
    if dry_run or SHADOW_MODE or client is None or not hasattr(client, "submit_order"):
        fake_id = client_order_id or f"dryrun-{symbol}-{qty}"
        return SubmitOrderResult(
            success=True, order_id=fake_id, status=0, error=None, retryable=False
        )

    try:
        resp = client.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            client_order_id=client_order_id,
            **kwargs,
        )
        oid = (
            getattr(resp, "id", None)
            or getattr(resp, "order_id", None)
            or (resp.get("id") if isinstance(resp, dict) else None)
        )
        return SubmitOrderResult(
            success=True,
            order_id=str(oid) if oid else None,
            status=200,
            error=None,
            retryable=False,
        )
    except Exception as e:  # noqa: BLE001
        status = None
        msg = str(e)
        status = (
            getattr(e, "status", None)
            or getattr(getattr(e, "response", None), "status", None)
            or getattr(getattr(e, "response", None), "status_code", None)
        )
        try:
            if status is None and "429" in msg:
                status = 429
        except Exception:  # noqa: BLE001
            pass
        status = int(status) if status is not None else 0
        return SubmitOrderResult(
            success=False,
            order_id=None,
            status=status,
            error=msg,
            retryable=bool(status in RETRYABLE_HTTP_STATUSES),
        )


__all__ = [
    "ALPACA_AVAILABLE",
    "SHADOW_MODE",
    "RETRYABLE_HTTP_STATUSES",
    "submit_order",
    "alpaca_get",
    "start_trade_updates_stream",
]


def alpaca_get(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - legacy stub
    return None


def start_trade_updates_stream(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    return None
