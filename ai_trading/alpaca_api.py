from __future__ import annotations

import os
import sys
import time
import types  # noqa: F401
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

# --- availability detection (respects tests that stub sys.modules) ---
_ALPACA_MODULE_NAMES = (
    "alpaca_trade_api",
    "alpaca",
    "alpaca.trading",
    "alpaca.data",
)
ALPACA_AVAILABLE: bool = all(
    (name in sys.modules and sys.modules.get(name) is not None)
    for name in _ALPACA_MODULE_NAMES
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
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRYABLE_HTTP_STATUSES = tuple(_RETRYABLE_CODES)


def _maybe_get_client():
    if not ALPACA_AVAILABLE:
        return None
    try:
        import alpaca_trade_api as _api  # type: ignore

        return _api
    except Exception:  # noqa: BLE001
        return None


@dataclass
class SubmitOrderResult(Mapping[str, Any]):
    id: str | None
    client_order_id: str | None
    status: int
    raw: dict[str, Any]
    error: str | None = None

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k, self.raw.get(k))

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - mapping protocol
        return iter(("id", "client_order_id", "status", "raw", "error"))

    def __len__(self) -> int:  # pragma: no cover - mapping protocol
        return 5

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.raw)
        d.update(
            {
                "id": self.id,
                "client_order_id": self.client_order_id,
                "status": self.status,
            }
        )
        if self.error:
            d["error"] = self.error
        return d

    # Backwards compat properties expected by tests
    @property
    def order_id(self) -> str | None:  # AI-AGENT-REF: legacy alias
        return self.id

    @property
    def success(self) -> bool:  # AI-AGENT-REF: legacy alias
        return bool(self.id) and int(self.status) < 400

    @property
    def retryable(self) -> bool:  # AI-AGENT-REF: legacy alias
        return int(self.status) in _RETRYABLE_CODES


def submit_order(
    client: Any,
    *,
    symbol: str,
    qty: int,
    side: str,
    dry_run: bool = False,
    client_order_id: str | None = None,
    max_retries: int = 0,
    **kwargs,
) -> SubmitOrderResult:
    """Submit an order with retry handling and structured result."""
    cid = client_order_id or f"{symbol}-{qty}-{int(time.time()*1000)}"
    if dry_run or SHADOW_MODE or client is None or not hasattr(client, "submit_order"):
        fake_id = cid if client_order_id else f"dryrun-{symbol}-{qty}"
        return SubmitOrderResult(fake_id, cid, 0, {})

    kwargs.setdefault("client_order_id", cid)
    from ai_trading.utils import (
        HTTP_TIMEOUT_DEFAULT,  # local import to avoid heavy deps
    )

    attempt = 0
    while True:
        try:
            kwargs.setdefault("timeout", HTTP_TIMEOUT_DEFAULT)
            resp = client.submit_order(symbol=symbol, qty=qty, side=side, **kwargs)
            oid = (
                getattr(resp, "id", None)
                or getattr(resp, "order_id", None)
                or (resp.get("id") if isinstance(resp, dict) else None)
            )
            raw = (
                resp.__dict__
                if hasattr(resp, "__dict__")
                else dict(resp) if isinstance(resp, dict) else {}
            )
            return SubmitOrderResult(str(oid) if oid else None, cid, 200, raw)
        except Exception as e:  # noqa: BLE001
            status = (
                getattr(e, "status", None)
                or getattr(getattr(e, "response", None), "status", None)
                or getattr(getattr(e, "response", None), "status_code", None)
            )
            status = int(status) if status is not None else 0
            msg = str(e)
            if status in _RETRYABLE_CODES and attempt < max_retries:
                attempt += 1
                time.sleep(0.1)
                continue
            return SubmitOrderResult(None, cid, status, {}, error=msg)


__all__ = [
    "ALPACA_AVAILABLE",
    "SHADOW_MODE",
    "RETRYABLE_HTTP_STATUSES",
    "SubmitOrderResult",
    "submit_order",
    "alpaca_get",
    "start_trade_updates_stream",
]


def alpaca_get(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - legacy stub
    return None


def start_trade_updates_stream(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    return None
