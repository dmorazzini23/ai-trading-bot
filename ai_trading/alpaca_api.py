# ai_trading/alpaca_api.py
from __future__ import annotations

import asyncio
import logging
from time import sleep
from types import SimpleNamespace
from typing import Any

from requests import HTTPError

from ai_trading.utils import clamp_timeout, http

logger = logging.getLogger(__name__)

# AI-AGENT-REF: lightweight Alpaca API helpers
SHADOW_MODE = False
DRY_RUN = False
partial_fill_tracker: dict[str, Any] = {}
partial_fills: list[str] = []

_DATA_BASE = "https://data.alpaca.markets"  # market data v2

HTTP_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}  # AI-AGENT-REF: retry list


def _resolve_url(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    from ai_trading.config.settings import get_settings

    S = get_settings()
    trading_base = (
        getattr(S, "alpaca_base_url", "https://paper-api.alpaca.markets")
        or "https://paper-api.alpaca.markets"
    ).rstrip("/")
    # quotes, bars, trades → market data; otherwise default to trading
    if path_or_url.startswith("/v2/stocks/"):
        return _DATA_BASE + path_or_url
    return trading_base + path_or_url


def alpaca_get(path_or_url: str, *, params: dict | None = None, timeout: int | None = None) -> Any:
    """Tiny helper for authenticated GET to Alpaca endpoints."""
    from ai_trading.config.settings import get_settings

    S = get_settings()
    headers = getattr(S, "alpaca_headers", {})
    url = _resolve_url(path_or_url)
    timeout = clamp_timeout(timeout, 10, 0.5)
    resp = http.get(url, headers=headers, params=params or {}, timeout=timeout)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    return resp.json() if "json" in ctype else resp.text


# --- Trade updates stream (optional if SDK present) ---


def _require_alpaca():
    """Import and return alpaca_trade_api or raise a helpful error."""
    # AI-AGENT-REF: lazy import guard
    try:
        import alpaca_trade_api as tradeapi  # type: ignore

        return tradeapi
    except Exception as e:  # pragma: no cover - safety
        raise RuntimeError(
            "alpaca_trade_api is required but not installed. "
            "Install with: pip install 'alpaca-trade-api>=3.0,<4'"
        ) from e


def _sdk_available() -> bool:
    try:
        _require_alpaca()
        return True
    except Exception:
        return False


async def _stream_with_sdk(
    api_key: str,
    api_secret: str,
    trading_client: Any,
    state: Any,
    *,
    paper: bool,
    running: asyncio.Event | None,
) -> None:
    """Example async stream using alpaca-trade-api's websockets."""
    tradeapi = _require_alpaca()
    Stream = tradeapi.stream.Stream

    base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    stream = Stream(api_key, api_secret, base_url=base_url)

    async def on_trade_update(data):
        # TODO: connect this to your BotState if/when needed
        logger.debug("trade_update: %s", data)

    stream.subscribe_trade_updates(on_trade_update)

    try:
        if running is None:
            await stream._run_forever()  # library internal run-loop
        else:
            # cooperative stop using your running Event
            task = asyncio.create_task(stream._run_forever())
            while running.is_set():
                await asyncio.sleep(0.5)
            task.cancel()
    finally:
        await stream.stop()


def start_trade_updates_stream(
    api_key: str,
    api_secret: str,
    trading_client: Any,
    state: Any,
    *,
    paper: bool = True,
    running: asyncio.Event | None = None,
) -> None:
    """
    Entrypoint expected by bot_engine: kicks off an async trade-updates stream.
    If the SDK isn't installed, we log and return (bot continues without streaming).
    """
    if not _sdk_available():
        logger.warning("Alpaca SDK not installed; trade updates stream disabled")
        return
    try:
        asyncio.run(
            _stream_with_sdk(
                api_key, api_secret, trading_client, state, paper=paper, running=running
            )
        )
    except RuntimeError:
        # If already in an event loop (rare here), fall back to a simple loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                _stream_with_sdk(
                    api_key,
                    api_secret,
                    trading_client,
                    state,
                    paper=paper,
                    running=running,
                )
            )
        finally:
            loop.close()


def submit_order(api: Any, req: Any) -> Any:
    """Pass through provider order id and normalize to object with .id."""  # AI-AGENT-REF: test helper
    if SHADOW_MODE:
        return SimpleNamespace(id=None, status="shadow")
    if DRY_RUN:
        return SimpleNamespace(id=None, status="dry_run")
    resp = api.submit_order(**req.__dict__)
    if isinstance(resp, dict):
        return SimpleNamespace(**resp)
    return resp


def submit_order_with_retry(api: Any, req: Any, retries: int = 3, backoff_s: float = 0.1) -> Any:
    """Retry submit_order on transient HTTP errors."""  # AI-AGENT-REF: bounded retry
    for attempt in range(retries + 1):
        try:
            return submit_order(api, req)
        except HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status not in HTTP_RETRYABLE_STATUS_CODES or attempt == retries:
                raise
            sleep(backoff_s * (2**attempt))


def submit_order_generic_retry(fn, *, retries: int = 3):
    """Generic retry wrapper used by tests."""  # AI-AGENT-REF: helper for callables
    for i in range(retries + 1):
        try:
            return fn()
        except HTTPError as e:
            sc = getattr(e.response, "status_code", None)
            if sc not in HTTP_RETRYABLE_STATUS_CODES or i == retries:
                raise
            sleep(0.05 * (2**i))


def handle_trade_update(event: Any) -> None:
    oid = getattr(event.order, "id", None)
    if event.event == "partial_fill":
        if oid in partial_fill_tracker:
            return
        partial_fill_tracker[oid] = event.order.filled_qty
        partial_fills.append(oid)
        logger.debug("ORDER_PARTIAL_FILL")
    elif event.event == "fill":
        partial_fill_tracker.pop(oid, None)
        logger.debug("ORDER_FILLED")
