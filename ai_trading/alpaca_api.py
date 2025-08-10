# ai_trading/alpaca_api.py
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Optional

import requests

from ai_trading.config import get_settings

logger = logging.getLogger(__name__)
S = get_settings()

# Base endpoints
_TRADING_BASE = (S.alpaca_base_url or "https://paper-api.alpaca.markets").rstrip("/")
_DATA_BASE = "https://data.alpaca.markets"  # market data v2

_HEADERS = {
    "APCA-API-KEY-ID": S.alpaca_api_key or "",
    "APCA-API-SECRET-KEY": S.alpaca_secret_key or "",
}

def _resolve_url(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    # quotes, bars, trades → market data; otherwise default to trading
    if path_or_url.startswith("/v2/stocks/"):
        return _DATA_BASE + path_or_url
    return _TRADING_BASE + path_or_url

def alpaca_get(path_or_url: str, *, params: Optional[dict] = None, timeout: int = 10) -> Any:
    """Tiny helper for authenticated GET to Alpaca endpoints."""
    url = _resolve_url(path_or_url)
    resp = requests.get(url, headers=_HEADERS, params=params or {}, timeout=timeout)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    return resp.json() if "json" in ctype else resp.text

# --- Trade updates stream (optional if SDK present) ---

def _sdk_available() -> bool:
    try:
        import alpaca_trade_api  # noqa: F401
        return True
    except Exception:
        return False

async def _stream_with_sdk(
    api_key: str, api_secret: str, trading_client: Any, state: Any, *, paper: bool, running: Optional[asyncio.Event]
) -> None:
    """Example async stream using alpaca-trade-api's websockets."""
    from alpaca_trade_api.stream import Stream

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
    api_key: str, api_secret: str, trading_client: Any, state: Any, *, paper: bool = True, running: Optional[asyncio.Event] = None
) -> None:
    """
    Entrypoint expected by bot_engine: kicks off an async trade-updates stream.
    If the SDK isn't installed, we log and return (bot continues without streaming).
    """
    if not _sdk_available():
        logger.warning("Alpaca SDK not installed; trade updates stream disabled")
        return
    try:
        asyncio.run(_stream_with_sdk(api_key, api_secret, trading_client, state, paper=paper, running=running))
    except RuntimeError:
        # If already in an event loop (rare here), fall back to a simple loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_stream_with_sdk(api_key, api_secret, trading_client, state, paper=paper, running=running))
        finally:
            loop.close()