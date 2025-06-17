"""Lightweight Alpaca REST helpers with retry and validation."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import pandas as pd
import requests
from alerts import send_slack_alert

if "ALPACA_API_KEY" in os.environ:
    os.environ.setdefault("APCA_API_KEY_ID", os.environ["ALPACA_API_KEY"])
if "ALPACA_SECRET_KEY" in os.environ:
    os.environ.setdefault("APCA_API_SECRET_KEY", os.environ["ALPACA_SECRET_KEY"])

SHADOW_MODE = os.getenv("SHADOW_MODE", "0") == "1"

logger = logging.getLogger(__name__)

_warn_counts = defaultdict(int)

ALPACA_BASE_URL = "https://api.alpaca.markets"
HEADERS = {
    "APCA-API-KEY-ID": "your_key_id",  # Replace with your env var or config loader
    "APCA-API-SECRET-KEY": "your_secret_key",
}


def _warn_limited(key: str, msg: str, *args, limit: int = 3, **kwargs) -> None:
    """Log ``msg`` up to ``limit`` times for a given ``key``."""

    if _warn_counts[key] < limit:
        logger.warning(msg, *args, **kwargs)
        _warn_counts[key] += 1
        if _warn_counts[key] == limit:
            logger.warning("Further '%s' warnings suppressed", key)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def alpaca_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Perform a GET request to the Alpaca API with retries."""

    url = f"{ALPACA_BASE_URL}{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def get_account() -> Optional[Dict[str, Any]]:
    """Return account details or ``None`` on failure."""

    data = alpaca_get("/v2/account")
    if data is None:
        logger.error("Failed to get Alpaca account data")
    return data


def submit_order(api, req, log: logging.Logger | None = None):
    """Submit an order with retry and optional shadow mode."""
    log = log or logger
    if SHADOW_MODE:
        log.info(
            f"SHADOW_MODE: Would place order: {getattr(req, 'symbol', '')} {getattr(req, 'qty', '')} "
            f"{getattr(req, 'side', '')} {req.__class__.__name__} {getattr(req, 'time_in_force', '')}"
        )
        return {
            "status": "shadow",
            "symbol": getattr(req, "symbol", ""),
            "qty": getattr(req, "qty", 0),
        }

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            order = api.submit_order(order_data=req)
            if hasattr(order, "status_code") and getattr(order, "status_code") == 429:
                raise requests.exceptions.HTTPError("API rate limit exceeded (429)")
            return order
        except requests.exceptions.HTTPError as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = attempt * 2
                _warn_limited(
                    "order-rate-limit",
                    "Rate limit hit for Alpaca order (attempt %s/%s), sleeping %ss",
                    attempt,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue
            log.error("HTTPError in Alpaca submit_order: %s", e, exc_info=True)
            send_slack_alert(f"HTTP error submitting order: {e}")
            if attempt == max_retries:
                raise
        except Exception as e:
            log.error(
                "Error in Alpaca submit_order (attempt %s): %s",
                attempt,
                e,
                exc_info=True,
            )
            if attempt == max_retries:
                send_slack_alert(f"Failed to submit order after {max_retries} attempts: {e}")
                raise
            time.sleep(attempt * 2)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def fetch_bars(
    api: Any,
    symbols: list[str] | str,
    timeframe: Any,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Return OHLCV bars ``DataFrame`` from Alpaca REST API."""

    bars_df = api.get_bars(symbols, timeframe, start=start, end=end).df

    if bars_df is None or bars_df.empty:
        logger.warning("No data received for symbols: %s", symbols)
        return pd.DataFrame()

    bars_df = bars_df.reset_index()
    logger.info(
        "Fetched data shape: %s cols=%s",
        bars_df.shape,
        bars_df.columns.tolist(),
    )
    if "timestamp" not in bars_df.columns:
        logger.error("Timestamp column missing in bars_df for symbols: %s", symbols)
        return pd.DataFrame()

    return bars_df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
