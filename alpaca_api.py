import logging
import os

if "ALPACA_API_KEY" in os.environ:
    os.environ.setdefault("APCA_API_KEY_ID", os.environ["ALPACA_API_KEY"])
if "ALPACA_SECRET_KEY" in os.environ:
    os.environ.setdefault("APCA_API_SECRET_KEY", os.environ["ALPACA_SECRET_KEY"])
import time
from collections import defaultdict

import pandas as pd
import requests

from alerts import send_slack_alert

SHADOW_MODE = os.getenv("SHADOW_MODE", "0") == "1"

logger = logging.getLogger(__name__)

_warn_counts = defaultdict(int)


def _warn_limited(key: str, msg: str, *args, limit: int = 3, **kwargs) -> None:
    if _warn_counts[key] < limit:
        logger.warning(msg, *args, **kwargs)
        _warn_counts[key] += 1
        if _warn_counts[key] == limit:
            logger.warning("Further '%s' warnings suppressed", key)


def submit_order(api, req, log: logging.Logger | None = None):
    """Submit an order with rate limit handling and optional shadow mode."""
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


def fetch_bars(api, symbols, timeframe, start=None, end=None):
    """Return OHLCV bars DataFrame from Alpaca REST API."""
    bars_df = api.get_bars(symbols, timeframe, start=start, end=end).df
    if bars_df.empty:
        logging.warning(f"No data received for symbols: {symbols}")
        return pd.DataFrame()

    bars_df.reset_index(inplace=True)
    logging.info(f"Fetched data shape: {bars_df.shape} with columns {bars_df.columns.tolist()}")
    logging.debug(f"Fetched data sample:\n{bars_df.head()}")

    if "timestamp" not in bars_df.columns:
        logging.error(f"Timestamp column missing in bars_df for symbols: {symbols}")
        return pd.DataFrame()

    return bars_df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
