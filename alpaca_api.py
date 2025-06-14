import logging
import os
import time
import requests

from alerts import send_slack_alert

SHADOW_MODE = os.getenv("SHADOW_MODE", "0") == "1"

logger = logging.getLogger(__name__)


def submit_order(api, req, log: logging.Logger | None = None):
    """Submit an order with rate limit handling and optional shadow mode."""
    log = log or logger
    if SHADOW_MODE:
        log.info(
            f"SHADOW_MODE: Would place order: {getattr(req, 'symbol', '')} {getattr(req, 'qty', '')} "
            f"{getattr(req, 'side', '')} {req.__class__.__name__} {getattr(req, 'time_in_force', '')}"
        )
        return {"status": "shadow", "symbol": getattr(req, 'symbol', ''), "qty": getattr(req, 'qty', 0)}

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            order = api.submit_order(order_data=req)
            if hasattr(order, 'status_code') and getattr(order, 'status_code') == 429:
                raise requests.exceptions.HTTPError("API rate limit exceeded (429)")
            return order
        except requests.exceptions.HTTPError as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = attempt * 2
                log.warning(
                    f"Rate limit hit for Alpaca order (attempt {attempt}/{max_retries}), sleeping {wait}s"
                )
                time.sleep(wait)
                continue
            log.error("HTTPError in Alpaca submit_order: %s", e, exc_info=True)
            send_slack_alert(f"HTTP error submitting order: {e}")
            if attempt == max_retries:
                raise
        except Exception as e:
            log.error("Error in Alpaca submit_order (attempt %s): %s", attempt, e, exc_info=True)
            if attempt == max_retries:
                send_slack_alert(f"Failed to submit order after {max_retries} attempts: {e}")
                raise
            time.sleep(attempt * 2)

