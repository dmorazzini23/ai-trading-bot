"""Alert utilities with throttling support."""

import logging
import time
from threading import Lock
import requests

from config import get_env

SLACK_WEBHOOK = get_env("SLACK_WEBHOOK")

_alert_lock = Lock()
_last_sent = {}
THROTTLE_SEC = 30

logger = logging.getLogger(__name__)


def send_slack_alert(message: str, *, key: str | None = None) -> None:
    """Send a Slack alert with basic throttling.

    Parameters
    ----------
    message : str
        Text to send.
    key : str, optional
        Distinct alert type for throttling. Defaults to ``message``.
    """
    if not SLACK_WEBHOOK:
        return

    ident = key or message
    now = time.monotonic()
    with _alert_lock:
        last = _last_sent.get(ident, 0.0)
        if now - last < THROTTLE_SEC:
            logger.info("Alert throttled: %s", ident)
            return
        _last_sent[ident] = now

    try:
        requests.post(SLACK_WEBHOOK, json={"text": message}, timeout=5)
    except Exception as exc:  # pragma: no cover - network issues
        logger.error("Failed to send Slack alert: %s", exc)

