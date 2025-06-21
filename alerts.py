"""Alert utilities with throttling support."""

import logging
import time
from threading import Lock
from validate_env import settings

import requests

logger = logging.getLogger(__name__)

SLACK_WEBHOOK = settings.SLACK_WEBHOOK

_alert_lock = Lock()
_last_sent = {}
THROTTLE_SEC = 30


def send_slack_alert(
    message: str,
    *,
    key: str | None = None,
    throttle_sec: float | None = None,
) -> None:
    """Send a Slack alert with basic throttling.

    Parameters
    ----------
    message : str
        Text to send.
    key : str, optional
        Distinct alert type for throttling. Defaults to ``message``.
    throttle_sec : float | None, optional
        Custom throttling interval for this call. Defaults to ``THROTTLE_SEC``.
    """
    throttle = THROTTLE_SEC if throttle_sec is None else throttle_sec
    if not SLACK_WEBHOOK:
        return

    ident = key or message
    now = time.monotonic()
    with _alert_lock:
        last = _last_sent.get(ident, 0.0)
        if now - last < throttle:
            logger.info("Alert throttled: %s", ident)
            return
        _last_sent[ident] = now

    try:
        requests.post(SLACK_WEBHOOK, json={"text": message}, timeout=5)
    except Exception as exc:  # pragma: no cover - network issues
        logger.error("Failed to send Slack alert: %s", exc)
