import logging
import requests

from config import get_env

SLACK_WEBHOOK = get_env("SLACK_WEBHOOK")

logger = logging.getLogger(__name__)


def send_slack_alert(message: str) -> None:
    """Send a Slack alert if ``SLACK_WEBHOOK`` is configured."""
    if SLACK_WEBHOOK:
        try:
            requests.post(SLACK_WEBHOOK, json={"text": message}, timeout=5)
        except Exception as exc:  # pragma: no cover - network issues
            logger.error("Failed to send Slack alert: %s", exc)

