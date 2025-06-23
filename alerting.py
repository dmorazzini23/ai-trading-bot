"""Simple alerting utilities for sending Slack messages."""

from __future__ import annotations

import logging

import requests

from validate_env import settings

logger = logging.getLogger(__name__)


def send_slack_alert(message: str) -> None:
    """Send ``message`` to Slack via the configured webhook.

    Parameters
    ----------
    message : str
        The text to send to Slack.
    """
    webhook = settings.SLACK_WEBHOOK
    if not webhook:
        logger.warning("SLACK_WEBHOOK not set; cannot send alert")
        return
    try:
        requests.post(webhook, json={"text": message}, timeout=5)
    except requests.RequestException as exc:  # pragma: no cover - network issues
        logger.error("Failed to send Slack alert: %s", exc)
