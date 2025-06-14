from datetime import datetime, timedelta, timezone
import logging

from config import get_env
from alerts import send_slack_alert

REBALANCE_INTERVAL_MIN = int(get_env("REBALANCE_INTERVAL_MIN", "1440"))

logger = logging.getLogger(__name__)
_last_rebalance = datetime.now(timezone.utc)


def rebalance_portfolio(ctx) -> None:
    """Placeholder for portfolio rebalancing logic."""
    logger.info("Rebalancing portfolio")
    send_slack_alert("Portfolio rebalancing triggered")
    # Actual rebalance logic would go here


def maybe_rebalance(ctx) -> None:
    """Rebalance when interval has elapsed."""
    global _last_rebalance
    now = datetime.now(timezone.utc)
    if (now - _last_rebalance) >= timedelta(minutes=REBALANCE_INTERVAL_MIN):
        rebalance_portfolio(ctx)
        _last_rebalance = now
