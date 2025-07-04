"""Portfolio rebalancing utilities."""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import config
from alerts import send_slack_alert

logger = logging.getLogger(__name__)

REBALANCE_INTERVAL_MIN = int(config.get_env("REBALANCE_INTERVAL_MIN", "1440"))

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


def start_rebalancer(ctx) -> threading.Thread:
    """Run :func:`maybe_rebalance` every minute in a background thread."""
    def loop() -> None:
        while True:
            try:
                maybe_rebalance(ctx)
            except Exception as exc:  # pragma: no cover - background errors
                logger.error("Rebalancer loop error: %s", exc)
            time.sleep(60)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t

