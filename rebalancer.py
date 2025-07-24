"""Portfolio rebalancing utilities."""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import config
from portfolio import compute_portfolio_weights

logger = logging.getLogger(__name__)

REBALANCE_INTERVAL_MIN = int(config.get_env("REBALANCE_INTERVAL_MIN", "10"))

_last_rebalance = datetime.now(timezone.utc)


def rebalance_portfolio(ctx) -> None:
    """Placeholder for portfolio rebalancing logic."""
    logger.info("Rebalancing portfolio")
    # Actual rebalance logic would go here


def maybe_rebalance(ctx) -> None:
    """Rebalance when interval has elapsed."""
    global _last_rebalance
    now = datetime.now(timezone.utc)
    if (now - _last_rebalance) >= timedelta(minutes=REBALANCE_INTERVAL_MIN):
        portfolio = getattr(ctx, "portfolio_weights", {})
        # always trigger at least one rebalance if no existing weights
        if not portfolio:
            rebalance_portfolio(ctx)
            _last_rebalance = now
        else:
            current = compute_portfolio_weights(ctx, list(portfolio.keys()))
            drift = (
                max(abs(current.get(s, 0) - portfolio.get(s, 0)) for s in current)
                if current
                else 0.0
            )
            if drift > config.PORTFOLIO_DRIFT_THRESHOLD:
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
            # AI-AGENT-REF: reduce loop churn
            time.sleep(600)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t

