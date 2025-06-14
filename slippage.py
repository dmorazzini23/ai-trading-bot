import logging
from config import get_env
from alerts import send_slack_alert

SLIPPAGE_THRESHOLD = float(get_env("SLIPPAGE_THRESHOLD", "0.003"))

logger = logging.getLogger(__name__)


def monitor_slippage(expected: float | None, actual: float, symbol: str) -> None:
    """Check slippage and send alert when above threshold."""
    if expected:
        pct = abs(actual - expected) / expected
        if pct > SLIPPAGE_THRESHOLD:
            msg = f"High slippage {pct:.2%} on {symbol}"
            logger.warning(msg)
            send_slack_alert(msg)
