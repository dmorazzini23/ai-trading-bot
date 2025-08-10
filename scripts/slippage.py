import logging
from ai_trading.telemetry.metrics_logger import log_metrics
from validate_env import settings

logger = logging.getLogger(__name__)

SLIPPAGE_THRESHOLD = settings.SLIPPAGE_THRESHOLD


def monitor_slippage(expected: float | None, actual: float, symbol: str) -> None:
    """Check slippage and send alert when above threshold."""
    if expected:
        pct = abs(actual - expected) / expected
        log_metrics({"symbol": symbol, "slippage_pct": pct}, filename="metrics/slippage.csv")
        if pct > SLIPPAGE_THRESHOLD:
            msg = f"High slippage {pct:.2%} on {symbol}"
            logger.warning(msg)
