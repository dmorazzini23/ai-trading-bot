import logging
try:  # AI-AGENT-REF: canonical env validation
    from ai_trading.validation.validate_env import Settings
    settings = Settings()
except Exception:  # noqa: BLE001
    settings = None

logger = logging.getLogger(__name__)

SLIPPAGE_THRESHOLD = settings.SLIPPAGE_THRESHOLD


def monitor_slippage(expected: float | None, actual: float, symbol: str) -> None:
    """Check slippage and send alert when above threshold."""
    if expected:
        pct = abs(actual - expected) / expected
        from ai_trading.logging import _get_metrics_logger  # AI-AGENT-REF: lazy metrics import
        _get_metrics_logger().log_metrics(
            {"symbol": symbol, "slippage_pct": pct}, filename="metrics/slippage.csv"
        )
        if pct > SLIPPAGE_THRESHOLD:
            msg = f"High slippage {pct:.2%} on {symbol}"
            logger.warning(msg)
