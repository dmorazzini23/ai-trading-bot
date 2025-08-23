import logging
try:
    from ai_trading.validation.validate_env import Settings
    settings = Settings()
except (KeyError, ValueError, TypeError):
    settings = None
logger = logging.getLogger(__name__)
SLIPPAGE_THRESHOLD = settings.SLIPPAGE_THRESHOLD

def monitor_slippage(expected: float | None, actual: float, symbol: str) -> None:
    """Check slippage and send alert when above threshold."""
    if expected:
        pct = abs(actual - expected) / expected
        from ai_trading.logging import _get_metrics_logger
        _get_metrics_logger().log_metrics({'symbol': symbol, 'slippage_pct': pct}, filename='metrics/slippage.csv')
        if pct > SLIPPAGE_THRESHOLD:
            msg = f'High slippage {pct:.2%} on {symbol}'
            logger.warning(msg)