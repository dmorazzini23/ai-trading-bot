from __future__ import annotations

from ai_trading.logging import get_logger

logger = get_logger(__name__)
SLIPPAGE_THRESHOLD = 0.01

def estimate(price: float, side: str, bps: float=0.0) -> float:
    """
    Deterministic slippage estimator.
    - price: last/mark price
    - side : 'buy' or 'sell'
    - bps  : basis points to add/subtract (default 0 for CI determinism)
    """
    if not isinstance(price, int | float) or price <= 0:
        raise ValueError('price must be a positive number')
    side_l = (side or '').lower()
    if side_l not in {'buy', 'sell'}:
        raise ValueError("side must be 'buy' or 'sell'")
    adj = price * (bps / 10000.0)
    return price + adj if side_l == 'buy' else price - adj


def monitor_slippage(expected_price: float, executed_price: float, symbol: str) -> float:
    """Return slippage ratio and emit warning when threshold is exceeded."""
    if expected_price <= 0:
        raise ValueError("expected_price must be positive")
    slip_ratio = abs(float(executed_price) - float(expected_price)) / float(expected_price)
    if slip_ratio > float(SLIPPAGE_THRESHOLD):
        logger.warning(
            "SLIPPAGE_ALERT | %s expected=%.4f executed=%.4f slippage=%.4f"
            % (
                symbol,
                float(expected_price),
                float(executed_price),
                slip_ratio,
            ),
        )
    return slip_ratio
