from __future__ import annotations
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def compute_portfolio_weights(ctx, symbols: List[str]) -> Dict[str, float]:
    """
    Equal-weight portfolio with basic data sanity. Uses latest close via ctx.data_fetcher.
    """
    if not symbols:
        return {}
    try:
        from ai_trading.utils import get_latest_close  # local util
    except Exception:
        # minimal fallback to avoid hard crash
        def get_latest_close(df): return 1.0

    prices = [get_latest_close(ctx.data_fetcher.get_daily_df(ctx, s)) for s in symbols]
    pairs = [(s, p) for s, p in zip(symbols, prices) if (isinstance(p, (int, float)) and p > 0)]
    if not pairs:
        logger.error("compute_portfolio_weights: no valid prices; returning empty weights")
        return {}
    symbols, _ = zip(*pairs)
    w = 1.0 / len(symbols)
    return {s: w for s in symbols}