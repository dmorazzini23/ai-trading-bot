import logging
from typing import List, Dict

from utils import get_latest_close

logger = logging.getLogger(__name__)

# AI-AGENT-REF: moved from bot_engine to break import cycle

def compute_portfolio_weights(ctx, symbols: List[str]) -> Dict[str, float]:
    """Equal-weight portfolio using a dummy-price fallback for missing data."""
    n = len(symbols)
    if n == 0:
        logger.warning("No tickers to weight—skipping.")
        return {}

    prices = [get_latest_close(ctx.data_fetcher.get_daily_df(ctx, s)) for s in symbols]
    inv_prices = [1.0 / p if p > 0 else 1.0 for p in prices]
    total_inv = sum(inv_prices)
    weights = {s: inv / total_inv for s, inv in zip(symbols, inv_prices)}
    logger.info("PORTFOLIO_WEIGHTS", extra={"weights": weights})
    return weights

