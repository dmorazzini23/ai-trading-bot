import logging
import threading

from ai_trading.utils.base import get_latest_close

logger = logging.getLogger(__name__)

# AI-AGENT-REF: moved from bot_engine to break import cycle

_portfolio_lock = threading.RLock()


def compute_portfolio_weights(ctx, symbols: list[str]) -> dict[str, float]:
    """Equal-weight portfolio using a dummy-price fallback for missing data."""
    with _portfolio_lock:
        n = len(symbols)
        if n == 0:
            logger.debug("No tickers to weight—skipping (no existing positions).")
            return {}

        if n > 50:  # Prevent excessive diversification
            logger.warning("Too many symbols (%d), limiting to 50", n)
            symbols = symbols[:50]

        prices = [
            get_latest_close(ctx.data_fetcher.get_daily_df(ctx, s)) for s in symbols
        ]

        # Filter out invalid prices
        valid_prices = [(s, p) for s, p in zip(symbols, prices, strict=False) if p > 0]
        if not valid_prices:
            logger.error("No valid prices found for any symbols")
            return {}

        symbols, prices = zip(*valid_prices, strict=False)
        inv_prices = [1.0 / p if p > 0 else 1.0 for p in prices]
        total_inv = sum(inv_prices) or 1.0  # Prevent division by zero
        weights = {
            s: inv / total_inv for s, inv in zip(symbols, inv_prices, strict=False)
        }

        # Validate weights sum to 1.0
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning("Portfolio weights sum to %.3f, normalizing", weight_sum)
            weights = {s: w / weight_sum for s, w in weights.items()}

        logger.info("PORTFOLIO_WEIGHTS", extra={"weights": weights})
        return weights


def is_high_volatility(current_stddev: float, baseline_stddev: float) -> bool:
    """Return ``True`` when ``current_stddev`` exceeds twice the baseline."""
    # AI-AGENT-REF: placeholder for future regime logic
    return current_stddev > 2 * baseline_stddev


def log_portfolio_summary(ctx) -> None:
    """Log cash, equity, exposure and position summary."""
    try:
        # Add timeout to prevent hanging
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("API call timed out")

        # Set 10 second timeout for API calls
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)

        try:
            acct = ctx.api.get_account()
            positions = ctx.api.list_open_positions()
        finally:
            signal.alarm(0)  # Cancel the alarm

        cash = float(acct.cash)
        equity = float(acct.equity)
        logger.debug("Raw Alpaca positions: %s", positions)
        exposure = (
            sum(abs(float(p.market_value)) for p in positions) / equity * 100
            if equity > 0
            else 0.0
        )
        try:
            adaptive_cap = ctx.risk_engine._adaptive_global_cap()
        except AttributeError:
            adaptive_cap = 0.0  # Risk engine not available or method missing
        except (TypeError, ValueError) as e:
            logger.debug("Risk engine calculation error: %s", e)
            adaptive_cap = 0.0
        logger.info(
            "Portfolio summary: cash=$%.2f, equity=$%.2f, exposure=%.2f%%, positions=%d",
            cash,
            equity,
            exposure,
            len(positions),
        )
        logger.info(
            "Weights vs positions: weights=%s, positions=%s, cash=$%.2f",
            getattr(ctx, "portfolio_weights", {}),
            {p.symbol: int(p.qty) for p in positions},
            cash,
        )
        logger.info("CYCLE SUMMARY adaptive_cap=%.1f", adaptive_cap)
    except TimeoutError:
        logger.error("Portfolio summary timed out", 
                    extra={"component": "portfolio_summary", "error_type": "timeout"})
    except (AttributeError, KeyError) as exc:
        logger.warning("Portfolio summary failed - missing attribute/key: %s", exc,
                      extra={"component": "portfolio_summary", "error_type": "attribute"})
    except (ValueError, TypeError) as exc:
        logger.warning("Portfolio summary failed - data conversion error: %s", exc,
                      extra={"component": "portfolio_summary", "error_type": "data"})
    except Exception as exc:  # Final safety net for unexpected errors
        logger.warning("Portfolio summary failed - unexpected error: %s", exc,
                      extra={"component": "portfolio_summary", "error_type": "unexpected"})
