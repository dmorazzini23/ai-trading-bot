from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from ai_trading.data.bars import (
    StockBarsRequest,
    TimeFrame,
    TimeFrameUnit,
    _ensure_df,
    empty_bars_dataframe,
    safe_get_stock_bars,
)

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    import pandas as pd
logger = logging.getLogger(__name__)
_portfolio_lock = threading.RLock()

def _last_close_from(df: pd.DataFrame) -> float | None:
    """Extract last close value from DataFrame."""
    if df is None or df.empty:
        return None
    for col in ('close', 'Close', 'c'):
        if col in df.columns:
            return float(df[col].iloc[-1])
    lower = {c.lower(): c for c in df.columns}
    if 'close' in lower:
        return float(df[lower['close']].iloc[-1])
    return None

def get_latest_price(ctx, symbol: str) -> float | None:
    """Return latest price using daily bars with minute fallback."""
    import pandas as pd

    df_day = _ensure_df(ctx.data_fetcher.get_daily_df(ctx, symbol))
    price = _last_close_from(df_day)
    if price is not None:
        return price
    now_utc = pd.Timestamp.now(tz="UTC")
    start_iso = (now_utc.normalize() - pd.Timedelta(days=1)).isoformat()
    end_iso = (now_utc + pd.Timedelta(minutes=1)).isoformat()
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_iso,
            end=end_iso,
            feed="iex",
        )
        df_min = _ensure_df(
            safe_get_stock_bars(
                getattr(ctx, "data_client", None), req, symbol, "PRICE_SNAPSHOT"
            )
        )
        if df_min.empty:
            req.feed = "sip"
            df_min = _ensure_df(
                safe_get_stock_bars(
                    getattr(ctx, "data_client", None),
                    req,
                    symbol,
                    "PRICE_SNAPSHOT_SIP",
                )
            )
    except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError, OSError):
        df_min = pd.DataFrame()
    return _last_close_from(df_min)

def compute_portfolio_weights(ctx, symbols: list[str]) -> dict[str, float]:
    """Equal-weight portfolio using a dummy-price fallback for missing data."""
    with _portfolio_lock:
        n = len(symbols)
        if n == 0:
            logger.debug('No tickers to weight—skipping (no existing positions).')
            return {}
        if n > 50:
            logger.warning('Too many symbols (%d), limiting to 50', n)
            symbols = symbols[:50]
        closes = {s: get_latest_price(ctx, s) for s in symbols}
        closes = {s: c for s, c in closes.items() if isinstance(c, int | float) and c > 0}
        if not closes:
            logger.error('No valid prices found for any symbols')
            return {}
        inv_prices = {s: 1.0 / c for s, c in closes.items()}
        total_inv = sum(inv_prices.values()) or 1.0
        weights = {s: inv / total_inv for s, inv in inv_prices.items()}
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning('Portfolio weights sum to %.3f, normalizing', weight_sum)
            weights = {s: w / weight_sum for s, w in weights.items()}
        logger.info('PORTFOLIO_WEIGHTS', extra={'weights': weights})
        return weights

def is_high_volatility(current_stddev: float, baseline_stddev: float) -> bool:
    """Return ``True`` when ``current_stddev`` exceeds twice the baseline."""
    return current_stddev > 2 * baseline_stddev

def log_portfolio_summary(ctx) -> None:
    """Log cash, equity, exposure and position summary."""
    try:
        import signal
        import pandas as pd

        def timeout_handler(signum, frame):
            raise TimeoutError("API call timed out")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        try:
            acct = ctx.api.get_account()
            positions = ctx.api.list_positions()
        finally:
            signal.alarm(0)
        cash = float(acct.cash)
        equity = float(acct.equity)
        logger.debug('Raw Alpaca positions: %s', positions)
        exposure = sum((abs(float(p.market_value)) for p in positions)) / equity * 100 if equity > 0 else 0.0
        try:
            adaptive_cap = ctx.risk_engine._adaptive_global_cap()
        except AttributeError:
            adaptive_cap = 0.0
        except (TypeError, ValueError) as e:
            logger.debug('Risk engine calculation error: %s', e)
            adaptive_cap = 0.0
        logger.info('Portfolio summary: cash=$%.2f, equity=$%.2f, exposure=%.2f%%, positions=%d', cash, equity, exposure, len(positions))
        logger.info('Weights vs positions: weights=%s, positions=%s, cash=$%.2f', getattr(ctx, 'portfolio_weights', {}), {p.symbol: int(p.qty) for p in positions}, cash)
        logger.info('CYCLE SUMMARY adaptive_cap=%.1f', adaptive_cap)
    except TimeoutError:
        logger.error('Portfolio summary timed out', extra={'component': 'portfolio_summary', 'error_type': 'timeout'})
    except (AttributeError, KeyError) as exc:
        logger.warning('Portfolio summary failed - missing attribute/key: %s', exc, extra={'component': 'portfolio_summary', 'error_type': 'attribute'})
    except (ValueError, TypeError) as exc:
        logger.warning('Portfolio summary failed - data conversion error: %s', exc, extra={'component': 'portfolio_summary', 'error_type': 'data'})
    except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError, OSError) as exc:
        logger.warning('Portfolio summary failed - unexpected error: %s', exc, extra={'component': 'portfolio_summary', 'error_type': 'unexpected'})