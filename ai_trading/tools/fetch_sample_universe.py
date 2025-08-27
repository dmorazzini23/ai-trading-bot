from __future__ import annotations
import argparse
import os
from datetime import UTC, datetime, timedelta
from ai_trading.logging import get_logger
from ai_trading.utils import http
from ai_trading.utils.prof import StageTimer
from ai_trading.utils.timing import HTTP_TIMEOUT
logger = get_logger(__name__)
__all__ = ['run', 'parse_cli_and_run']

def run(symbols: list[str], timeout: float | None=None) -> int:
    """Fetch daily data for ``symbols`` using the pooled HTTP client."""
    if not symbols:
        return 0
    from ai_trading.data.fetch import _build_daily_url
    end = datetime.now(UTC)
    start = end - timedelta(days=7)
    urls = [_build_daily_url(sym, start, end) for sym in symbols]
    failures = 0
    with StageTimer(logger, 'UNIVERSE_FETCH', universe_size=len(symbols)):
        results = http.map_get(urls, timeout=timeout)
    logger.info('HTTP_POOL_STATS', extra=http.pool_stats())
    for (resp, err), sym in zip(results, symbols, strict=False):
        if err or not resp or resp[1] != 200:
            logger.error('fetch failed for %s', sym)
            failures += 1
    return 0 if failures == 0 else 1

def parse_cli_and_run() -> int:
    parser = argparse.ArgumentParser(description='Fetch sample universe using pooled HTTP')
    parser.add_argument('--symbols', help='Comma separated symbols', default='')
    parser.add_argument('--timeout', type=float, default=None)
    args = parser.parse_args()
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    else:
        env_syms = os.getenv('SAMPLE_UNIVERSE', 'AAPL,MSFT,GOOGL')
        symbols = [s.strip() for s in env_syms.split(',') if s.strip()]
    timeout = args.timeout
    if timeout is None:
        timeout = HTTP_TIMEOUT
    return run(symbols, timeout=timeout)
if __name__ == '__main__':
    import sys
    sys.exit(parse_cli_and_run())