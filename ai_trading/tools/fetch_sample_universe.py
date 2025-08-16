from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, UTC

from ai_trading.logging import get_logger
from ai_trading.utils import http
from ai_trading.utils.prof import StageTimer

logger = get_logger(__name__)

__all__ = ["run", "parse_cli_and_run"]


# AI-AGENT-REF: manual probe for pooled HTTP fetch

def run(symbols: list[str], timeout: float | None = None) -> int:
    """Fetch daily data for ``symbols`` using the pooled HTTP client."""
    if not symbols:
        return 0
    from ai_trading.data_fetcher import _build_daily_url  # local import to ease patching

    end = datetime.now(UTC)
    start = end - timedelta(days=7)
    urls = [_build_daily_url(sym, start, end) for sym in symbols]
    failures = 0
    with StageTimer(logger, "UNIVERSE_FETCH", universe_size=len(symbols)):
        results = http.map_get(urls, timeout=timeout)
    logger.info("HTTP_POOL_STATS", extra=http.pool_stats())
    for (url, code, _), sym in zip(results, symbols):
        if code != 200:
            logger.error("fetch failed for %s status=%s", sym, code)
            failures += 1
    return 0 if failures == 0 else 1


def parse_cli_and_run() -> int:
    parser = argparse.ArgumentParser(description="Fetch sample universe using pooled HTTP")
    parser.add_argument("--symbols", help="Comma separated symbols", default="")
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        env_syms = os.getenv("SAMPLE_UNIVERSE", "AAPL,MSFT,GOOGL")
        symbols = [s.strip() for s in env_syms.split(",") if s.strip()]

    timeout = args.timeout
    if timeout is None:
        try:
            timeout = float(os.getenv("HTTP_TIMEOUT_S", "10"))
        except Exception:
            timeout = 10.0
    return run(symbols, timeout=timeout)


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(parse_cli_and_run())
