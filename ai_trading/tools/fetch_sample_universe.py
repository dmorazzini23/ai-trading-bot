"""Utility to fetch sample market data for a set of symbols."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime, timedelta
from typing import Iterable

from ai_trading.logging import get_logger
from ai_trading.utils import http
from ai_trading.utils.http import clamp_request_timeout
from ai_trading.utils.prof import StageTimer
from ai_trading.utils.timing import HTTP_TIMEOUT

logger = get_logger(__name__)

__all__ = ["run", "parse_cli_and_run"]


def run(symbols: list[tuple[str, str]], timeout: float | None = None) -> int:
    """Fetch daily data for ``symbols`` using the pooled HTTP client.

    Parameters
    ----------
    symbols
        Sequence of ``(symbol, name)`` pairs. ``name`` is currently unused but
        included so callers can supply metadata without additional mapping.
    timeout
        Optional request timeout passed to the pooled HTTP client.
    """

    if not symbols:
        return 0

    from ai_trading.data.fetch import _build_daily_url

    end = datetime.now(UTC)
    start = end - timedelta(days=7)
    urls = [_build_daily_url(sym, start, end) for sym, _ in symbols]
    failures = 0
    timeout = clamp_request_timeout(timeout)
    with StageTimer(logger, "UNIVERSE_FETCH", universe_size=len(symbols)):
        results = http.map_get(urls, timeout=timeout)
    logger.info("HTTP_POOL_STATS", extra=http.pool_stats())
    for (resp, err), (sym, _) in zip(results, symbols, strict=False):
        if err or not resp or resp[1] != 200:
            logger.error("fetch failed for %s", sym)
            failures += 1
    return 0 if failures == 0 else 1


def _parse_symbols(raw: Iterable[str]) -> list[tuple[str, str]]:
    symbols: list[tuple[str, str]] = []
    for item in raw:
        sym, _, name = item.partition(":")
        sym = sym.strip()
        name = (name or sym).strip()
        if sym:
            symbols.append((sym, name))
    return symbols


def parse_cli_and_run() -> int:
    parser = argparse.ArgumentParser(description="Fetch sample universe using pooled HTTP")
    parser.add_argument(
        "--symbols",
        help="Comma separated symbols, optionally 'SYM:Name'",
        default="",
    )
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    if args.symbols:
        raw_syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        env_syms = os.getenv("SAMPLE_UNIVERSE", "AAPL,MSFT,GOOGL")
        raw_syms = [s.strip() for s in env_syms.split(",") if s.strip()]

    symbols = _parse_symbols(raw_syms)

    timeout = args.timeout
    if timeout is None:
        timeout = HTTP_TIMEOUT
    return run(symbols, timeout=clamp_request_timeout(timeout))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    import sys

    sys.exit(parse_cli_and_run())
