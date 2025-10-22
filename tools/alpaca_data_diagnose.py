#!/usr/bin/env python3
"""
Quick diagnostic for Alpaca market data availability.

This script attempts to fetch a recent minute bar and the latest NBBO
quote for a supplied symbol using the runtime configuration. It prints a
succinct JSON payload so operators can quickly determine whether the
primary data plane is returning usable payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame


def _load_env_from_file() -> None:
    """Populate environment variables from a local .env if present."""

    candidates: Iterable[Path] = (
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = value.strip()
        except OSError:
            continue
        else:
            break


def _resolve_feed(feed: str | None) -> DataFeed | None:
    if feed is None:
        return None
    try:
        return DataFeed(feed)
    except ValueError as exc:  # pragma: no cover - guarded by argparse choices
        raise RuntimeError(f"Unsupported data feed: {feed}") from exc


def _client(api_key: str, api_secret: str, base_url: str | None) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        api_key=api_key,
        secret_key=api_secret,
        url_override=base_url,
    )


def diagnose(
    symbol: str,
    key_id: str | None,
    secret_key: str | None,
    base_url: str | None,
    feed: str | None = None,
) -> dict[str, object]:
    key_id = key_id or os.getenv("ALPACA_API_KEY")
    secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
    if not key_id or not secret_key:
        raise RuntimeError("Missing explicit Alpaca credentials in env")
    client = _client(key_id, secret_key, base_url)
    now = datetime.now(UTC)
    start = now - timedelta(minutes=15)

    result: dict[str, object] = {
        "symbol": symbol,
        "timestamp": now.isoformat(),
        "feed": feed or "sip",
    }

    feed_enum = _resolve_feed(feed)

    try:
        bars_request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=now,
            limit=5,
            timeframe=TimeFrame.Minute,
            feed=feed_enum,
        )
        bar_set = client.get_stock_bars(bars_request)
    except Exception as exc:  # pragma: no cover - diagnostic path
        result["minute_error"] = str(exc)
    else:
        bars_for_symbol: Sequence[object] | None
        if hasattr(bar_set, "data"):
            bars_for_symbol = bar_set.data.get(symbol)
        elif isinstance(bar_set, dict):
            bars_for_symbol = bar_set.get(symbol)
        else:
            bars_for_symbol = None
        if not bars_for_symbol:
            result["minute_status"] = "empty"
        else:
            bar = bars_for_symbol[-1]
            result["minute_status"] = "ok"
            result["minute_close"] = getattr(bar, "close", None)
            result["minute_volume"] = getattr(bar, "volume", None)

    try:
        quote_request = StockLatestQuoteRequest(
            symbol_or_symbols=symbol,
            feed=feed_enum,
        )
        quote_response = client.get_stock_latest_quote(quote_request)
    except Exception as exc:  # pragma: no cover - diagnostic path
        result["quote_error"] = str(exc)
    else:
        if isinstance(quote_response, dict):
            quote = quote_response.get(symbol)
        else:
            quote = None
        if quote is None:
            result["quote_status"] = "missing"
        else:
            bid = getattr(quote, "bid_price", None)
            ask = getattr(quote, "ask_price", None)
            result["quote_status"] = "ok" if bid and ask else "incomplete"
            result["bid"] = bid
            result["ask"] = ask
            quote_ts = getattr(quote, "timestamp", None)
            if hasattr(quote_ts, "isoformat"):
                quote_ts = quote_ts.isoformat()
            result["quote_ts"] = quote_ts

    return result


def main() -> int:
    _load_env_from_file()

    parser = argparse.ArgumentParser(description="Diagnose Alpaca data availability.")
    parser.add_argument("symbol", help="Ticker to probe (e.g. AAPL)")
    parser.add_argument("--key-id", default=None, help="Alpaca API key (defaults to env ALPACA_API_KEY)")
    parser.add_argument(
        "--secret-key",
        default=None,
        help="Alpaca API secret key (defaults to env ALPACA_SECRET_KEY)",
    )
    parser.add_argument(
        "--data-url",
        default="https://data.alpaca.markets/v2",
        help="Alpaca data API base URL",
    )
    parser.add_argument(
        "--feed",
        default=None,
        choices=("iex", "sip"),
        help="Data feed to request (default: account default)",
    )
    args = parser.parse_args()

    payload = diagnose(args.symbol.upper(), args.key_id, args.secret_key, args.data_url, feed=args.feed)
    print(json.dumps(payload, indent=2, sort_keys=True))

    if payload.get("minute_status") != "ok" or payload.get("quote_status") not in {"ok", "incomplete"}:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
