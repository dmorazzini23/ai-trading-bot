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
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import os

from alpaca_trade_api.common import URL
from alpaca_trade_api.rest import REST


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


def _client(key_id: str | None, secret_key: str | None, base_url: str | None) -> REST:
    api_base = URL(base_url or "https://paper-api.alpaca.markets")
    return REST(key_id=key_id, secret_key=secret_key, base_url=api_base, api_version="v2")


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
    start_iso = start.replace(microsecond=0).isoformat()
    end_iso = now.replace(microsecond=0).isoformat()

    result: dict[str, object] = {
        "symbol": symbol,
        "timestamp": now.isoformat(),
        "feed": feed or "sip",
    }

    try:
        bars = client.get_bars(symbol, "1Min", start=start_iso, end=end_iso, limit=5, feed=feed)
    except Exception as exc:  # pragma: no cover - diagnostic path
        result["minute_error"] = str(exc)
    else:
        if not bars:
            result["minute_status"] = "empty"
        else:
            bar = bars[-1]
            result["minute_status"] = "ok"
            result["minute_close"] = getattr(bar, "c", None) or getattr(bar, "close", None)
            result["minute_volume"] = getattr(bar, "v", None) or getattr(bar, "volume", None)

    try:
        quote = client.get_latest_quote(symbol, feed=feed)
    except Exception as exc:  # pragma: no cover - diagnostic path
        result["quote_error"] = str(exc)
    else:
        if quote is None:
            result["quote_status"] = "missing"
        else:
            bid = getattr(quote, "bid_price", None)
            ask = getattr(quote, "ask_price", None)
            result["quote_status"] = "ok" if bid and ask else "incomplete"
            result["bid"] = bid
            result["ask"] = ask
            quote_ts = getattr(quote, "timestamp", None) or getattr(quote, "t", None)
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
