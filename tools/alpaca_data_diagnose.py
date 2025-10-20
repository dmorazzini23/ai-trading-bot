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

from alpaca_trade_api.common import URL
from alpaca_trade_api.rest import REST


def _client(key_id: str | None, secret_key: str | None, base_url: str | None) -> REST:
    api_base = URL(base_url or "https://paper-api.alpaca.markets")
    return REST(key_id, secret_key, api_base, api_version="v2")


def diagnose(symbol: str, key_id: str | None, secret_key: str | None, base_url: str | None) -> dict[str, object]:
    client = _client(key_id, secret_key, base_url)
    now = datetime.now(UTC)
    start = now - timedelta(minutes=15)
    start_iso = start.replace(microsecond=0).isoformat()
    end_iso = now.replace(microsecond=0).isoformat()

    result: dict[str, object] = {"symbol": symbol, "timestamp": now.isoformat()}

    try:
        bars = client.get_bars(symbol, "1Min", start=start_iso, end=end_iso, limit=5)
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
        quote = client.get_latest_quote(symbol)
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
    parser = argparse.ArgumentParser(description="Diagnose Alpaca data availability.")
    parser.add_argument("symbol", help="Ticker to probe (e.g. AAPL)")
    parser.add_argument("--key-id", default=None, help="Alpaca API key ID (defaults to env ALPACA_API_KEY_ID)")
    parser.add_argument(
        "--secret-key",
        default=None,
        help="Alpaca API secret key (defaults to env ALPACA_API_SECRET_KEY)",
    )
    parser.add_argument(
        "--data-url",
        default="https://data.alpaca.markets/v2",
        help="Alpaca data API base URL",
    )
    args = parser.parse_args()

    payload = diagnose(args.symbol.upper(), args.key_id, args.secret_key, args.data_url)
    print(json.dumps(payload, indent=2, sort_keys=True))

    if payload.get("minute_status") != "ok" or payload.get("quote_status") not in {"ok", "incomplete"}:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
