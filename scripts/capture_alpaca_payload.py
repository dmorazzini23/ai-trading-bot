#!/usr/bin/env python3
"""Capture raw Alpaca bar payloads for diagnostics.

Usage examples:

    python scripts/capture_alpaca_payload.py --symbol AVGO --minutes 30

The script writes the JSON payload plus request metadata to
``artifacts/alpaca_payload_<symbol>_<timestamp>.json`` and echoes the path.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]

from ai_trading.config.management import get_env, reload_env
from ai_trading.utils.env import get_alpaca_data_base_url, get_alpaca_http_headers


def _ensure_credentials() -> None:
    """Raise ``RuntimeError`` if Alpaca credentials are missing."""

    api_key = get_env("ALPACA_API_KEY", default=None)
    secret_key = get_env("ALPACA_SECRET_KEY", default=None)
    data_key = get_env("ALPACA_DATA_API_KEY", default=None)
    data_secret = get_env("ALPACA_DATA_SECRET_KEY", default=None)
    if not any((api_key, data_key)):
        raise RuntimeError("No Alpaca API credentials detected")
    if api_key and not secret_key:
        raise RuntimeError("ALPACA_API_KEY present but ALPACA_SECRET_KEY missing")
    if data_key and not data_secret:
        raise RuntimeError("ALPACA_DATA_API_KEY present but ALPACA_DATA_SECRET_KEY missing")


def _build_params(symbol: str, timeframe: str, minutes: int, adjustment: str) -> dict[str, Any]:
    """Return query parameters for the bars request."""

    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(minutes=max(minutes, 1))
    return {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": now.isoformat(),
        "limit": 10000,
        "feed": get_env("ALPACA_DATA_FEED", default="iex"),
        "adjustment": adjustment,
    }


def _capture_payload(symbol: str, timeframe: str, minutes: int, adjustment: str) -> Path:
    """Request Alpaca bars and persist the full payload for analysis."""

    reload_env()
    _ensure_credentials()
    base_url = get_alpaca_data_base_url().rstrip("/")
    url = f"{base_url}/v2/stocks/bars"
    headers = get_alpaca_http_headers()
    params = _build_params(symbol, timeframe, minutes, adjustment)
    response = requests.get(url, params=params, headers=headers, timeout=15)
    payload: Any
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw_text": response.text}

    snapshot = {
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "status_code": response.status_code,
        "url": response.url,
        "params": params,
        "headers_used": {k: headers.get(k) for k in ("APCA-API-KEY-ID", "APCA-API-SECRET-KEY", "Authorization") if k in headers},
        "payload": payload,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"alpaca_payload_{symbol}_{timestamp}.json"
    output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AVGO", help="Symbol to capture (default: AVGO)")
    parser.add_argument("--timeframe", default="1Min", help="Timeframe to request (default: 1Min)")
    parser.add_argument("--minutes", type=int, default=90, help="Minutes of history to request (default: 90)")
    parser.add_argument(
        "--adjustment",
        default="raw",
        choices=("raw", "split", "dividend", "all"),
        help="Adjustment mode for Alpaca bars (default: raw)",
    )
    args = parser.parse_args()

    try:
        path = _capture_payload(args.symbol.upper(), args.timeframe, args.minutes, args.adjustment)
    except Exception as exc:
        raise SystemExit(f"Failed to capture Alpaca payload: {exc}") from exc

    print(f"Captured Alpaca payload to {path}")


if __name__ == "__main__":
    main()
