#!/usr/bin/env python
"""Fetch daily OHLCV CSVs into ./data using yfinance.

Writes data/<SYMBOL>.csv with at least a 'close' column suitable for the
optimizer tool. Defaults to 2y of daily data. Keeps footprint small per
AGENTS.md (no large artifacts).
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yfinance as yf  # type: ignore
except Exception as exc:  # pragma: no cover - require yfinance at runtime
    raise SystemExit("yfinance is required. pip install yfinance") from exc


def fetch_symbol(symbol: str, period: str = "2y", interval: str = "1d") -> Path:
    sym = symbol.upper()
    t = yf.Ticker(sym)
    df = t.history(period=period, interval=interval)
    if df is None or df.empty or ("Close" not in df.columns and "close" not in df.columns):
        raise SystemExit(f"No data for {sym}")
    out = df[["Close"]].rename(columns={"Close": "close"})
    out.index.name = "date"
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{sym}.csv"
    out.to_csv(path)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch CSVs via yfinance into ./data")
    p.add_argument("symbols", nargs="+", help="Symbols to fetch (e.g., SPY QQQ AAPL)")
    p.add_argument("--period", default="2y", help="yfinance period (default 2y)")
    p.add_argument("--interval", default="1d", help="yfinance interval (default 1d)")
    args = p.parse_args(argv)
    for sym in args.symbols:
        path = fetch_symbol(sym, args.period, args.interval)
        print({"symbol": sym.upper(), "path": str(path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

