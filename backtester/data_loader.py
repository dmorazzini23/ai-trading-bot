"""Utilities for loading and repairing cached OHLCV data."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from alpaca_trade_api import REST, TimeFrame
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _get_api() -> REST:
    """Return an Alpaca REST client using environment variables."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    return REST(api_key, secret_key, base_url)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _fetch_from_alpaca(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch daily OHLCV bars for ``symbol`` from Alpaca."""
    api = _get_api()
    bars = api.get_bars(
        symbol,
        TimeFrame.Day,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        adjustment="raw",
    ).df
    if bars is None or bars.empty:
        raise ValueError(f"No data returned for {symbol}")
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)
    df = bars.reset_index()
    rename_map = {
        "timestamp": "timestamp",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    df = df[[c for c in ["timestamp"] + REQUIRED_COLS if c in df.columns]]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Fetched data missing columns {missing} for {symbol}")
    return df


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except OSError as exc:  # pragma: no cover - disk error
        logger.warning("Failed to cache %s: %s", path, exc)


def load_symbol_data(symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    """Return daily OHLCV data for ``symbol`` with automatic repair and fetch."""
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / f"{symbol}.csv"
    df = pd.DataFrame()
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                logger.warning("%s missing columns %s; deleting", csv_path, missing)
                csv_path.unlink(missing_ok=True)
                df = pd.DataFrame()
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp"])
                return df[["timestamp"] + REQUIRED_COLS]
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            logger.error("Failed to read %s: %s", csv_path, exc)
            try:
                csv_path.unlink(missing_ok=True)
                logger.info("Deleted corrupted file %s", csv_path)
            except OSError:
                logger.error("Could not delete corrupted file %s", csv_path)
            df = pd.DataFrame()

    # Determine fetch range
    end_dt = end or datetime.now(datetime.UTC)
    start_dt = start or end_dt - timedelta(days=365 * 2)

    try:
        df = _fetch_from_alpaca(symbol, start_dt, end_dt)
    except Exception as exc:
        logger.error("Data fetch failed for %s: %s", symbol, exc)
        raise
    _save_csv(df, csv_path)
    return df[["timestamp"] + REQUIRED_COLS]
