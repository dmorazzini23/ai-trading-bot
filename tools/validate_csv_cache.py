#!/usr/bin/env python3.12
"""Validate cached CSV files in the data directory."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable
import concurrent.futures

import pandas as pd

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
DATA_DIR = Path("data")
LOG_FILE = Path("logs") / "data_validation.log"


def load_tickers(path: Path) -> set[str]:
    """Return a set of ticker symbols from ``path``."""
    tickers: set[str] = set()
    if not path.exists():
        return tickers
    with path.open() as f:
        for line in f:
            symbol = line.strip()
            if symbol and symbol.lower() != "symbol":
                tickers.add(symbol)
    return tickers


def find_csv_files(directory: Path, tickers: Iterable[str]) -> list[Path]:
    """Return CSV files under ``directory`` whose stem matches ``tickers``."""
    ticker_set = set(tickers)
    return [p for p in directory.rglob("*.csv") if p.stem in ticker_set]


def validate_csv(path: Path) -> tuple[bool, str]:
    """Validate CSV ``path`` and return (is_valid, reason)."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"read error: {exc}"

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"missing columns: {', '.join(missing)}"

    for col in REQUIRED_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"non-numeric values in {col}"

    if df[REQUIRED_COLS].isnull().any().any():
        return False, "NaN values detected"

    return True, ""


def setup_logger(enable: bool) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger("csv_validator")
    logger.setLevel(logging.INFO)
    if enable:
        LOG_FILE.parent.mkdir(exist_ok=True)
        handler = logging.FileHandler(LOG_FILE)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main() -> None:  # pragma: no cover - CLI utility
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Validate cached CSV files")
    parser.add_argument("--repair", action="store_true", help="Delete invalid files")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics")
    parser.add_argument("--log", action="store_true", help="Log issues to file")
    args = parser.parse_args()

    logger = setup_logger(args.log)

    tickers = load_tickers(Path("tickers.csv"))
    csv_files = find_csv_files(DATA_DIR, tickers)

    valid_count = 0
    invalid_count = 0

    with concurrent.futures.ProcessPoolExecutor() as pool:
        results = list(pool.map(validate_csv, csv_files))

    for csv_path, (valid, reason) in zip(csv_files, results):
        if valid:
            valid_count += 1
        else:
            invalid_count += 1
            msg = f"INVALID: {csv_path} - {reason}"
            print(msg)
            if args.log:
                logger.warning(msg)
            if args.repair:
                try:
                    csv_path.unlink()
                    del_msg = f"Deleted {csv_path}"
                    print(del_msg)
                    if args.log:
                        logger.info(del_msg)
                except OSError as exc:  # pragma: no cover - disk issues
                    err_msg = f"Failed to delete {csv_path}: {exc}"
                    print(err_msg)
                    if args.log:
                        logger.error(err_msg)

    if args.summary:
        summary = [
            f"\N{white heavy check mark} {valid_count} valid files",
            f"\N{warning sign} {invalid_count} invalid files" + (
                " (auto-repaired)" if args.repair and invalid_count else ""
            ),
        ]
        print("\n".join(summary))
        if args.log:
            print(f"Logs saved to {LOG_FILE}")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
