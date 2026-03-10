"""Utility helpers for symbol/universe loading."""

from __future__ import annotations

import csv
from pathlib import Path

# Mapping for data providers that expect alternate share-class separators.
_SYMBOL_FIXES: dict[str, str] = {}
_HEADER_TOKENS = {"symbol", "symbols", "ticker", "tickers"}


def normalize_symbol(symbol: str) -> str:
    """Return uppercase symbol with provider-specific fixes applied."""
    up = symbol.strip().upper()
    return _SYMBOL_FIXES.get(up, up)


def load_universe(path_or_csv: str | None, limit: int | None = None) -> list[str]:
    """Load symbols from file path or CSV string and return sanitized list."""
    raw: list[str] = []
    if path_or_csv:
        p = Path(path_or_csv)
        if p.exists() and p.is_file():
            raw = _symbols_from_file(p)
        else:
            raw = _split_symbols(path_or_csv)
    if not raw:
        raise RuntimeError(
            f"No tickers found at '{path_or_csv}'. Provide a tickers.csv or CSV env list."
        )
    seen: set[str] = set()
    out: list[str] = []
    for sym in raw:
        up = normalize_symbol(sym)
        if up and up not in seen:
            seen.add(up)
            out.append(up)
    if limit and limit > 0:
        out = out[:limit]
    return out


def _symbols_from_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows: list[list[str]] = []
    for row in csv.reader(text.splitlines()):
        normalized_row = [cell.strip() for cell in row if cell is not None]
        if any(cell for cell in normalized_row):
            rows.append(normalized_row)
    if not rows:
        return []

    first_cell = rows[0][0].strip().lower() if rows[0] and rows[0][0] else ""
    if first_cell in _HEADER_TOKENS:
        data_rows = rows[1:]
        return [row[0].strip() for row in data_rows if row and row[0].strip()]

    # Preserve comma-separated env-style payloads saved as one-line files.
    if len(rows) == 1 and len(rows[0]) > 1:
        return [cell.strip() for cell in rows[0] if cell.strip()]

    # Default CSV interpretation: first column contains symbols.
    return [row[0].strip() for row in rows if row and row[0].strip()]


def _split_symbols(s: str) -> list[str]:
    s = s.replace("\r", "")
    parts: list[str] = []
    for line in s.split("\n"):
        if "," in line:
            parts.extend((x.strip() for x in line.split(",")))
        else:
            parts.append(line.strip())
    return [p for p in parts if p]


__all__ = ["load_universe", "normalize_symbol"]
