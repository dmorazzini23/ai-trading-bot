"""Utility helpers for symbol/universe loading."""

from __future__ import annotations

from pathlib import Path

# Mapping for data providers that expect alternate share-class separators.
# Yahoo Finance uses dashes instead of dots for class shares (e.g., ``BRK-B``).
_SYMBOL_FIXES: dict[str, str] = {"BRK.B": "BRK-B"}


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
            s = p.read_text(encoding="utf-8", errors="ignore")
            raw = _split_symbols(s)
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
    out.sort()
    if limit and limit > 0:
        out = out[:limit]
    return out


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

