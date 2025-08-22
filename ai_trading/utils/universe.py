from __future__ import annotations

from pathlib import Path


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
        up = sym.strip().upper()
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
            parts.extend(x.strip() for x in line.split(","))
        else:
            parts.append(line.strip())
    return [p for p in parts if p]
