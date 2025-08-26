from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
CANON = {'open': {'open', 'o', 'Open', 'OPEN'}, 'high': {'high', 'h', 'High', 'HIGH'}, 'low': {'low', 'l', 'Low', 'LOW'}, 'close': {'close', 'c', 'Close', 'CLOSE', 'adj_close', 'Adj Close'}, 'volume': {'volume', 'v', 'Volume', 'VOL', 'Vol'}}

def standardize_ohlcv(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Return a copy with canonical lowercase ['open','high','low','close','volume'] where possible.
    Unknown columns are preserved. Missing OHLCV fields are left absent (caller must handle).
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).strip() if not isinstance(c, str) else c.strip() for c in out.columns]
    lower_map = {}
    for canon, variants in CANON.items():
        for v in variants:
            if v in out.columns:
                lower_map[v] = canon
        for v in list(variants):
            if v.lower() in out.columns:
                lower_map[v.lower()] = canon
    if lower_map:
        out = out.rename(columns=lower_map)
    return out