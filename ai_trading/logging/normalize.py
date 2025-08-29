"""Central helpers to canonicalize logging payload fields."""
from __future__ import annotations
from collections.abc import Mapping, MutableMapping
from typing import Any

def _as_lower_str(value: Any) -> str:
    try:
        return str(value).strip().lower()
    except (KeyError, ValueError, TypeError):
        return ''

def canon_timeframe(value: Any) -> str:
    """Return canonical timeframe string: "1Min" or "1Day".

    Accepts strings, enums, or odd callables (from stubs). Defaults to "1Day".
    """
    s = _as_lower_str(value)
    if s in {'1min', '1m', 'minute', '1 minute'}:
        return '1Min'
    if s in {'1day', '1d', 'day', '1 day'}:
        return '1Day'
    if 'min' in s:
        return '1Min'
    if 'day' in s:
        return '1Day'
    return '1Day'

def canon_feed(value: Any) -> str:
    """Return canonical feed: "iex" or "sip". Defaults to "sip" on ambiguity."""
    s = _as_lower_str(value)
    if 'iex' in s:
        return 'iex'
    if 'sip' in s:
        return 'sip'
    return 'sip'

def canon_symbol(value: Any) -> str:
    """Return canonical stock symbol for Alpaca REST calls.

    Alpaca expects class share separators to use dots rather than dashes
    (e.g., ``BRK.B``).  This helper normalizes incoming symbols by
    uppercasing and replacing a single dash with a dot when present.  Any
    non-string input results in an empty string.
    """
    try:
        sym = str(value).strip().upper()
    except (KeyError, ValueError, TypeError):
        return ''
    if '-' in sym:
        parts = sym.split('-')
        if len(parts) == 2 and all(parts):
            sym = '.'.join(parts)
    return sym

def normalize_extra(extra: Mapping[str, Any] | None) -> dict:
    """Return a copy of `extra` with canonical feed/timeframe if present."""
    if extra is None:
        return {}
    out: MutableMapping[str, Any] = dict(extra)
    if 'feed' in out:
        out['feed'] = canon_feed(out['feed'])
    if 'timeframe' in out:
        out['timeframe'] = canon_timeframe(out['timeframe'])
    return dict(out)
