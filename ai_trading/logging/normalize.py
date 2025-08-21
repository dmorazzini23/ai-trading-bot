"""Central helpers to canonicalize logging payload fields."""  # AI-AGENT-REF: shared normalization utilities

from __future__ import annotations

from collections.abc import Mapping, MutableMapping  # AI-AGENT-REF: type hints for helpers
from typing import Any


def _as_lower_str(value: Any) -> str:
    try:
        return str(value).strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def canon_timeframe(value: Any) -> str:
    """Return canonical timeframe string: "1Min" or "1Day".

    Accepts strings, enums, or odd callables (from stubs). Defaults to "1Day".
    """  # AI-AGENT-REF: unify timeframe normalization

    s = _as_lower_str(value)
    if s in {"1min", "1m", "minute", "1 minute"}:
        return "1Min"
    if s in {"1day", "1d", "day", "1 day"}:
        return "1Day"
    if "min" in s:
        return "1Min"
    if "day" in s:
        return "1Day"
    return "1Day"


def canon_feed(value: Any) -> str:
    """Return canonical feed: "iex" or "sip". Defaults to "sip" on ambiguity."""  # AI-AGENT-REF: unify feed normalization

    s = _as_lower_str(value)
    if "iex" in s:
        return "iex"
    if "sip" in s:
        return "sip"
    return "sip"


def normalize_extra(extra: Mapping[str, Any] | None) -> dict:
    """Return a copy of `extra` with canonical feed/timeframe if present."""  # AI-AGENT-REF: ensure logging extras use canonical values

    if extra is None:
        return {}
    out: MutableMapping[str, Any] = dict(extra)
    if "feed" in out:
        out["feed"] = canon_feed(out["feed"])  # type: ignore[index]
    if "timeframe" in out:
        out["timeframe"] = canon_timeframe(out["timeframe"])  # type: ignore[index]
    return dict(out)

