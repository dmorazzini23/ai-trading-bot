from __future__ import annotations

from ai_trading.logging.normalize import (
    canon_feed,
    canon_timeframe,
    normalize_extra,
)


def _fn_stub():  # AI-AGENT-REF: stub function to mimic leakage
    pass


def test_canon_timeframe_basic():
    assert canon_timeframe("1m") == "1Min"
    assert canon_timeframe("1Min") == "1Min"
    assert canon_timeframe("1day") == "1Day"
    assert canon_timeframe("1Day") == "1Day"


def test_canon_timeframe_odd_values():
    # AI-AGENT-REF: odd inputs fallback to Day
    assert canon_timeframe(_fn_stub) == "1Day"
    assert canon_timeframe(None) == "1Day"


def test_canon_feed_basic():
    assert canon_feed("IEX") == "iex"
    assert canon_feed("sip") == "sip"


def test_normalize_extra_applies_canonicalization():
    e = normalize_extra({"feed": _fn_stub, "timeframe": _fn_stub})
    assert e["feed"] in {"iex", "sip"}
    assert e["timeframe"] in {"1Min", "1Day"}

