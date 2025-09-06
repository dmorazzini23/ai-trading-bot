"""Compatibility wrapper around :mod:`ai_trading.analysis.sentiment`.

This module exposes a small helper that delegates to the analysis
sentiment utilities while keeping the public surface area minimal.
"""
from __future__ import annotations

from ai_trading.analysis import sentiment as _sentiment

# Backwards-compatibility alias; legacy code expected ``_C`` to exist.
# It now simply points at the underlying sentiment module. If unused it
# can be removed in a future cleanup.
_C = _sentiment


def analyze_text(text: str) -> dict[str, float]:
    """Return sentiment probabilities for *text*.

    Delegates to :func:`ai_trading.analysis.sentiment.analyze_text` and
    returns a dictionary with the keys ``available``, ``pos``, ``neg`` and
    ``neu``.
    """

    return _sentiment.analyze_text(text)


__all__ = ["analyze_text"]

