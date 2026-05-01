from __future__ import annotations

from ai_trading.core.submit_runtime import _canonical_intent_side


def test_canonical_intent_side_rejects_unknown_values() -> None:
    assert _canonical_intent_side("hold") is None
    assert _canonical_intent_side("") is None


def test_canonical_intent_side_preserves_known_sides() -> None:
    assert _canonical_intent_side("buy") == "buy"
    assert _canonical_intent_side("cover") == "buy"
    assert _canonical_intent_side("sell") == "sell"
    assert _canonical_intent_side("short") == "sell_short"
