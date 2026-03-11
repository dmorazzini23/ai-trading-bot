from __future__ import annotations

from ai_trading.core.bot_engine import _clip_delta_to_symbol_notional_cap


def test_symbol_notional_cap_clip_reduces_requested_delta() -> None:
    adjusted, details = _clip_delta_to_symbol_notional_cap(
        symbol="MSFT",
        current_shares=200,
        delta_shares=80,
        price=100.0,
        max_symbol_notional=25_000.0,
    )
    assert adjusted == 50
    assert details is not None
    assert details["requested_delta_shares"] == 80
    assert details["adjusted_delta_shares"] == 50
    assert details["adjusted_projected_notional"] <= details["max_symbol_notional"]


def test_symbol_notional_cap_blocks_worsening_when_already_over_cap() -> None:
    adjusted, details = _clip_delta_to_symbol_notional_cap(
        symbol="MSFT",
        current_shares=300,
        delta_shares=20,
        price=100.0,
        max_symbol_notional=25_000.0,
    )
    assert adjusted == 0
    assert details is not None
    assert details["current_notional"] > details["max_symbol_notional"]


def test_symbol_notional_cap_allows_derisking_from_over_cap() -> None:
    adjusted, details = _clip_delta_to_symbol_notional_cap(
        symbol="MSFT",
        current_shares=300,
        delta_shares=-40,
        price=100.0,
        max_symbol_notional=25_000.0,
    )
    assert adjusted == -40
    assert details is None
