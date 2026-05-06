from __future__ import annotations

from types import SimpleNamespace

from ai_trading.risk.short_selling import (
    borrowable_share_count,
    is_asset_borrowable,
    is_short_side,
    normalize_short_side,
)


def test_short_side_aliases_normalize_to_sell_short() -> None:
    for side in ("short", "sellshort", "sell-short", "short_sell", "sell to open"):
        assert normalize_short_side(side) == "sell_short"
        assert is_short_side(side) is True


def test_borrowability_helper_respects_flags_and_share_count() -> None:
    asset = SimpleNamespace(shortable=True, easy_to_borrow=True, shortable_shares="25")

    assert borrowable_share_count(asset) == 25
    assert is_asset_borrowable(asset, qty=25) is True
    assert is_asset_borrowable(asset, qty=26) is False
    assert is_asset_borrowable({"shortable": False, "easy_to_borrow": True}, qty=1) is False
    assert is_asset_borrowable({"shortable": True, "easy_to_borrow": False}, qty=1) is False
    assert is_asset_borrowable({}, qty=1) is False
