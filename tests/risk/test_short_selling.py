from __future__ import annotations

from types import SimpleNamespace

from ai_trading.risk.short_selling import is_short_side, normalize_short_side


def test_short_side_aliases_normalize_to_sell_short() -> None:
    for side in ("short", "sellshort", "sell-short", "short_sell", "sell to open"):
        assert normalize_short_side(side) == "sell_short"
        assert is_short_side(side) is True


