from __future__ import annotations

from datetime import date
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import corp_actions


def test_corporate_action_factors_cover_supported_action_types() -> None:
    split = corp_actions.CorporateAction("AAPL", date(2020, 8, 31), "split", 4.0)
    dividend = corp_actions.CorporateAction(
        "MSFT",
        date(2024, 2, 1),
        "dividend",
        1.0,
        dividend_amount=0.75,
    )
    merger = corp_actions.CorporateAction("OLD", date(2024, 1, 1), "merger", 1.5)
    zero_merger = corp_actions.CorporateAction("OLD", date(2024, 1, 1), "merger", 0.0)
    spin = corp_actions.CorporateAction("SPIN", date(2024, 1, 1), "spin_off", 1.0)
    unknown = corp_actions.CorporateAction("UNK", date(2024, 1, 1), "other", 2.0)

    assert split.price_adjustment_factor == pytest.approx(0.25)
    assert split.volume_adjustment_factor == pytest.approx(4.0)
    assert dividend.price_adjustment_factor == 1.0
    assert dividend.volume_adjustment_factor == 1.0
    assert merger.price_adjustment_factor == pytest.approx(1.5)
    assert zero_merger.price_adjustment_factor == 1.0
    assert spin.price_adjustment_factor == 1.0
    assert unknown.volume_adjustment_factor == 1.0


def test_registry_persists_sorts_and_filters_actions(tmp_path: Any) -> None:
    registry = corp_actions.CorporateActionRegistry(str(tmp_path))

    registry.add_action("aapl", "2020-08-31", "split", 4.0, description="4-for-1")
    registry.add_action("AAPL", date(2014, 6, 9), "split", 7.0, description="7-for-1")
    registry.add_action("AAPL", date(2024, 2, 9), "dividend", 1.0, dividend_amount=0.24)

    all_actions = registry.get_actions("aapl")
    filtered = registry.get_actions("AAPL", date(2020, 1, 1), date(2021, 1, 1))
    reloaded = corp_actions.CorporateActionRegistry(str(tmp_path))

    assert [action.ex_date for action in all_actions] == [
        date(2014, 6, 9),
        date(2020, 8, 31),
        date(2024, 2, 9),
    ]
    assert [action.description for action in filtered] == ["4-for-1"]
    assert reloaded.get_actions("AAPL")[1].ratio == 4.0
    assert registry.get_actions("MSFT") == []




