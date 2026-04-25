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


def test_adjustment_factors_and_bars_use_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    registry = corp_actions.CorporateActionRegistry(str(tmp_path))
    registry.add_action("AAPL", "2020-08-31", "split", 4.0)
    monkeypatch.setattr(corp_actions, "_global_registry", registry)

    price_factor, volume_factor = registry.get_adjustment_factors(
        "AAPL",
        reference_date=date(2020, 9, 2),
        target_date=date(2020, 8, 28),
    )
    forward_price_factor, forward_volume_factor = registry.get_adjustment_factors(
        "AAPL",
        reference_date=date(2020, 8, 28),
        target_date=date(2020, 9, 2),
    )

    assert (price_factor, volume_factor) == pytest.approx((4.0, 0.25))
    assert (forward_price_factor, forward_volume_factor) == pytest.approx((0.25, 4.0))
    assert registry.get_adjustment_factors("AAPL", date(2020, 9, 2), date(2020, 9, 2)) == (
        1.0,
        1.0,
    )

    bars = pd.DataFrame(
        {
            "open": [100.0, 200.0],
            "close": [101.0, 201.0],
            "volume": [1_000.0, 2_000.0],
            "note": ["pre", "post"],
        },
        index=pd.to_datetime(["2020-08-28", "2020-09-02"]),
    )

    adjusted = corp_actions.adjust_bars(bars, "AAPL", reference_date=date(2020, 9, 2))

    assert adjusted.loc[pd.Timestamp("2020-08-28"), "open"] == pytest.approx(400.0)
    assert adjusted.loc[pd.Timestamp("2020-08-28"), "volume"] == pytest.approx(250.0)
    assert adjusted.loc[pd.Timestamp("2020-09-02"), "close"] == pytest.approx(201.0)
    assert adjusted.loc[pd.Timestamp("2020-08-28"), "note"] == "pre"
    assert corp_actions.apply_adjustment_factor(10.0, 0.25) == pytest.approx(2.5)


def test_populate_common_splits_uses_global_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    registry = corp_actions.CorporateActionRegistry(str(tmp_path))
    monkeypatch.setattr(corp_actions, "_global_registry", registry)

    corp_actions.populate_common_splits()

    assert {action.symbol for action in registry.get_actions("TSLA")} == {"TSLA"}
    assert registry.get_actions("AAPL")[0].ratio == 4.0
    assert registry.get_actions("NVDA")[0].source == "manual"
