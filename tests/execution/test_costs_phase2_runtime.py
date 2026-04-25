from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ai_trading.execution import costs as costs_mod


def test_symbol_costs_execution_and_holding_components() -> None:
    costs = costs_mod.SymbolCosts(
        symbol="SPY",
        half_spread_bps=1.5,
        slip_k=2.0,
        commission_bps=0.25,
        borrow_fee_bps=12.0,
        overnight_bps=0.5,
    )

    assert costs.total_cost_bps == 3.25
    assert costs.slippage_cost_bps(4.0) == pytest.approx(4.0)
    assert costs.slippage_cost_bps(-1.0) == pytest.approx(2.0 * (0.1**0.5))
    assert costs.total_execution_cost_bps(4.0) == pytest.approx(7.25)
    assert costs.borrow_cost_bps(2.0) == 24.0
    assert costs.overnight_cost_bps(2.0) == 1.0
    assert costs.total_holding_cost_bps(2.0, is_short=True) == 25.0
    assert costs.updated_at.tzinfo is UTC


def test_symbol_cost_model_loads_saves_and_defaults(tmp_path: Path) -> None:
    cost_file = tmp_path / "symbol_costs.json"
    cost_file.write_text(
        """
        {
          "AAPL": {
            "symbol": "AAPL",
            "half_spread_bps": 1.0,
            "slip_k": 1.2,
            "commission_bps": 0.1,
            "min_commission": 0.5,
            "borrow_fee_bps": 5.0,
            "overnight_bps": 0.2,
            "locate_required": false,
            "updated_at": "2026-04-20T12:00:00+00:00",
            "sample_count": 3
          }
        }
        """,
        encoding="utf-8",
    )

    model = costs_mod.SymbolCostModel(str(tmp_path))

    assert model.get_costs("aapl").sample_count == 3
    default = model.get_costs("msft")
    assert default.symbol == "MSFT"
    assert "MSFT" in model._costs

    model._save_cost_data()
    persisted = cost_file.read_text(encoding="utf-8")
    assert '"MSFT"' in persisted


def test_update_costs_adjusts_spread_or_slippage_and_persists(tmp_path: Path) -> None:
    model = costs_mod.SymbolCostModel(str(tmp_path))
    initial = model.get_costs("SPY")
    initial_spread = initial.half_spread_bps
    initial_slip = initial.slip_k

    model.update_costs("SPY", realized_cost_bps=20.0, volume_ratio=0.5, learning_rate=0.2)
    spread_updated = model.get_costs("SPY")

    assert spread_updated.half_spread_bps > initial_spread
    assert spread_updated.slip_k == initial_slip
    assert spread_updated.sample_count == 1

    model.update_costs("SPY", realized_cost_bps=40.0, volume_ratio=4.0, learning_rate=0.2)
    slip_updated = model.get_costs("SPY")

    assert slip_updated.slip_k > spread_updated.slip_k
    assert slip_updated.sample_count == 2
    assert (tmp_path / "symbol_costs.json").exists()


def test_position_impact_and_size_adjustment_respect_minimums_and_scaling(tmp_path: Path) -> None:
    model = costs_mod.SymbolCostModel(str(tmp_path))
    model._costs["SPY"] = costs_mod.SymbolCosts(
        symbol="SPY",
        half_spread_bps=5.0,
        slip_k=8.0,
        commission_bps=1.0,
        min_commission=2.0,
    )

    tiny = model.calculate_position_impact("SPY", position_value=100.0, volume_ratio=1.0)
    assert tiny["cost_dollars"] == 2.0
    assert tiny["effective_bps"] == 200.0

    adjusted, cost_info = model.adjust_position_size(
        "SPY",
        target_size=1_000.0,
        max_cost_bps=10.0,
        volume_ratio=4.0,
    )

    assert adjusted < 1_000.0
    assert cost_info["original_size"] == 1_000.0
    assert 0.0 < cost_info["scaling_factor"] < 1.0

    unchanged, unchanged_cost = model.adjust_position_size("SPY", 0.0)
    assert unchanged == 0.0
    assert unchanged_cost == {}


def test_short_availability_and_holding_cost_adjustments(tmp_path: Path) -> None:
    model = costs_mod.SymbolCostModel(str(tmp_path))
    model._costs["EASY"] = costs_mod.SymbolCosts(
        symbol="EASY",
        half_spread_bps=1.0,
        slip_k=1.0,
        borrow_fee_bps=10.0,
        overnight_bps=0.5,
        locate_required=True,
    )
    model._costs["HARD"] = costs_mod.SymbolCosts(
        symbol="HARD",
        half_spread_bps=1.0,
        slip_k=1.0,
        borrow_fee_bps=75.0,
        overnight_bps=0.5,
        locate_required=True,
    )

    assert model.check_short_availability("EASY") == (True, "Available")
    assert model.check_short_availability("HARD") == (False, "Hard to borrow - high fee")

    rejected, rejected_info = model.adjust_for_holding_costs(
        "HARD",
        target_size=-500.0,
        expected_holding_days=2.0,
        is_short=True,
    )
    assert rejected == 0.0
    assert rejected_info == {"rejected": True, "reason": "Hard to borrow - high fee"}

    scaled, holding_info = model.adjust_for_holding_costs(
        "EASY",
        target_size=-1_000.0,
        expected_holding_days=2.0,
        max_holding_cost_bps=5.0,
        is_short=True,
    )
    assert -1_000.0 < scaled < 0.0
    assert holding_info["original_size"] == -1_000.0
    assert holding_info["scaling_factor"] == pytest.approx(5.0 / 21.0)


def test_snapshot_statistics_and_global_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = costs_mod.SymbolCostModel(str(tmp_path))
    model._costs["AAPL"] = costs_mod.SymbolCosts("AAPL", 1.0, 1.2)
    model._costs["MSFT"] = costs_mod.SymbolCosts("MSFT", 2.0, 2.2)
    monkeypatch.setattr(costs_mod, "_global_cost_model", model)
    monkeypatch.setattr(costs_mod.pd.DataFrame, "to_parquet", lambda self, path, compression=None: Path(path).write_text("ok"))

    stats = model.get_cost_statistics()
    assert stats["num_symbols"] == 2
    assert stats["avg_spread_bps"] == pytest.approx(3.0)

    snapshot = model.save_daily_snapshot(date(2026, 4, 20))
    assert snapshot.endswith("20260420.parquet")

    assert costs_mod.get_cost_model() is model
    assert costs_mod.get_symbol_costs("AAPL").symbol == "AAPL"
    assert costs_mod.calculate_execution_cost("AAPL", 1_000.0)["cost_bps"] > 0.0
