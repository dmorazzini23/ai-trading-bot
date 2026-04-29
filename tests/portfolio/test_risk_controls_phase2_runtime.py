from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.portfolio import risk_controls


def _returns_frame(rows: int = 80) -> Any:
    return pd.DataFrame(
        {
            "AAPL": [0.01, -0.01] * (rows // 2),
            "MSFT": [0.008, -0.008] * (rows // 2),
            "TLT": [-0.006, 0.006] * (rows // 2),
        }
    )


def test_turnover_budget_tracks_remaining_and_rejects_over_budget() -> None:
    budget = risk_controls.TurnoverBudget(total_budget=0.5, remaining_turnover=0.5)

    assert budget.add_trade(10_000.0, 100_000.0) is True
    assert budget.used_turnover == pytest.approx(0.1)
    assert budget.remaining_turnover == pytest.approx(0.4)
    assert budget.add_trade(-10_000.0, 100_000.0) is True
    assert budget.used_turnover == pytest.approx(0.2)
    assert budget.remaining_turnover == pytest.approx(0.3)
    assert budget.add_trade(50_000.0, 100_000.0) is False
    assert budget.add_trade(1.0, 0.0) is False


def test_adaptive_risk_controller_vol_kelly_drawdown_and_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = risk_controls.AdaptiveRiskController(
        risk_controls.RiskBudget(max_position_risk=0.02, kelly_multiplier=0.5)
    )
    frame = _returns_frame()

    vols = controller.calculate_volatilities(frame, lookback_days=20)
    assert set(vols) == {"AAPL", "MSFT", "TLT"}
    assert all(value > 0 for value in vols.values())

    short_vols = controller.calculate_volatilities(pd.DataFrame({"NEW": [0.01, -0.01]}), lookback_days=20)
    assert short_vols == {"NEW": 0.2}

    monkeypatch.setattr(risk_controls, "_import_clustering", lambda: (None, None, None, False))
    assert controller.calculate_correlation_clusters(frame) == {"AAPL": 0, "MSFT": 0, "TLT": 0}

    kelly = controller.calculate_kelly_fractions(
        {"AAPL": 0.02, "MSFT": -0.01, "NEW": 0.1, "ZERO": 0.1},
        {"AAPL": 0.2, "MSFT": 0.2, "ZERO": 0.0},
    )
    assert kelly["AAPL"] > 0.0
    assert kelly["MSFT"] < 0.0
    assert kelly["NEW"] == 0.0
    assert kelly["ZERO"] == 0.0

    controller.update_drawdown_governor(-0.2, is_positive_day=False)
    assert controller.drawdown_multiplier == 0.5
    for _ in range(5):
        controller.update_drawdown_governor(0.01, is_positive_day=True)
    assert controller.drawdown_multiplier > 0.5
    controller.update_drawdown_governor(0.0, is_positive_day=False)
    assert controller.green_days_count == 0


def test_cluster_limits_and_position_size_turnover_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = risk_controls.AdaptiveRiskController(
        risk_controls.RiskBudget(max_cluster_risk=0.15, max_turnover_daily=0.05)
    )
    positions = {
        "AAPL": risk_controls.PositionRisk("AAPL", 10_000.0, 0.01, 0.1, 0.1),
        "MSFT": risk_controls.PositionRisk("MSFT", 10_000.0, 0.01, 0.1, 0.1),
    }

    clusters = controller.check_cluster_limits(positions, {"AAPL": 1, "MSFT": 1})
    assert clusters[0].is_over_limit is True
    assert clusters[0].symbols == ["AAPL", "MSFT"]

    monkeypatch.setattr(
        controller,
        "calculate_volatilities",
        lambda _returns_data: {"AAPL": 0.2, "MSFT": 0.2},
    )
    monkeypatch.setattr(
        controller,
        "calculate_correlation_clusters",
        lambda _returns_data: {"AAPL": 0, "MSFT": 1},
    )
    monkeypatch.setattr(
        controller,
        "calculate_kelly_fractions",
        lambda _expected, _vols: {"AAPL": 0.1, "MSFT": 0.1},
    )

    targets = controller.calculate_position_sizes(
        {"AAPL": 1.0, "MSFT": 1.0},
        _returns_frame(),
        portfolio_value=100_000.0,
        current_positions={"AAPL": 0.0, "MSFT": 0.0},
    )

    assert targets["AAPL"] == pytest.approx(5_000.0)
    assert targets["MSFT"] == pytest.approx(0.0)
    assert controller.turnover_budget.remaining_turnover == pytest.approx(0.0)


def test_position_sizing_resets_stale_daily_turnover_before_budgeting(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = risk_controls.AdaptiveRiskController(
        risk_controls.RiskBudget(max_turnover_daily=0.05)
    )
    controller.turnover_budget = risk_controls.TurnoverBudget(
        date=datetime.now(UTC).date() - timedelta(days=1),
        used_turnover=0.05,
        remaining_turnover=0.0,
        total_budget=0.05,
    )
    monkeypatch.setattr(
        controller,
        "calculate_volatilities",
        lambda _returns_data: {"AAPL": 0.2},
    )
    monkeypatch.setattr(
        controller,
        "calculate_correlation_clusters",
        lambda _returns_data: {"AAPL": 0},
    )
    monkeypatch.setattr(
        controller,
        "calculate_kelly_fractions",
        lambda _expected, _vols: {"AAPL": 0.1},
    )

    targets = controller.calculate_position_sizes(
        {"AAPL": 1.0},
        _returns_frame(),
        portfolio_value=100_000.0,
        current_positions={"AAPL": 0.0},
    )

    assert controller.turnover_budget.date == datetime.now(UTC).date()
    assert targets["AAPL"] == pytest.approx(5_000.0)


def test_adaptive_risk_controller_preserves_short_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = risk_controls.AdaptiveRiskController(
        risk_controls.RiskBudget(max_turnover_daily=1.0)
    )
    monkeypatch.setattr(
        controller,
        "calculate_volatilities",
        lambda _returns_data: {"AAPL": 0.2, "MSFT": 0.2},
    )
    monkeypatch.setattr(
        controller,
        "calculate_correlation_clusters",
        lambda _returns_data: {"AAPL": 0, "MSFT": 1},
    )

    targets = controller.calculate_position_sizes(
        {"AAPL": 1.0, "MSFT": -1.0},
        _returns_frame(),
        portfolio_value=100_000.0,
        current_positions={"AAPL": 0.0, "MSFT": 0.0},
    )

    assert targets["AAPL"] > 0
    assert targets["MSFT"] < 0


def test_reset_summary_and_global_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = risk_controls.AdaptiveRiskController()
    old_date = datetime.now(UTC).date() - timedelta(days=1)
    controller.turnover_budget = risk_controls.TurnoverBudget(
        date=old_date,
        used_turnover=0.5,
        remaining_turnover=0.5,
        total_budget=1.0,
    )
    controller._cluster_assignments = {"AAPL": 1, "MSFT": 2}
    controller._volatilities = {"AAPL": 0.2}
    monkeypatch.setattr(risk_controls, "_global_risk_controller", controller)

    controller.reset_daily_budget()
    summary = controller.get_risk_summary()

    assert controller.turnover_budget.date == datetime.now(UTC).date()
    assert summary["num_clusters"] == 2
    assert summary["num_symbols"] == 1
    assert risk_controls.get_risk_controller() is controller

    monkeypatch.setattr(
        controller,
        "calculate_position_sizes",
        lambda signals, returns_data, portfolio_value, current_positions=None: {
            "called": float(portfolio_value)
        },
    )
    assert risk_controls.calculate_adaptive_positions({"AAPL": 1.0}, _returns_frame(), 123.0) == {
        "called": 123.0
    }
