from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.strategies import performance_allocator as pa


def _trade(
    *,
    pnl: float,
    return_pct: float,
    days_ago: int = 0,
    symbol: str = "SPY",
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "entry_price": 100.0,
        "exit_price": 100.0 + return_pct * 100.0,
        "quantity": 10,
        "pnl": pnl,
        "timestamp": datetime.now(UTC) - timedelta(days=days_ago),
        "success": pnl > 0,
        "return_pct": return_pct,
    }


def test_threshold_resolution_falls_back_to_settings_then_default(monkeypatch):
    monkeypatch.setattr(
        pa,
        "get_settings",
        lambda: SimpleNamespace(score_confidence_min="0.42", conf_threshold="0.75"),
    )
    assert pa._resolve_conf_threshold(None) == pytest.approx(0.42)

    monkeypatch.setattr(
        pa,
        "get_settings",
        lambda: SimpleNamespace(score_confidence_min="bad", conf_threshold="0.55"),
    )
    assert pa._resolve_conf_threshold(
        cast(Any, SimpleNamespace(score_confidence_min=2.0))
    ) == pytest.approx(0.55)

    monkeypatch.setattr(
        pa,
        "get_settings",
        lambda: SimpleNamespace(score_confidence_min="bad", conf_threshold=None),
    )
    assert pa._resolve_conf_threshold(
        cast(Any, SimpleNamespace(score_confidence_min="bad"))
    ) == 0.6


def test_confidence_multiplier_clamps_bad_inputs_and_thresholds():
    assert pa._compute_conf_multiplier(-0.1, 0.7, 2.0, 1.0) == 1.0
    assert pa._compute_conf_multiplier(0.8, 2.0, 0.5, 0.0) == 1.0
    assert pa._compute_conf_multiplier(1.0, 0.7, 1.8, 2.0) == pytest.approx(1.8)


def test_allocator_initializes_from_dataclass_and_dict():
    from_dataclass = pa.PerformanceBasedAllocator(pa.AllocatorConfig(score_confidence_min=0.8))
    from_dict = pa.PerformanceBasedAllocator({"performance_window_days": 7})

    assert from_dataclass.config["score_confidence_min"] == 0.8
    assert from_dict.window_days == 7


def test_score_to_weight_gates_once(monkeypatch):
    events: list[tuple[str, dict[str, object] | None]] = []
    monkeypatch.setattr(
        pa.logger,
        "info",
        lambda msg, *args, extra=None, **_kwargs: events.append((msg, extra)),
    )
    allocator = pa.PerformanceBasedAllocator({"score_confidence_min": 0.7})

    assert allocator.score_to_weight("bad") == 0.0  # type: ignore[arg-type]
    assert allocator.score_to_weight(0.1) == 0.0
    assert allocator.score_to_weight(0.9) == 0.9
    gate_events = [event for event in events if event[0] == "ALLOC_CONFIDENCE_GATE"]
    assert len(gate_events) == 1
    assert gate_events[0][1] == {
        "threshold": 0.7,
        "note": "candidates below this score are zero-weighted",
    }


def test_allocate_filters_empty_strategies_and_boosts_valid_weights(monkeypatch):
    monkeypatch.setattr(
        pa,
        "get_settings",
        lambda: SimpleNamespace(score_size_max_boost=1.5, score_size_gamma=1.0),
    )
    allocator = pa.PerformanceBasedAllocator()
    high = SimpleNamespace(confidence=0.95, weight="2.0")
    invalid_weight = SimpleNamespace(confidence=0.8, weight="bad")
    low = SimpleNamespace(confidence=0.1, weight=99.0)

    result = allocator.allocate(
        {"momentum": [high, invalid_weight, low], "empty": [], "none": None},
        cast(Any, SimpleNamespace(score_confidence_min=0.7)),
    )

    assert set(result) == {"momentum"}
    assert result["momentum"] == [high, invalid_weight]
    assert high.weight > 2.0
    assert invalid_weight.weight > 1.0


def test_record_trade_result_requires_fields_and_handles_bad_values():
    allocator = pa.PerformanceBasedAllocator()

    allocator.record_trade_result("bad", {"symbol": "SPY"})
    assert len(allocator.strategy_trades["bad"]) == 0

    allocator.record_trade_result(
        "bad-math",
        {
            "symbol": "SPY",
            "entry_price": 0.0,
            "exit_price": 1.0,
            "pnl": 1.0,
            "timestamp": datetime.now(UTC),
        },
    )
    assert len(allocator.strategy_trades["bad-math"]) == 0

    allocator.record_trade_result(
        "momentum",
        {
            "symbol": "SPY",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "quantity": 5,
            "pnl": 25.0,
            "timestamp": datetime.now(UTC),
        },
    )
    recorded = allocator.strategy_trades["momentum"][0]
    assert recorded["return_pct"] == pytest.approx(0.05)
    assert isinstance(recorded["recorded_at"], datetime)


def test_performance_score_defaults_recent_trade_floor_and_error_path():
    allocator = pa.PerformanceBasedAllocator(
        {"min_trades_threshold": 3, "performance_window_days": 5}
    )
    assert allocator._calculate_performance_score("missing") == 0.5

    allocator.strategy_trades["old"].extend(
        [_trade(pnl=10, return_pct=0.01, days_ago=30) for _ in range(3)]
    )
    assert allocator._calculate_performance_score("old") == 0.3

    allocator.strategy_trades["broken"].extend(
        [
            {"timestamp": datetime.now(UTC), "return_pct": "bad"},
            {"timestamp": datetime.now(UTC), "return_pct": 0.01},
            {"timestamp": datetime.now(UTC), "return_pct": 0.02},
        ]
    )
    assert allocator._calculate_performance_score("broken") == 0.3


def test_performance_score_uses_weighted_metrics_for_recent_trades():
    allocator = pa.PerformanceBasedAllocator(
        {"min_trades_threshold": 3, "performance_window_days": 10}
    )
    returns = [0.01, 0.02, -0.005, 0.03, 0.015]
    allocator.strategy_trades["momentum"].extend(
        [_trade(pnl=r * 1000, return_pct=r, days_ago=i) for i, r in enumerate(returns)]
    )

    score = allocator._calculate_performance_score("momentum")

    assert 0.0 <= score <= 1.0
    assert score > 0.3


def test_normalize_metric_and_scores_to_weights_edges():
    allocator = pa.PerformanceBasedAllocator({"allocation_temperature": 1.0})

    assert allocator._normalize_metric(1.0, 2.0, 2.0) == 0.5
    assert allocator._normalize_metric(-1.0, 0.0, 1.0) == 0.0
    assert allocator._normalize_metric(2.0, 0.0, 1.0) == 1.0
    assert allocator._scores_to_weights({}) == {}

    weights = allocator._scores_to_weights({"a": 0.1, "b": 0.9})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["b"] > weights["a"]


def test_apply_allocation_bounds_caps_and_redistributes():
    allocator = pa.PerformanceBasedAllocator(
        {"min_allocation_pct": 0.2, "max_allocation_pct": 0.5}
    )

    bounded = allocator._apply_allocation_bounds({"a": 0.9, "b": 0.05, "c": 0.05})

    assert sum(bounded.values()) == pytest.approx(1.0)
    assert all(0.0 <= weight <= 0.5 for weight in bounded.values())
    assert allocator._apply_allocation_bounds({}) == {}


def test_calculate_strategy_allocations_handles_empty_and_error_fallback(monkeypatch):
    allocator = pa.PerformanceBasedAllocator()

    assert allocator.calculate_strategy_allocations([], 1000.0) == {}

    monkeypatch.setattr(
        allocator,
        "_calculate_performance_score",
        lambda _strategy: (_ for _ in ()).throw(ValueError("bad score")),
    )
    result = allocator.calculate_strategy_allocations(["a", "b"], 1000.0)
    assert result == {"a": 500.0, "b": 500.0}


def test_calculate_strategy_allocations_updates_state(monkeypatch):
    allocator = pa.PerformanceBasedAllocator({"max_allocation_pct": 0.9})
    monkeypatch.setattr(
        allocator,
        "_calculate_performance_score",
        lambda strategy: {"a": 0.9, "b": 0.2}[strategy],
    )

    result = allocator.calculate_strategy_allocations(["a", "b"], 1000.0)

    assert sum(result.values()) == pytest.approx(1000.0)
    assert result["a"] > result["b"]
    assert sum(allocator.strategy_allocations.values()) == pytest.approx(1.0)


def test_performance_report_empty_windows_and_error_path(monkeypatch):
    allocator = pa.PerformanceBasedAllocator({"min_trades_threshold": 3})
    assert allocator.get_strategy_performance_report("missing") == {
        "strategy": "missing",
        "error": "No trade history available",
    }

    allocator.strategy_trades["momentum"].extend(
        [
            _trade(pnl=10, return_pct=0.01, days_ago=1),
            _trade(pnl=-5, return_pct=-0.005, days_ago=2),
            _trade(pnl=20, return_pct=0.02, days_ago=3),
        ]
    )
    allocator.strategy_allocations["momentum"] = 0.25
    report = allocator.get_strategy_performance_report("momentum")
    assert report["strategy"] == "momentum"
    assert report["total_trades"] == 3
    assert report["current_allocation"] == 0.25
    assert report["windows"]["5d"]["trades"] == 3.0

    monkeypatch.setattr(
        allocator,
        "_calculate_performance_score",
        lambda _strategy: (_ for _ in ()).throw(ValueError("bad report")),
    )
    error_report = allocator.get_strategy_performance_report("momentum")
    assert error_report["error"].startswith("Report generation failed:")


def test_should_rebalance_time_rank_change_and_error_paths(monkeypatch):
    allocator = pa.PerformanceBasedAllocator()

    allocator.last_update = datetime.now(UTC) - timedelta(hours=25)
    assert allocator.should_rebalance_allocations() is True

    allocator.last_update = datetime.now(UTC)
    allocator.strategy_allocations = {"a": 0.8, "b": 0.2}
    monkeypatch.setattr(
        allocator,
        "_calculate_performance_score",
        lambda strategy: {"a": 0.1, "b": 0.9}[strategy],
    )
    assert allocator.should_rebalance_allocations() is True

    allocator.strategy_allocations = {"a": 1.0}
    assert allocator.should_rebalance_allocations() is False

    allocator.strategy_allocations = {"a": 0.5, "b": 0.5}
    monkeypatch.setattr(
        allocator,
        "_calculate_performance_score",
        lambda _strategy: (_ for _ in ()).throw(ValueError("bad rebalance")),
    )
    assert allocator.should_rebalance_allocations() is False
