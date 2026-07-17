from __future__ import annotations

from typing import Any

import pytest

from ai_trading.execution.live_trading import ExecutionEngine
from ai_trading.governance.entry_control import (
    REPLAY_LIVE_ENTRY_CONTROL_ATTR,
    REPLAY_LIVE_PARITY_GATE_FAILED,
    build_replay_live_entry_control,
    evaluate_replay_live_order,
)


def _failed_gate(*, available: bool = True) -> dict[str, Any]:
    return {
        "enabled": True,
        "available": available,
        "ok": False,
        "status": "fail",
        "reason": "replay_violations" if available else "replay_gate_unavailable",
        "failed_checks": ["replay_counterfactual"],
    }


@pytest.mark.parametrize("available", [True, False])
def test_required_replay_failure_is_fail_closed_only_for_live(available: bool) -> None:
    live = build_replay_live_entry_control(
        gate=_failed_gate(available=available),
        required=True,
        execution_mode="live",
    )
    paper = build_replay_live_entry_control(
        gate=_failed_gate(available=available),
        required=True,
        execution_mode="paper",
    )
    sim = build_replay_live_entry_control(
        gate=_failed_gate(available=available),
        required=True,
        execution_mode="sim",
    )

    assert live["status"] == "blocked"
    assert live["reason"] == REPLAY_LIVE_PARITY_GATE_FAILED
    assert live["block_exposure_increasing"] is True
    assert paper["status"] == "monitor_only"
    assert paper["block_exposure_increasing"] is False
    assert sim["status"] == "monitor_only"
    assert sim["exposure_increasing_allowed"] is True


@pytest.mark.parametrize(
    "gate",
    [
        None,
        {"enabled": False, "available": True, "ok": True},
        {"enabled": True, "available": False, "ok": True},
    ],
)
def test_required_nonpassing_gate_snapshot_blocks_live_and_monitors_paper(
    gate: dict[str, Any] | None,
) -> None:
    live = build_replay_live_entry_control(
        gate=gate,
        required=True,
        execution_mode="live",
    )
    paper = build_replay_live_entry_control(
        gate=gate,
        required=True,
        execution_mode="paper",
    )

    assert live["status"] == "blocked"
    assert live["block_exposure_increasing"] is True
    assert paper["status"] == "monitor_only"
    assert paper["block_exposure_increasing"] is False


def test_optional_replay_gate_remains_nonblocking() -> None:
    control = build_replay_live_entry_control(
        gate=None,
        required=False,
        execution_mode="live",
    )

    assert control["status"] == "disabled"
    assert control["block_exposure_increasing"] is False


def test_live_reduction_is_clipped_before_it_can_flip_through_zero() -> None:
    control = build_replay_live_entry_control(
        gate=_failed_gate(),
        required=True,
        execution_mode="live",
    )

    decision = evaluate_replay_live_order(
        control=control,
        side="sell",
        requested_quantity=15,
        current_position_quantity=10,
        closing_position=True,
    )

    assert decision["allowed"] is True
    assert decision["reduction"] is True
    assert decision["effective_quantity"] == 10
    assert decision["clamped"] is True


@pytest.mark.parametrize(
    ("side", "position"),
    [("buy", 10), ("sell", -10), ("sell", 0), ("cover", 0)],
)
def test_live_opening_or_wrong_way_order_is_blocked(side: str, position: int) -> None:
    control = build_replay_live_entry_control(
        gate=_failed_gate(),
        required=True,
        execution_mode="live",
    )

    decision = evaluate_replay_live_order(
        control=control,
        side=side,
        requested_quantity=5,
        current_position_quantity=position,
        closing_position=False,
    )

    assert decision["allowed"] is False
    assert decision["reason"] == REPLAY_LIVE_PARITY_GATE_FAILED


def _engine_at_replay_boundary(
    *,
    execution_mode: str,
    position_quantity: float,
    shadow_mode: bool = False,
) -> tuple[ExecutionEngine, list[dict[str, Any]]]:
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine.execution_mode = execution_mode
    engine.shadow_mode = shadow_mode
    setattr(
        engine,
        REPLAY_LIVE_ENTRY_CONTROL_ATTR,
        build_replay_live_entry_control(
            gate=_failed_gate(),
            required=True,
            execution_mode=execution_mode,
        ),
    )
    engine._position_quantity = lambda _symbol: position_quantity  # type: ignore[method-assign]
    engine._warmup_data_only_mode_active = lambda: True  # type: ignore[method-assign]
    skipped: list[dict[str, Any]] = []
    engine._skip_submit = lambda **kwargs: skipped.append(kwargs)  # type: ignore[method-assign]
    return engine, skipped


def test_live_engine_blocks_opening_at_submission_boundary() -> None:
    engine, skipped = _engine_at_replay_boundary(
        execution_mode="live",
        position_quantity=0,
    )

    result = engine.execute_order("AAPL", "buy", 5, order_type="market")

    assert result is None
    assert skipped[0]["reason"] == REPLAY_LIVE_PARITY_GATE_FAILED
    assert engine._last_replay_entry_control_decision["allowed"] is False


def test_required_live_engine_fails_closed_when_control_snapshot_is_absent(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE",
        "1",
    )
    engine, skipped = _engine_at_replay_boundary(
        execution_mode="live",
        position_quantity=0,
    )
    delattr(engine, REPLAY_LIVE_ENTRY_CONTROL_ATTR)

    result = engine.execute_order("AAPL", "buy", 5, order_type="market")

    assert result is None
    assert skipped[0]["reason"] == REPLAY_LIVE_PARITY_GATE_FAILED
    decision = engine._last_replay_entry_control_decision
    assert decision["allowed"] is False
    assert decision["gate_reason"] == "replay_live_entry_control_snapshot_unavailable"


def test_paper_engine_continues_past_failed_replay_boundary_for_evidence() -> None:
    engine, skipped = _engine_at_replay_boundary(
        execution_mode="paper",
        position_quantity=0,
    )

    result = engine.execute_order("AAPL", "buy", 5, order_type="market")

    assert result is None
    assert skipped[0]["reason"] == "warmup_data_only"
    assert engine._last_replay_entry_control_decision["allowed"] is True
    assert engine._last_replay_entry_control_decision["monitor_only"] is True


def test_live_engine_allows_only_non_flipping_reduction_quantity() -> None:
    engine, skipped = _engine_at_replay_boundary(
        execution_mode="live",
        position_quantity=10,
    )

    result = engine.execute_order("AAPL", "sell", 15, order_type="market")

    assert result is None
    assert skipped[0]["reason"] == "warmup_data_only"
    decision = engine._last_replay_entry_control_decision
    assert decision["allowed"] is True
    assert decision["effective_quantity"] == 10
    assert decision["clamped"] is True


def test_whole_engine_shadow_mode_remains_monitor_only() -> None:
    engine, skipped = _engine_at_replay_boundary(
        execution_mode="live",
        position_quantity=0,
        shadow_mode=True,
    )

    result = engine.execute_order("AAPL", "buy", 5, order_type="market")

    assert result is None
    assert skipped[0]["reason"] == "warmup_data_only"
    assert engine._last_replay_entry_control_decision["monitor_only"] is True
