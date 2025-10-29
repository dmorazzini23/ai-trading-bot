"""Tests for execution timing wall-clock accounting."""

import pytest

from ai_trading.execution import timing


def test_record_cycle_wall_tracks_maximum_elapsed() -> None:
    """Wall-clock recorder should retain the largest elapsed span."""

    captured: list[tuple[float, dict]] = []

    def _on_complete(elapsed: float, metadata: dict) -> None:
        captured.append((elapsed, metadata))

    timing.reset_cycle(on_span_complete=_on_complete)

    timing.record_cycle_wall(0.05, {"stage": "cycle_execute"})
    first_seconds = timing.cycle_seconds()
    assert first_seconds >= 0.05

    timing.record_cycle_wall(0.02, {"stage": "cycle_execute"})
    assert timing.cycle_seconds() == pytest.approx(first_seconds, rel=1e-6)

    timing.record_cycle_wall(0.10, {"stage": "cycle_execute"})
    assert timing.cycle_seconds() >= 0.10

    assert captured
    last_elapsed, last_meta = captured[-1]
    assert last_elapsed >= 0.10
    assert last_meta.get("stage") == "cycle_execute"
