"""Tests for execution guard helpers."""

from ai_trading.execution import guards


def _reset_state() -> None:
    guards.STATE.pdt = guards.PDTState()
    guards.STATE.shadow_cycle = False
    guards.STATE.shadow_cycle_forced = False
    guards.STATE.stale_symbols = 0
    guards.STATE.universe_size = 0


def test_shadow_cycle_persists_after_stale_ratio_trip() -> None:
    _reset_state()
    guards.begin_cycle(universe_size=10, degraded=False)
    for _ in range(4):
        guards.mark_symbol_stale()
    guards.end_cycle(stale_threshold_ratio=0.30)
    assert guards.shadow_active()

    guards.begin_cycle(universe_size=8, degraded=False)
    assert guards.shadow_active()

    guards.end_cycle(stale_threshold_ratio=0.50)
    guards.begin_cycle(universe_size=5, degraded=False)
    assert not guards.shadow_active()


def test_shadow_cycle_respects_degraded_flag() -> None:
    _reset_state()
    guards.begin_cycle(universe_size=12, degraded=True)
    assert guards.shadow_active()

    guards.end_cycle(stale_threshold_ratio=0.30)
    guards.begin_cycle(universe_size=12, degraded=False)
    assert not guards.shadow_active()
