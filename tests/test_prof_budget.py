import time

import pytest

from ai_trading.utils import prof
from ai_trading.utils.prof import SoftBudget


def test_soft_budget_direct_usage_elapsed_and_over():
    budget = SoftBudget(50)

    initial_elapsed = budget.elapsed_ms()
    assert initial_elapsed <= 1

    time.sleep(0.005)
    assert budget.elapsed_ms() >= 1
    assert budget.remaining() > 0.0

    time.sleep(0.07)
    assert budget.over_budget() is True
    assert budget.remaining() == 0.0


def test_soft_budget_context_manager_resets_start():
    budget = SoftBudget(50)
    time.sleep(0.005)
    assert budget.elapsed_ms() >= 1

    with budget as managed_budget:
        assert managed_budget is budget
        elapsed_on_enter = managed_budget.elapsed_ms()
        assert elapsed_on_enter <= 1

        time.sleep(0.005)
        assert managed_budget.elapsed_ms() >= 1
        assert managed_budget.remaining() > 0.0

    time.sleep(0.07)
    assert budget.over_budget() is True


def test_elapsed_ms_rounds_to_nearest_monotonic(monkeypatch):
    sequence = iter(
        [
            0,  # __init__
            499_999,  # first elapsed -> < 0.5ms, rounds to 0
            500_000,  # second elapsed -> == 0.5ms, rounds up to 1
        ]
    )

    monkeypatch.setattr(prof.time, "perf_counter_ns", lambda: next(sequence))

    budget = SoftBudget(10)
    assert budget.elapsed_ms() == 0
    assert budget.elapsed_ms() == 1


def test_over_budget_and_remaining_use_existing_start(monkeypatch):
    sequence = iter(
        [
            0,  # __init__
            4_000_000,  # over_budget -> 4ms elapsed
            7_000_000,  # remaining -> 7ms elapsed (> budget)
            7_000_000,  # elapsed_ms -> same observation to confirm start retained
        ]
    )
    monkeypatch.setattr(prof.time, "perf_counter_ns", lambda: next(sequence))

    budget = SoftBudget(5)
    assert budget.over_budget() is False
    assert budget.remaining() == 0.0

    # remaining should not reset start; calling elapsed should still reflect total time
    assert budget.elapsed_ms() == 7
