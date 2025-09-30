import pytest

from ai_trading.utils.prof import SoftBudget


def test_soft_budget_direct_regression(monkeypatch):
    sequence = iter(
        [
            0,  # __init__
            400_000,  # elapsed_ms -> <1ms, emits minimal tick
            700_000,  # remaining -> accumulates fractional ns, still 1ms
            1_400_000,  # elapsed_ms -> crosses 1ms boundary
            5_500_000,  # over_budget -> accumulates to 5ms
            5_500_000,  # remaining -> no further time passes
        ]
    )

    monkeypatch.setattr(
        "ai_trading.utils.prof.time.perf_counter_ns", lambda: next(sequence)
    )

    budget = SoftBudget(5)

    assert budget.elapsed_ms() == 1
    assert budget.remaining() == pytest.approx(0.004)
    assert budget.elapsed_ms() == 1
    assert budget.over_budget() is True
    assert budget.remaining() == 0.0


def test_soft_budget_context_manager_regression(monkeypatch):
    sequence = iter(
        [
            0,  # __init__
            200_000,  # __enter__ reset start
            500_000,  # elapsed_ms -> <1ms, emits minimal tick
            900_000,  # remaining -> still relying on accumulated fractional ns
            6_200_000,  # over_budget -> exceeds 5ms budget
            6_200_000,  # elapsed_ms after context exit -> same observation
            6_200_000,  # remaining -> no further time passes
        ]
    )

    monkeypatch.setattr(
        "ai_trading.utils.prof.time.perf_counter_ns", lambda: next(sequence)
    )

    budget = SoftBudget(5)

    with budget as managed:
        assert managed.elapsed_ms() == 1
        assert managed.remaining() == pytest.approx(0.004)
        assert managed.over_budget() is True

    assert budget.elapsed_ms() == 6
    assert budget.remaining() == 0.0


def test_soft_budget_short_sleep_respects_budget(monkeypatch):
    sequence = iter(
        [
            0,  # __init__
            500_000,  # elapsed_ms -> <1ms
            500_000,  # over_budget -> same observation, still under budget
            1_500_000,  # elapsed_ms -> crosses 1ms boundary
            1_500_000,  # over_budget -> >= budget
        ]
    )

    monkeypatch.setattr(
        "ai_trading.utils.prof.time.perf_counter_ns", lambda: next(sequence)
    )

    budget = SoftBudget(1)

    assert budget.elapsed_ms() == 1
    assert budget.over_budget() is False
    assert budget.elapsed_ms() == 1
    assert budget.over_budget() is True


def test_soft_budget_reset_clears_fractional_state(monkeypatch):
    sequence = iter(
        [
            0,  # __init__
            500_000,  # elapsed_ms -> <1ms
            800_000,  # reset -> establishes new baseline
            900_000,  # elapsed_ms after reset -> <1ms
            900_000,  # over_budget -> same observation, still under budget
        ]
    )

    monkeypatch.setattr(
        "ai_trading.utils.prof.time.perf_counter_ns", lambda: next(sequence)
    )

    budget = SoftBudget(5)

    assert budget.elapsed_ms() == 1
    budget.reset()
    assert budget.elapsed_ms() == 1
    assert budget.over_budget() is False


def test_soft_budget_fractional_over_budget_threshold(monkeypatch):
    sequence = iter(
        [
            0,  # __init__
            900_000,  # elapsed_ms -> <1ms
            1_800_000,  # over_budget -> cumulative 1.8ms (< 2ms)
            2_300_000,  # over_budget -> cumulative 2.3ms (>= 2ms)
        ]
    )

    monkeypatch.setattr(
        "ai_trading.utils.prof.time.perf_counter_ns", lambda: next(sequence)
    )

    budget = SoftBudget(2)

    assert budget.elapsed_ms() == 1
    assert budget.over_budget() is False
    assert budget.over_budget() is True
