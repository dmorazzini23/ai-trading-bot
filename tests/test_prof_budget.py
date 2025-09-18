import time

from ai_trading.utils.prof import SoftBudget


def test_soft_budget_direct_usage_elapsed_and_over():
    budget = SoftBudget(interval_sec=0.05, fraction=1.0)

    initial_elapsed = budget.elapsed_ms()
    assert initial_elapsed <= 1

    time.sleep(0.005)
    assert budget.elapsed_ms() >= 1
    assert budget.remaining() > 0.0

    time.sleep(0.07)
    assert budget.over() is True
    assert budget.remaining() == 0.0


def test_soft_budget_context_manager_resets_start():
    budget = SoftBudget(interval_sec=0.05, fraction=1.0)
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
    assert budget.over() is True
