import time

from ai_trading.utils.prof import SoftBudget


def test_soft_budget_elapsed_and_over():
    b = SoftBudget(interval_sec=0.1, fraction=0.5)
    time.sleep(0.02)
    assert b.elapsed_ms() >= 20
    time.sleep(0.05)
    assert b.over() is True or b.remaining() == 0.0
