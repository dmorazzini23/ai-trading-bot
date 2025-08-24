from time import perf_counter

from ai_trading.utils import sleep


def test_utils_sleep_is_measurable():
    start = perf_counter()
    sleep(0.01)
    elapsed = perf_counter() - start
    # The same threshold used in the smoke timing test
    assert elapsed >= 0.009
    # AI-AGENT-REF: ensure sleep wrapper blocks

