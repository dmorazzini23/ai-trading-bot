from time import perf_counter
from ai_trading.utils import sleep


def test_utils_sleep_is_measurable():
    start = perf_counter()
    sleep(0)  # request 0 -> enforced minimum
    elapsed = perf_counter() - start
    assert elapsed >= 0.009  # AI-AGENT-REF: ensure busy-wait measurable
