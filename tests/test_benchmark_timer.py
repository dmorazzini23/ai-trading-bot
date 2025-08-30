from ai_trading.utils.benchmark import BenchmarkTimer, measure


def test_benchmark_timer_records_elapsed_ms():
    with BenchmarkTimer() as timer:
        sum(range(10))
    assert timer.elapsed_ms >= 0


def test_measure_returns_result_and_ms():
    def add(a, b):
        return a + b

    result, elapsed_ms = measure(add, 1, 2)
    assert result == 3
    assert elapsed_ms >= 0
