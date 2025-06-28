import time
import signals

def test_parallel_vs_serial_prep_speed():
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    data = {s: None for s in symbols}  # placeholder mocks

    start_serial = time.perf_counter()
    for s in symbols:
        signals.prepare_indicators(data)  # simulate sequential
    duration_serial = time.perf_counter() - start_serial

    start_parallel = time.perf_counter()
    signals.prepare_indicators_parallel(symbols, data)
    duration_parallel = time.perf_counter() - start_parallel

    assert duration_parallel < duration_serial * 0.9, \
        f"Parallel too slow vs serial: {duration_parallel:.4f} vs {duration_serial:.4f}"
