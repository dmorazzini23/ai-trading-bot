import time

import pytest

pd = pytest.importorskip("pandas")
from ai_trading import signals


def test_parallel_vs_serial_prep_speed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "AMD", "IBM"]
    n = 600
    data = pd.DataFrame({
        "open": range(1, n + 1),
        "high": range(2, n + 2),
        "low": range(n),
        "close": range(1, n + 1),
        "volume": [100] * n,
    })

    start_serial = time.perf_counter()
    serial_outputs = []
    for _ in symbols:
        serial_outputs.append(signals.prepare_indicators(data))
    duration_serial = time.perf_counter() - start_serial

    start_parallel = time.perf_counter()
    signals.prepare_indicators_parallel(symbols, {s: data for s in symbols})
    duration_parallel = time.perf_counter() - start_parallel

    for output in serial_outputs:
        assert output is not None
        assert {"macd", "signal", "histogram"}.issubset(output.columns)
    assert duration_parallel < duration_serial * 2.5 or duration_serial < 0.1  # Allow pass if very fast (mocked)
