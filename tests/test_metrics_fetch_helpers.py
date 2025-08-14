from types import SimpleNamespace

# AI-AGENT-REF: tests for fetch logging helpers
from ai_trading.telemetry.metrics_logger import (
    alpaca_fetch_bars_total,
    alpaca_fetch_errors_total,
    log_fetch_fail,
    log_fetch_ok,
)


class DummyLogger:
    def __init__(self):
        self.records = []

    def info(self, msg, extra=None):
        self.records.append(("INFO", msg, extra or {}))

    def warning(self, msg, extra=None):
        self.records.append(("WARNING", msg, extra or {}))


def test_log_fetch_ok_increments_and_logs():
    logger = DummyLogger()
    Bar = SimpleNamespace
    bars = [Bar(timestamp=SimpleNamespace(isoformat=lambda: "2025-08-13T20:55:00+00:00")) for _ in range(3)]
    log_fetch_ok(logger, "SPY", bars, bars[-1].timestamp)
    assert any(r[0] == "INFO" and r[1] == "DATA.FETCH.OK" for r in logger.records)


def test_log_fetch_fail_increments_and_logs():
    logger = DummyLogger()
    log_fetch_fail(logger, "SPY", "429", RuntimeError("rate limited"))
    assert any(r[0] == "WARNING" and r[1] == "DATA.FETCH.FAIL" for r in logger.records)
