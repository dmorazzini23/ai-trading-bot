from __future__ import annotations

from time import perf_counter

from ai_trading.utils.timing import sleep

def test_sleep_unaffected_by_monkeypatch(monkeypatch) -> None:
    """sleep should block even if time.sleep is monkeypatched."""  # AI-AGENT-REF: ensure robustness
    slept = {"count": 0}

    def fake_sleep(_: float) -> None:
        slept["count"] += 1

    import time as real_time

    monkeypatch.setattr(real_time, "sleep", fake_sleep)
    start = perf_counter()
    sleep(0.01)
    elapsed = perf_counter() - start
    assert slept["count"] == 0
    assert elapsed >= 0.009
