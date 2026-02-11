from __future__ import annotations

import threading

from ai_trading.data import fetch


def test_gap_ratio_state_is_isolated_per_thread() -> None:
    """Per-call fetch state should not bleed across concurrent threads."""

    original_state = dict(getattr(fetch, "_state", {}) or {})
    barrier = threading.Barrier(2)
    results: dict[str, float | None] = {}
    errors: list[Exception] = []

    def _worker(name: str, ratio: float) -> None:
        try:
            fetch._set_fetch_state({})
            fetch._record_gap_ratio_state(ratio, metadata={"gap_ratio": ratio})
            barrier.wait(timeout=2)
            results[name] = fetch._current_gap_ratio()
        except Exception as exc:  # pragma: no cover - defensive thread capture
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=("one", 0.1))
    t2 = threading.Thread(target=_worker, args=("two", 0.2))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    fetch._set_fetch_state(original_state)

    assert not errors
    assert results["one"] == 0.1
    assert results["two"] == 0.2
