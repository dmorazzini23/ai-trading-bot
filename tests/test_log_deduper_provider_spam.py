import logging

import pytest

from tests.optdeps import require

log_mod = require("ai_trading.logging")


@pytest.mark.unit
def test_throttle_summaries_flush_each_cycle(caplog):
    flush = getattr(log_mod, "flush_log_throttle_summaries", None)
    throttle_filter = getattr(log_mod, "_THROTTLE_FILTER", None)
    if flush is None or throttle_filter is None:
        pytest.skip("Log throttle helpers unavailable")

    caplog.set_level(logging.INFO)
    with throttle_filter._lock:  # type: ignore[attr-defined]
        throttle_filter._state.clear()  # type: ignore[attr-defined]

    logger = log_mod.get_logger("ai_trading.tests.deduper")

    logger.info("PROVIDER_SPAM", extra={"provider": "feed-a"})
    logger.info("PROVIDER_SPAM", extra={"provider": "feed-a"})

    with throttle_filter._lock:  # type: ignore[attr-defined]
        first_cycle_suppressed = int(
            throttle_filter._state.get("PROVIDER_SPAM", {}).get("suppressed", 0)  # type: ignore[attr-defined]
        )
    assert first_cycle_suppressed >= 1

    flush()

    messages = [rec.getMessage() for rec in caplog.records]
    summaries = [msg for msg in messages if msg.startswith("LOG_THROTTLE_SUMMARY")]
    assert summaries

    def _suppressed_count(summary: str) -> int:
        for token in summary.split():
            if token.startswith("suppressed="):
                try:
                    return int(token.split("=", 1)[1])
                except ValueError:  # pragma: no cover - defensive parsing
                    continue
        raise AssertionError(f"suppressed count missing in summary: {summary}")

    assert _suppressed_count(summaries[-1]) == first_cycle_suppressed

    with throttle_filter._lock:  # type: ignore[attr-defined]
        suppressed_after = int(
            throttle_filter._state.get("PROVIDER_SPAM", {}).get("suppressed", 0)  # type: ignore[attr-defined]
        )
    assert suppressed_after == 0

    logger.info("PROVIDER_SPAM", extra={"provider": "feed-a"})
    logger.info("PROVIDER_SPAM", extra={"provider": "feed-a"})

    with throttle_filter._lock:  # type: ignore[attr-defined]
        second_cycle_suppressed = int(
            throttle_filter._state.get("PROVIDER_SPAM", {}).get("suppressed", 0)  # type: ignore[attr-defined]
        )
    assert second_cycle_suppressed >= 1

    flush()

    messages = [rec.getMessage() for rec in caplog.records]
    summaries = [msg for msg in messages if msg.startswith("LOG_THROTTLE_SUMMARY")]
    assert len(summaries) == 2
    assert _suppressed_count(summaries[-1]) == second_cycle_suppressed
