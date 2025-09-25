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


@pytest.mark.unit
def test_provider_dedupe_emits_summary(caplog):
    deduper = getattr(log_mod, "provider_log_deduper", None)
    record_suppressed = getattr(log_mod, "record_provider_log_suppressed", None)
    reset = getattr(log_mod, "reset_provider_log_dedupe", None)
    flush = getattr(log_mod, "flush_log_throttle_summaries", None)
    if None in {deduper, record_suppressed, reset, flush}:
        pytest.skip("Provider dedupe helpers unavailable")

    reset()
    caplog.set_level(logging.INFO)
    logger = log_mod.get_logger("ai_trading.tests.provider")

    raw_ttl = getattr(require("ai_trading.config.settings").get_settings(), "logging_dedupe_ttl_s", 0)
    ttl = max(1, int(raw_ttl))
    key = "DATA_PROVIDER_SWITCHOVER:primary->backup"

    assert deduper.should_log(key, ttl)
    logger.info("DATA_PROVIDER_SWITCHOVER", extra={"from_provider": "primary", "to_provider": "backup"})

    assert not deduper.should_log(key, ttl)
    record_suppressed("DATA_PROVIDER_SWITCHOVER")

    flush()

    def _summary_count(record) -> tuple[str, int]:
        message = record.getMessage()
        count = None
        for token in message.split():
            if token.startswith("suppressed="):
                try:
                    count = int(token.split("=", 1)[1])
                except ValueError:  # pragma: no cover - defensive parsing
                    count = None
        return message, count if count is not None else -1

    summaries = [rec for rec in caplog.records if rec.getMessage().startswith("LOG_THROTTLE_SUMMARY")]
    matching = [rec for rec in summaries if 'key="DATA_PROVIDER_SWITCHOVER"' in rec.getMessage()]
    assert matching, "Expected summary for DATA_PROVIDER_SWITCHOVER"
    summary_message, suppressed = _summary_count(matching[-1])
    assert suppressed == 1, summary_message

    caplog.clear()
    flush()
    assert not any(
        'key="DATA_PROVIDER_SWITCHOVER"' in rec.getMessage()
        for rec in caplog.records
        if rec.getMessage().startswith("LOG_THROTTLE_SUMMARY")
    )

    reset()
