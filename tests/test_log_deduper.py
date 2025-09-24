"""Unit tests for the lightweight log deduper helper."""

from __future__ import annotations

from ai_trading.logging import LogDeduper


def test_log_deduper_respects_ttl():
    deduper = LogDeduper()

    assert deduper.should_log("key", ttl_s=10, now=100.0)
    assert not deduper.should_log("key", ttl_s=10, now=105.0)
    assert deduper.should_log("key", ttl_s=10, now=110.0)


def test_log_deduper_tracks_keys_independently():
    deduper = LogDeduper()

    assert deduper.should_log("a", ttl_s=5, now=50.0)
    assert deduper.should_log("b", ttl_s=5, now=50.0)
    assert not deduper.should_log("a", ttl_s=5, now=54.0)
    assert deduper.should_log("b", ttl_s=5, now=55.1)
