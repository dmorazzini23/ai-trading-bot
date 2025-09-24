"""Tests for the lightweight log deduplication helper."""

from __future__ import annotations

import pytest

from ai_trading.logging import LogDeduper


def test_log_deduper_enforces_ttl() -> None:
    deduper = LogDeduper()
    assert deduper.should_log("order", ttl_s=10, now=100.0)
    assert not deduper.should_log("order", ttl_s=10, now=105.0)
    assert deduper.should_log("order", ttl_s=10, now=111.0)


def test_log_deduper_handles_multiple_keys() -> None:
    deduper = LogDeduper()
    assert deduper.should_log("a", ttl_s=5, now=0.0)
    assert deduper.should_log("b", ttl_s=5, now=1.0)
    assert not deduper.should_log("a", ttl_s=5, now=3.0)
    assert deduper.should_log("b", ttl_s=5, now=6.0)


def test_log_deduper_rejects_negative_ttl() -> None:
    deduper = LogDeduper()
    with pytest.raises(ValueError):
        deduper.should_log("bad", ttl_s=-1, now=0.0)
