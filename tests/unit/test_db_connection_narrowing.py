"""Tests for narrowed exceptions in database connection."""

from __future__ import annotations

import time

from ai_trading.database.connection import DatabaseManager


def test_connect_handles_timeouterror(monkeypatch) -> None:
    """TimeoutError during connect should return False."""  # AI-AGENT-REF: narrow exception test
    db = DatabaseManager("sqlite:///:memory:")

    def boom(_):
        raise TimeoutError("sleep timeout")

    monkeypatch.setattr(time, "sleep", boom)
    assert db.connect() is False
