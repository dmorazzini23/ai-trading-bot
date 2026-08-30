from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.tools.full_validation_artifact import build_full_validation_artifact


def test_full_validation_artifact_is_green_and_non_authoritative() -> None:
    payload = build_full_validation_artifact(
        commit="abc123",
        generated_at=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert payload["full_validation_green"] is True
    assert payload["commit"] == "abc123"
    assert payload["promotion_authority"] is False
    assert payload["live_money_authority"] is False
