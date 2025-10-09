from __future__ import annotations

import pytest

from ai_trading.data import fetch as fetch_mod


@pytest.fixture(autouse=True)
def _reset_gap_env(monkeypatch):
    for key in (
        "AI_TRADING_GAP_RATIO_LIMIT",
        "DATA_MAX_GAP_RATIO_BPS",
        "MAX_GAP_RATIO_BPS",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def test_gap_ratio_limit_env_ratio(monkeypatch):
    monkeypatch.setenv("AI_TRADING_GAP_RATIO_LIMIT", "0.02")

    ratio = fetch_mod._resolve_gap_ratio_limit()
    assert ratio == pytest.approx(0.02)

    reason = fetch_mod._format_gap_ratio_reason(0.012, ratio)
    assert reason == "gap_ratio=1.20% > limit=2.00%"
