"""Sanity checks for centralized timing helpers exported by ai_trading.utils."""
from __future__ import annotations

import importlib

import pytest

from ai_trading.utils.timing import HTTP_TIMEOUT, clamp_timeout
from ai_trading.utils.sleep import sleep


def test_timing_exports_exist_and_behave() -> None:
    assert isinstance(HTTP_TIMEOUT, (int, float)) and HTTP_TIMEOUT > 0
    assert clamp_timeout(None) == pytest.approx(float(HTTP_TIMEOUT))
    assert clamp_timeout(0.0) >= 0.0
    assert clamp_timeout(-1) == pytest.approx(float(HTTP_TIMEOUT))
    assert sleep(0) == 0.0


def test_invalid_ai_http_timeout_does_not_break_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HTTP_TIMEOUT", raising=False)
    monkeypatch.setenv("AI_HTTP_TIMEOUT", "not-a-number")

    import ai_trading.utils.timing as timing

    original_timeout = timing.HTTP_TIMEOUT
    reloaded = timing
    try:
        reloaded = importlib.reload(timing)

        assert reloaded.HTTP_TIMEOUT == pytest.approx(10.0)
    finally:
        reloaded.HTTP_TIMEOUT = original_timeout
