"""Tests for ai_trading.runner.start."""
from __future__ import annotations

import importlib
import pytest
from types import SimpleNamespace

from ai_trading import runner
from ai_trading.core import bot_engine


class _DummyAPI:
    pass


def test_start_initializes_api():  # AI-AGENT-REF: ensure api set on start
    bot_engine._global_ctx = None
    api = _DummyAPI()
    ctx = runner.start(api)
    assert getattr(ctx, "api") is api
    bot_engine._global_ctx = None


def test_start_replaces_existing_api_when_client_provided():
    """Regression: explicit clients should override any previously attached API."""

    bot_engine._global_ctx = None
    runner.start(None)

    new_api = _DummyAPI()
    ctx = runner.start(new_api)
    assert getattr(ctx, "api") is new_api

    bot_engine._global_ctx = None


def test_start_fails_closed_when_api_unavailable_outside_test_runtime(monkeypatch: pytest.MonkeyPatch):
    """Runner must fail closed in non-test runtime when API attachment fails."""

    local_runner = importlib.reload(runner)

    # Use a fresh synthetic context so existing global runtime state cannot mask
    # the fail-closed path by providing a pre-attached API.
    fresh_ctx = SimpleNamespace(_context=SimpleNamespace())
    monkeypatch.setattr(local_runner.bot_engine, "get_ctx", lambda: fresh_ctx)
    monkeypatch.setattr(local_runner, "_allow_test_api_stub", lambda: False)
    monkeypatch.setattr(local_runner.bot_engine, "ensure_alpaca_attached", lambda _ctx: None)

    with pytest.raises(RuntimeError, match="RUNNER_API_ATTACH_FAILED"):
        local_runner.start(None)

    bot_engine._global_ctx = None
