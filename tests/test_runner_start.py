"""Tests for ai_trading.runner.start."""
from __future__ import annotations

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

