from datetime import UTC, datetime
import json

import pytest

from ai_trading.config import research_policy as policy


@pytest.mark.parametrize("value", [None, "{", "[]", '{"enabled":true}'])
def test_invalid_policy_blocks(tmp_path, monkeypatch, value):
    path = tmp_path / "policy.json"
    if value is not None:
        path.write_text(value)
    monkeypatch.setattr(policy, "POLICY_PATH", path)
    assert policy.training_block_reason() == "research_reset_policy_invalid"


def test_review_does_not_release_training(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"enabled": True, "start_date": "2026-09-08",
                                "review_date": "2026-10-08", "resume_policy": "explicit_review_required"}))
    monkeypatch.setattr(policy, "POLICY_PATH", path)
    assert policy.training_block_reason(datetime(2027, 1, 1, tzinfo=UTC)) == "research_reset_active"
    path.write_text('{"enabled":false}')
    assert policy.training_block_reason() is None


def test_after_hours_reset_precedes_data_and_env(monkeypatch):
    from ai_trading.training import after_hours

    monkeypatch.setattr(policy, "training_block_reason", lambda now=None: "research_reset_active")
    monkeypatch.setattr(after_hours, "reload_env", lambda **kw: pytest.fail("must not load training inputs"))
    monkeypatch.setattr(after_hours, "_build_training_dataset", lambda *a, **kw: pytest.fail("data read"))
    assert after_hours.run_after_hours_training()["reason"] == "research_reset_active"


def test_market_close_reset_precedes_training(monkeypatch):
    from ai_trading.core import bot_engine

    monkeypatch.setattr(policy, "training_block_reason", lambda now=None: "research_reset_active")
    monkeypatch.setattr(bot_engine, "market_is_open", lambda *a: pytest.fail("must block at entry"))
    bot_engine.on_market_close()


def test_main_reset_precedes_claim(monkeypatch):
    from ai_trading import main

    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setattr(policy, "training_block_reason", lambda now=None: "research_reset_active")
    monkeypatch.setattr(main, "should_stop", lambda: False)
    monkeypatch.setattr(main, "_claim_market_close_training", lambda *a: pytest.fail("claim during reset"))
    main._maybe_trigger_market_close_training(datetime(2026, 9, 10, 21, tzinfo=UTC))
