from __future__ import annotations

import importlib


def _reload_bot_engine():
    return importlib.reload(__import__("ai_trading.core.bot_engine", fromlist=["dummy"]))


def test_auto_train_policy_requires_paper_and_flag() -> None:
    be = _reload_bot_engine()
    assert be._auto_train_allowed("paper", allow_paper_train=True) is True
    assert be._auto_train_allowed("paper", allow_paper_train=False) is False
    assert be._auto_train_allowed("live", allow_paper_train=True) is False
    assert be._auto_train_allowed("sim", allow_paper_train=True) is False


