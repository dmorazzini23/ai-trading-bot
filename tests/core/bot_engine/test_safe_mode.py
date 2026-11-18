from __future__ import annotations

import types
from typing import Any, cast


import ai_trading.core.bot_engine as bot_engine


def _ns(**kwargs: Any) -> Any:
    return cast(Any, types.SimpleNamespace(**kwargs))


def test_safe_mode_hard_block_overrides_failsoft(monkeypatch):
    cfg = _ns(
        execution_mode="live",
        safe_mode_allow_paper=False,
        safe_mode_failsoft=True,
        degraded_feed_mode="hard_block",
        execution_market_on_degraded=True,
    )
    monkeypatch.setattr(bot_engine, "get_trading_config", lambda: cfg)
    monkeypatch.setattr(
        bot_engine.provider_monitor,
        "safe_mode_degraded_only",
        lambda: True,
        raising=False,
    )

    assert bot_engine._safe_mode_blocks_trading() is True

