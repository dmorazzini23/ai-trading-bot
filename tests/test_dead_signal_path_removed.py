"""The code left unreachable by removing SignalManager.evaluate stays removed.

Live neighbours that share names or state with the removed code are pinned
here as well, so the cleanup cannot drift into live behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

from ai_trading import settings as settings_mod
from ai_trading.core import bot_engine
from ai_trading.core.runtime import BotRuntime, enhance_runtime_with_context


def test_signal_manager_shell_and_dead_sentiment_helpers_are_gone() -> None:
    for name in (
        "SignalManager",
        "signal_manager",
        "_fetch_sentiment_ctx",
        "_sentiment_evidence_is_authoritative",
        "_LAST_PRICE",
        "PRICE_TTL_PCT",
        "update_signal_weights",
        "signals_evaluated",
    ):
        assert not hasattr(bot_engine, name), name
    assert "signal_manager" not in bot_engine.BotContext.__dataclass_fields__
    assert "signal_manager" not in BotRuntime.__dataclass_fields__


def test_live_sentiment_state_and_weight_writers_remain() -> None:
    assert callable(bot_engine.fetch_sentiment)
    assert isinstance(bot_engine._SENTIMENT_CACHE, dict)
    assert hasattr(bot_engine.sentiment_lock, "acquire")
    # The daily meta-learning optimizers still default to this output path.
    assert bot_engine.SIGNAL_WEIGHTS_FILE.endswith("signal_weights.csv")


def test_enhance_runtime_copies_remaining_context_attributes() -> None:
    runtime = BotRuntime(cfg=SimpleNamespace())  # type: ignore[arg-type]
    lazy = SimpleNamespace(
        api="api",
        data_client="data",
        data_fetcher="fetcher",
        risk_engine="risk",
        capital_scaler="scaler",
        execution_engine="exec",
        drawdown_circuit_breaker="dd",
        model="model",
        signal_manager="ignored",
    )

    enhanced = enhance_runtime_with_context(runtime, lazy)

    assert (enhanced.api, enhanced.data_fetcher, enhanced.execution_engine) == ("api", "fetcher", "exec")
    assert not hasattr(enhanced, "signal_manager")


def test_routing_getter_removed_but_settings_field_kept() -> None:
    assert not hasattr(settings_mod, "get_regime_signal_routing_enabled")
    # Removing the field would change the resolved configuration.
    assert "regime_signal_routing_enabled" in settings_mod.Settings.model_fields
