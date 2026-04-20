from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from ai_trading.core import bot_engine
from ai_trading.core import runtime_services


def test_legacy_health_payload_handles_runtime_not_ready(monkeypatch) -> None:
    monkeypatch.setattr(bot_engine, "_get_runtime_context_or_none", lambda: None)
    monkeypatch.setattr(
        bot_engine,
        "build_canonical_healthz_payload",
        lambda **kwargs: {
            "status": "degraded",
            "error": kwargs.get("error"),
        },
    )
    monkeypatch.setattr(bot_engine.state, "no_signal_events", 7, raising=False)
    monkeypatch.setattr(bot_engine.state, "indicator_failures", 3, raising=False)

    payload = bot_engine._legacy_health_payload()

    assert payload["status"] == "degraded"
    assert payload["error"] == "runtime not ready"
    assert payload["no_signal_events"] == 7
    assert payload["indicator_failures"] == 3


def test_legacy_health_payload_uses_regime_symbols_when_runtime_has_no_tickers(
    monkeypatch,
) -> None:
    called: dict[str, object] = {}

    def _capture_health(ctx, symbols, min_rows=None):  # noqa: ARG001
        called["symbols"] = list(symbols)
        return {"checked": len(symbols)}

    monkeypatch.setattr(
        bot_engine,
        "_get_runtime_context_or_none",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(runtime_services, "pre_trade_health_check_runtime", _capture_health)
    monkeypatch.setattr(bot_engine, "REGIME_SYMBOLS", ["SPY", "QQQ"], raising=False)
    monkeypatch.setattr(bot_engine, "build_canonical_healthz_payload", lambda **kwargs: {})

    payload = bot_engine._legacy_health_payload()

    assert payload["no_signal_events"] == getattr(bot_engine.state, "no_signal_events", 0)
    assert called["symbols"] == ["SPY", "QQQ"]


def test_reconcile_positions_preserves_warn_state_across_calls(monkeypatch) -> None:
    warned_values: list[bool] = []

    def _fake_reconcile(ctx, *, logger, targets_lock, warned):  # noqa: ARG001
        warned_values.append(bool(warned))
        return True

    monkeypatch.setattr(
        bot_engine,
        "_reconcile_position_targets_service",
        _fake_reconcile,
    )
    monkeypatch.setattr(runtime_services, "_reconcile_warned", False)

    bot_engine.reconcile_positions(cast(Any, SimpleNamespace()))
    bot_engine.reconcile_positions(cast(Any, SimpleNamespace()))

    assert warned_values == [False, True]
