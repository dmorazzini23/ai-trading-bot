from __future__ import annotations

import json
import os
import threading
import types
from datetime import UTC, datetime

import pandas as pd

import pytest

import ai_trading.portfolio as portfolio_mod

from ai_trading.services.execution import (
    LegacyLiveExecutionBlockedError,
    execute_signal_orders,
    execute_trade_cycle,
    submit_order,
)
from ai_trading.services.portfolio import ensure_portfolio_weights
from ai_trading.services.reconciliation import reconcile_position_targets
from ai_trading.services.risk_approval import (
    RiskApprovalService,
    ensure_policy_learning_artifacts,
    load_policy_runtime_toggles,
    write_policy_runtime_toggles,
)
from ai_trading.services.signal import (
    evaluate_signal_and_confirm,
    generate_directional_signals,
)


class _DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.messages.append((message, args, dict(kwargs)))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.messages.append((message, args, dict(kwargs)))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.messages.append((message, args, dict(kwargs)))

    def exception(self, message: str, *args: object, **kwargs: object) -> None:
        self.messages.append((message, args, dict(kwargs)))


def test_generate_directional_signals_matches_price_deltas() -> None:
    df = pd.DataFrame({"price": [10.0, 11.0, 9.0, 9.0]})

    result = generate_directional_signals(df)

    assert result.tolist() == [0, 1, -1, 0]


def test_evaluate_signal_and_confirm_enforces_threshold() -> None:
    logger = _DummyLogger()
    ctx = types.SimpleNamespace(
        signal_manager=types.SimpleNamespace(
            evaluate=lambda *_args: (1, 0.25, "momentum")
        )
    )

    result = evaluate_signal_and_confirm(
        ctx,
        types.SimpleNamespace(),
        "AAPL",
        pd.DataFrame({"price": [1, 2]}),
        model=None,
        conf_threshold=0.5,
        logger=logger,
    )

    assert result == (-1, 0.0, "")
    assert logger.messages[0][0] == "SKIP_LOW_SIGNAL"


def test_execute_signal_orders_submits_expected_orders() -> None:
    calls: list[tuple[str, int, str]] = []
    ctx = types.SimpleNamespace(
        api=types.SimpleNamespace(
            submit_order=lambda symbol, qty, side: calls.append((symbol, qty, side))
        )
    )
    signals = pd.Series([1, 0, -1], index=["A", "B", "C"])

    orders = execute_signal_orders(ctx, signals, logger=_DummyLogger())

    assert orders == [("A", "buy"), ("C", "sell")]
    assert calls == [("A", 1, "buy"), ("C", 1, "sell")]


def test_ensure_portfolio_weights_requires_canonical_weights(monkeypatch) -> None:
    logger = _DummyLogger()
    monkeypatch.delattr(portfolio_mod, "compute_portfolio_weights", raising=False)

    with pytest.raises(RuntimeError, match="compute_portfolio_weights"):
        ensure_portfolio_weights(
            types.SimpleNamespace(),
            ["AAPL", "MSFT"],
            logger=logger,
        )


def test_reconcile_position_targets_prunes_stale_entries() -> None:
    ctx = types.SimpleNamespace(
        api=types.SimpleNamespace(
            list_positions=lambda: [types.SimpleNamespace(symbol="AAPL", qty="2")]
        ),
        stop_targets={"AAPL": 100.0, "MSFT": 50.0},
        take_profit_targets={"AAPL": 120.0, "MSFT": 55.0},
    )

    warned = reconcile_position_targets(
        ctx,
        logger=_DummyLogger(),
        targets_lock=threading.Lock(),
        warned=False,
    )

    assert warned is False
    assert "MSFT" not in ctx.stop_targets
    assert "MSFT" not in ctx.take_profit_targets
    assert "AAPL" in ctx.stop_targets
    assert ctx._reconciliation_position_snapshots["AAPL"]["qty"] == 2.0


def test_execution_service_blocks_legacy_live_submit(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.delenv("AI_TRADING_ENABLE_LEGACY_LIVE_EXECUTION", raising=False)

    with pytest.raises(LegacyLiveExecutionBlockedError, match="blocked for live legacy execution"):
        submit_order(types.SimpleNamespace(), "AAPL", 1, "buy", price=100.0)


def test_execution_service_blocks_legacy_live_trade_cycle(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.delenv("AI_TRADING_ENABLE_LEGACY_LIVE_EXECUTION", raising=False)

    with pytest.raises(LegacyLiveExecutionBlockedError, match="blocked for live legacy execution"):
        execute_trade_cycle(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            "AAPL",
            1000.0,
            model=None,
            regime_ok=True,
        )


def test_risk_approval_service_bootstraps_and_updates_runtime_toggles(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_TRADING_POLICY_ABLATION_STATE_PATH",
        str(tmp_path / "runtime" / "policy_ablation_state.json"),
    )
    monkeypatch.setenv(
        "AI_TRADING_POLICY_ABLATION_EVENTS_PATH",
        str(tmp_path / "runtime" / "policy_ablation_events.jsonl"),
    )
    monkeypatch.setenv(
        "AI_TRADING_POLICY_ROLLBACK_STATE_PATH",
        str(tmp_path / "runtime" / "policy_rollback_state.json"),
    )
    runtime_toggles_path = tmp_path / "runtime" / "policy_runtime_toggles.json"
    monkeypatch.setenv(
        "AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH",
        str(runtime_toggles_path),
    )

    bootstrapped = ensure_policy_learning_artifacts(
        now=datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
    )
    assert bootstrapped["runtime_toggles_ready"] is True
    assert runtime_toggles_path.exists()

    payload = RiskApprovalService().update_manual_overrides(
        disabled_slices=["ranker:bandit", "gate:max_loss"],
        diagnostics={"operator": "ops@example.com"},
        source_updated_at="2026-04-17T00:00:00Z",
    )
    assert payload["state"]["disabled_slices"] == ["GATE:MAX_LOSS", "RANKER:BANDIT"]

    persisted = json.loads(runtime_toggles_path.read_text(encoding="utf-8"))
    assert persisted["diagnostics"]["operator"] == "ops@example.com"

    reloaded = load_policy_runtime_toggles()
    assert reloaded["toggles"]["rankers"]["bandit_enabled"] is False


def test_write_policy_runtime_toggles_uses_atomic_replace(
    tmp_path,
    monkeypatch,
) -> None:
    runtime_toggles_path = tmp_path / "runtime" / "policy_runtime_toggles.json"
    monkeypatch.setenv(
        "AI_TRADING_POLICY_RUNTIME_TOGGLES_PATH",
        str(runtime_toggles_path),
    )
    replace_calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def _spy_replace(src: str | bytes, dst: str | bytes) -> None:
        replace_calls.append((str(os.fspath(src)), str(os.fspath(dst))))
        real_replace(src, dst)

    monkeypatch.setattr("ai_trading.services.risk_approval.os.replace", _spy_replace)

    path, payload = write_policy_runtime_toggles(
        disabled_slices=["ranker:bandit"],
        diagnostics={"operator": "ops@example.com"},
    )

    assert path == runtime_toggles_path
    assert replace_calls
    assert replace_calls[-1][1] == os.fspath(runtime_toggles_path)
    assert json.loads(runtime_toggles_path.read_text(encoding="utf-8")) == payload
