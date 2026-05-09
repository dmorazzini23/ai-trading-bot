from __future__ import annotations

import json
import os
import threading
import types
from datetime import UTC, datetime
from typing import Any

import pandas as pd

import pytest

import ai_trading.portfolio as portfolio_mod

from ai_trading.services.execution import (
    ExecutionService,
    NonNettingLiveExecutionBlockedError,
    UnknownExecutionModeError,
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


def _request_snapshot(order_data: Any) -> tuple[str, float, str]:
    side = getattr(order_data, "side")
    side_value = getattr(side, "value", side)
    return (str(order_data.symbol), float(order_data.qty), str(side_value))


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


def test_execute_signal_orders_submits_expected_orders(monkeypatch) -> None:
    calls: list[tuple[str, int, str]] = []
    ctx = types.SimpleNamespace()
    signals = pd.Series([1, 0, -1], index=["A", "B", "C"])

    def _submit_order(_self, _ctx, symbol, qty, side, **_kwargs):
        calls.append((symbol, qty, side))
        return types.SimpleNamespace(status="accepted")

    monkeypatch.setattr(ExecutionService, "submit_order", _submit_order)
    orders = execute_signal_orders(ctx, signals, logger=_DummyLogger())

    assert orders == [("A", "buy"), ("C", "sell")]
    assert calls == [("A", 1, "buy"), ("C", 1, "sell")]


def test_execute_signal_orders_returns_only_successfully_submitted_orders(monkeypatch) -> None:
    calls: list[tuple[str, int, str]] = []
    ctx = types.SimpleNamespace()
    signals = pd.Series([1, -1], index=["A", "C"])

    def _submit_order(_self, _ctx, symbol, qty, side, **_kwargs):
        calls.append((symbol, qty, side))
        if symbol == "A":
            raise ValueError("broker rejected")
        return types.SimpleNamespace(status="accepted")

    monkeypatch.setattr(ExecutionService, "submit_order", _submit_order)
    orders = execute_signal_orders(ctx, signals, logger=_DummyLogger())

    assert orders == [("C", "sell")]
    assert calls == [("A", 1, "buy"), ("C", 1, "sell")]


def test_execute_signal_orders_excludes_terminal_rejected_response(monkeypatch) -> None:
    ctx = types.SimpleNamespace()
    signals = pd.Series([1, -1], index=["A", "C"])

    def _submit_order(_self, _ctx, symbol, _qty, _side, **_kwargs):
        return types.SimpleNamespace(status="rejected" if symbol == "A" else "accepted")

    monkeypatch.setattr(ExecutionService, "submit_order", _submit_order)
    orders = execute_signal_orders(ctx, signals, logger=_DummyLogger())

    assert orders == [("C", "sell")]


def test_execute_signal_orders_uses_canonical_submit_service(monkeypatch) -> None:
    calls: list[tuple[str, int, str]] = []

    def _submit_order(_self, _ctx, symbol, qty, side, **_kwargs) -> types.SimpleNamespace:
        calls.append((symbol, qty, side))
        return types.SimpleNamespace(status="accepted")

    monkeypatch.setattr(ExecutionService, "submit_order", _submit_order)
    ctx = types.SimpleNamespace()

    orders = execute_signal_orders(ctx, pd.Series([1], index=["AAPL"]), logger=_DummyLogger())

    assert orders == [("AAPL", "buy")]
    assert calls == [("AAPL", 1, "buy")]


def test_execute_signal_orders_blocks_non_netting_live_mode(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.delenv("AI_TRADING_ENABLE_NON_NETTING_LIVE_EXECUTION", raising=False)
    ctx = types.SimpleNamespace(
        api=types.SimpleNamespace(
            submit_order=lambda *, order_data: types.SimpleNamespace(status="accepted")
        )
    )

    with pytest.raises(NonNettingLiveExecutionBlockedError, match="execute_signal_orders"):
        execute_signal_orders(ctx, pd.Series([1], index=["AAPL"]), logger=_DummyLogger())


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


def test_reconcile_position_targets_preserves_fractional_positions() -> None:
    ctx = types.SimpleNamespace(
        api=types.SimpleNamespace(
            list_positions=lambda: [types.SimpleNamespace(symbol="AAPL", qty="0.5")]
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
    assert "AAPL" in ctx.stop_targets
    assert "AAPL" in ctx.take_profit_targets
    assert "MSFT" not in ctx.stop_targets
    assert "MSFT" not in ctx.take_profit_targets
    assert ctx._reconciliation_position_snapshots["AAPL"]["qty"] == 0.5


def test_reconcile_position_targets_prunes_epsilon_flat_fraction() -> None:
    ctx = types.SimpleNamespace(
        api=types.SimpleNamespace(
            list_positions=lambda: [types.SimpleNamespace(symbol="AAPL", qty="0.0000000001")]
        ),
        stop_targets={"AAPL": 100.0},
        take_profit_targets={"AAPL": 120.0},
    )

    reconcile_position_targets(
        ctx,
        logger=_DummyLogger(),
        targets_lock=threading.Lock(),
        warned=False,
    )

    assert ctx.stop_targets == {}
    assert ctx.take_profit_targets == {}


def test_reconcile_position_targets_supports_raw_alpaca_get_all_positions() -> None:
    ctx = types.SimpleNamespace(
        api=types.SimpleNamespace(
            get_all_positions=lambda: [types.SimpleNamespace(symbol="AAPL", qty="3")]
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
    assert ctx._reconciliation_position_snapshots["AAPL"]["qty"] == 3.0


def test_reconcile_position_targets_skips_missing_positions_capability() -> None:
    logger = _DummyLogger()
    ctx = types.SimpleNamespace(
        api=types.SimpleNamespace(),
        stop_targets={"AAPL": 100.0},
        take_profit_targets={"AAPL": 120.0},
    )

    warned = reconcile_position_targets(
        ctx,
        logger=logger,
        targets_lock=threading.Lock(),
        warned=False,
    )

    assert warned is True
    assert ctx.stop_targets == {"AAPL": 100.0}
    assert ctx.take_profit_targets == {"AAPL": 120.0}
    assert logger.messages == [
        ("Skipping reconciliation: broker client missing positions method", (), {})
    ]


def test_reconcile_position_targets_surfaces_broker_errors() -> None:
    def _list_positions() -> list[object]:
        raise OSError("broker unavailable")

    ctx = types.SimpleNamespace(
        api=types.SimpleNamespace(list_positions=_list_positions),
        stop_targets={"AAPL": 100.0},
        take_profit_targets={"AAPL": 120.0},
    )

    with pytest.raises(RuntimeError, match="position target reconciliation failed"):
        reconcile_position_targets(
            ctx,
            logger=_DummyLogger(),
            targets_lock=threading.Lock(),
            warned=False,
        )

    assert ctx._reconciliation_error == "broker unavailable"


def test_execution_service_blocks_non_netting_live_submit(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.delenv("AI_TRADING_ENABLE_NON_NETTING_LIVE_EXECUTION", raising=False)

    with pytest.raises(NonNettingLiveExecutionBlockedError, match="blocked for live non-netting execution"):
        submit_order(types.SimpleNamespace(), "AAPL", 1, "buy", price=100.0)


def test_execution_service_blocks_non_netting_live_trade_cycle(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.delenv("AI_TRADING_ENABLE_NON_NETTING_LIVE_EXECUTION", raising=False)

    with pytest.raises(NonNettingLiveExecutionBlockedError, match="blocked for live non-netting execution"):
        execute_trade_cycle(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            "AAPL",
            1000.0,
            model=None,
            regime_ok=True,
        )


def test_execution_service_non_netting_live_escape_hatch_is_test_only(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_ENABLE_NON_NETTING_LIVE_EXECUTION", "1")
    monkeypatch.setattr("ai_trading.services.execution.is_test_runtime", lambda: False)

    with pytest.raises(NonNettingLiveExecutionBlockedError, match="blocked for live non-netting execution"):
        submit_order(types.SimpleNamespace(), "AAPL", 1, "buy", price=100.0)


def test_execution_service_unknown_mode_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "mystery")
    monkeypatch.delenv("AI_TRADING_ALLOW_EXECUTION_MODE_SIM_FALLBACK", raising=False)
    monkeypatch.setattr("ai_trading.core.runtime_contract.is_testing_mode", lambda: False)

    with pytest.raises(UnknownExecutionModeError, match="EXECUTION_MODE must be one of"):
        submit_order(types.SimpleNamespace(), "AAPL", 1, "buy", price=100.0)


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
