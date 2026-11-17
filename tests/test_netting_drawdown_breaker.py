"""The netting cycle feeds the drawdown breaker and gates entries on it."""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine
from ai_trading.core.netting import NettedTarget, SleeveConfig, SleeveProposal
from ai_trading.core.netting_execution_context import _NON_DISABLEABLE_GATE_NAMES
from ai_trading.core.netting_symbol_cycle import process_netting_symbol
from ai_trading.policy.compiler import ExecutionApproval
from ai_trading.risk.circuit_breakers import CircuitBreakerState, DrawdownCircuitBreaker
from tests.test_netting_symbol_cycle import _make_net_target, _make_processor

OPEN_BREAKER = {
    "open": True,
    "current_drawdown": 0.09,
    "max_drawdown": 0.08,
    "peak_equity": 100_000.0,
    "equity": 91_000.0,
}


def _messages(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.getMessage() == name]


def _breaker() -> DrawdownCircuitBreaker:
    # Same construction as LazyBotContext with MAX_DRAWDOWN_THRESHOLD=0.08.
    return DrawdownCircuitBreaker(max_drawdown=0.08, recovery_threshold=0.8)


# ---- feeding the breaker -------------------------------------------------


def test_first_reading_sets_the_baseline_and_cannot_trip() -> None:
    breaker = _breaker()

    payload = breaker.score_account_snapshot(SimpleNamespace(equity="81655.06"))

    assert payload == {
        "open": False,
        "current_drawdown": 0.0,
        "max_drawdown": 0.08,
        "peak_equity": 81655.06,
        "equity": 81655.06,
    }
    assert breaker.state is CircuitBreakerState.CLOSED


def test_feed_tracks_peak_and_drawdown_from_object_and_mapping_snapshots() -> None:
    breaker = _breaker()

    breaker.score_account_snapshot(SimpleNamespace(equity="100000"))
    breaker.score_account_snapshot({"equity": 104_000.0})
    payload = breaker.score_account_snapshot({"equity": "98800"})

    assert payload["open"] is False
    assert payload["peak_equity"] == 104_000.0
    assert payload["current_drawdown"] == pytest.approx(0.05)
    assert payload["equity"] == 98_800.0


def test_trip_at_threshold_logs_one_structured_line(caplog: pytest.LogCaptureFixture) -> None:
    breaker = _breaker()
    breaker.score_account_snapshot({"equity": 100_000.0})

    with caplog.at_level(logging.INFO):
        still_closed = breaker.score_account_snapshot({"equity": 92_000.01})
        tripped = breaker.score_account_snapshot({"equity": 92_000.0})
        still_open = breaker.score_account_snapshot({"equity": 91_000.0})

    assert still_closed["open"] is False
    assert tripped["open"] is True
    assert still_open["open"] is True
    trips = _messages(caplog, "DRAWDOWN_BREAKER_TRIPPED")
    assert len(trips) == 1
    assert trips[0].levelno == logging.CRITICAL
    assert trips[0].current_drawdown == pytest.approx(0.08)
    assert trips[0].max_drawdown == 0.08
    assert trips[0].peak_equity == 100_000.0


def test_reset_once_drawdown_is_back_under_threshold(caplog: pytest.LogCaptureFixture) -> None:
    breaker = _breaker()
    breaker.score_account_snapshot({"equity": 100_000.0})
    breaker.score_account_snapshot({"equity": 90_000.0})
    assert breaker.state is CircuitBreakerState.OPEN

    with caplog.at_level(logging.INFO):
        at_threshold = breaker.score_account_snapshot({"equity": 92_000.0})
        recovered = breaker.score_account_snapshot({"equity": 92_500.0})

    assert at_threshold["open"] is True
    assert recovered["open"] is False
    assert recovered["peak_equity"] == 100_000.0
    resets = _messages(caplog, "DRAWDOWN_BREAKER_RESET")
    assert len(resets) == 1
    assert resets[0].current_drawdown == pytest.approx(0.075)
    assert resets[0].peak_equity == 100_000.0


@pytest.mark.parametrize(
    ("snapshot", "reason"),
    [
        (None, "account_snapshot_unavailable"),
        (SimpleNamespace(cash="5"), "account_equity_missing"),
        ({"equity": None}, "account_equity_missing"),
        ({"equity": "n/a"}, "account_equity_invalid"),
        ({"equity": float("nan")}, "account_equity_invalid"),
        ({"equity": float("inf")}, "account_equity_invalid"),
        ({"equity": 0}, "account_equity_invalid"),
        ({"equity": "-250.0"}, "account_equity_invalid"),
        ({"equity": True}, "account_equity_invalid"),
        ({"equity": object()}, "account_equity_invalid"),
    ],
)
@pytest.mark.parametrize("start_open", [False, True])
def test_unusable_equity_leaves_the_breaker_unchanged(
    caplog: pytest.LogCaptureFixture, snapshot: Any, reason: str, start_open: bool
) -> None:
    breaker = _breaker()
    breaker.score_account_snapshot({"equity": 100_000.0})
    breaker.score_account_snapshot({"equity": 90_000.0 if start_open else 99_000.0})
    before = breaker.get_status()
    assert before["trading_allowed"] is not start_open
    caplog.clear()

    with caplog.at_level(logging.INFO):
        payload = breaker.score_account_snapshot(snapshot)

    assert breaker.get_status() == before
    assert payload["open"] is start_open
    assert payload["equity"] is None
    skipped = _messages(caplog, "DRAWDOWN_BREAKER_FEED_SKIPPED")
    assert len(skipped) == 1
    assert skipped[0].levelno == logging.WARNING
    assert skipped[0].reason == reason
    assert skipped[0].breaker_state == ("open" if start_open else "closed")
    assert not _messages(caplog, "DRAWDOWN_BREAKER_TRIPPED")
    assert not _messages(caplog, "DRAWDOWN_BREAKER_RESET")


# ---- the cycle's feed ----------------------------------------------------


def test_cycle_feed_uses_the_engine_cached_snapshot_only() -> None:
    breaker = _breaker()
    calls: list[str] = []

    class _Api:
        def get_account(self) -> Any:
            raise AssertionError("the cycle feed must not make its own broker call")

    def _cached() -> Any:
        calls.append("cached")
        return SimpleNamespace(equity="100000")

    runtime = SimpleNamespace(drawdown_circuit_breaker=breaker, api=_Api())

    payload = bot_engine._score_drawdown_breaker(
        runtime, SimpleNamespace(_get_account_snapshot=_cached)
    )

    assert calls == ["cached"]
    assert payload["open"] is False
    assert breaker.peak_equity == 100_000.0


def test_cycle_feed_without_a_breaker_is_closed() -> None:
    assert bot_engine._score_drawdown_breaker(SimpleNamespace(), None) == {"open": False}
    assert bot_engine._score_drawdown_breaker(
        SimpleNamespace(drawdown_circuit_breaker=None), None
    ) == {"open": False}


@pytest.mark.parametrize("start_open", [False, True])
def test_cycle_feed_survives_a_failing_or_absent_account_snapshot(
    caplog: pytest.LogCaptureFixture, start_open: bool
) -> None:
    breaker = _breaker()
    breaker.score_account_snapshot({"equity": 100_000.0})
    breaker.score_account_snapshot({"equity": 90_000.0 if start_open else 99_000.0})
    before = breaker.get_status()
    runtime = SimpleNamespace(drawdown_circuit_breaker=breaker)

    def _boom() -> Any:
        raise ConnectionError("broker unreachable")

    with caplog.at_level(logging.INFO):
        failing = bot_engine._score_drawdown_breaker(
            runtime, SimpleNamespace(_get_account_snapshot=_boom)
        )
        absent = bot_engine._score_drawdown_breaker(runtime, SimpleNamespace())

    assert failing["open"] is start_open
    assert absent["open"] is start_open
    assert breaker.get_status() == before
    assert len(_messages(caplog, "DRAWDOWN_BREAKER_ACCOUNT_SNAPSHOT_FAILED")) == 1
    assert len(_messages(caplog, "DRAWDOWN_BREAKER_FEED_SKIPPED")) == 2


# ---- the per-symbol gate -------------------------------------------------


def _process(processor: Any, *, target_dollars: float) -> Any:
    return process_netting_symbol(
        processor=processor,
        symbol="AAPL",
        net_target=_make_net_target(bar_ts=processor.now, target_dollars=target_dollars),
        orders_submitted=0,
    )


def _no_submit(**_kwargs: Any) -> Any:
    raise AssertionError("a blocked entry must not reach submission")


@pytest.mark.parametrize(
    ("positions", "target_dollars"),
    [
        ({}, 100.0),  # open long from flat
        ({"AAPL": 5}, 100.0),  # add to a long
        ({"AAPL": -5}, -100.0),  # add to a short
    ],
)
def test_open_breaker_blocks_exposure_increasing_orders(
    positions: dict[str, int], target_dollars: float
) -> None:
    processor, records = _make_processor(
        drawdown_breaker=dict(OPEN_BREAKER),
        positions=positions,
        cfg=SimpleNamespace(seed="seed", shorts_allowed=True),
        execute_submission_func=_no_submit,
        apply_symbol_adjustments_func=_no_submit,
    )

    result = _process(processor, target_dollars=target_dollars)

    assert result.attempted_increment == 0
    assert result.submitted_increment == 0
    assert len(records) == 1
    assert records[0]["gates"] == ["DRAWDOWN_BREAKER_BLOCK"]
    assert records[0]["metrics"]["terminal_stage"] == "decision_filter"
    assert records[0]["metrics"]["terminal_reason"] == "DRAWDOWN_BREAKER_BLOCK"
    assert records[0]["config_snapshot"]["drawdown_breaker"] == OPEN_BREAKER
    assert "order" not in records[0] or not records[0]["order"]


@pytest.mark.parametrize(
    ("positions", "target_dollars", "expected_delta"),
    [
        ({"AAPL": 10}, 0.0, -10),  # full exit of a long
        ({"AAPL": 10}, 40.0, -6),  # partial reduction of a long
        ({"AAPL": -10}, 0.0, 10),  # cover a short
    ],
)
def test_open_breaker_lets_exposure_reducing_orders_through(
    positions: dict[str, int], target_dollars: float, expected_delta: int
) -> None:
    submitted: list[int] = []
    open_processor, open_records = _make_processor(
        drawdown_breaker=dict(OPEN_BREAKER), positions=dict(positions)
    )
    closed_processor, closed_records = _make_processor(positions=dict(positions))
    real_submit = open_processor.execute_submission_func

    def _submit(**kwargs: Any) -> Any:
        submitted.append(int(kwargs["delta_shares"]))
        return real_submit(**kwargs)

    open_processor.execute_submission_func = _submit

    open_result = _process(open_processor, target_dollars=target_dollars)
    closed_result = _process(closed_processor, target_dollars=target_dollars)

    assert submitted == [expected_delta]
    assert open_result.submitted_increment == 1
    assert open_result == closed_result
    assert open_records[0]["gates"] == closed_records[0]["gates"] == ["OK_TRADE"]
    assert "drawdown_breaker" not in open_records[0]["config_snapshot"]
    assert open_records[0]["config_snapshot"] == closed_records[0]["config_snapshot"]


def test_open_breaker_turns_a_reversal_into_an_exit() -> None:
    submitted: list[tuple[int, int]] = []
    processor, records = _make_processor(
        drawdown_breaker=dict(OPEN_BREAKER),
        positions={"AAPL": 10},
        cfg=SimpleNamespace(seed="seed", shorts_allowed=True),
    )
    real_submit = processor.execute_submission_func

    def _submit(**kwargs: Any) -> Any:
        submitted.append((int(kwargs["delta_shares"]), int(kwargs["net_target"].target_shares)))
        return real_submit(**kwargs)

    processor.execute_submission_func = _submit

    result = _process(processor, target_dollars=-200.0)

    assert result.submitted_increment == 1
    assert submitted == [(-10, 0)]
    assert records[0]["gates"] == ["DRAWDOWN_BREAKER_REDUCE_ONLY", "OK_TRADE"]
    assert records[0]["net_target"].target_dollars == 0.0
    assert records[0]["config_snapshot"]["drawdown_breaker"] == OPEN_BREAKER


def test_stop_lock_exit_is_not_held_back_by_an_open_breaker() -> None:
    submitted: list[int] = []
    processor, records = _make_processor(
        drawdown_breaker=dict(OPEN_BREAKER), positions={"AAPL": 5}
    )
    processor.state.stop_lock["AAPL"] = {"bar_ts": processor.now, "direction": "long"}
    real_submit = processor.execute_submission_func

    def _submit(**kwargs: Any) -> Any:
        submitted.append(int(kwargs["delta_shares"]))
        return real_submit(**kwargs)

    processor.execute_submission_func = _submit

    # The sleeves want to add to the long; the stop lock flattens it instead.
    result = _process(processor, target_dollars=100.0)

    assert result.submitted_increment == 1
    assert submitted == [-5]
    assert records[0]["gates"] == ["STOP_LOCK", "OK_TRADE"]


@pytest.mark.parametrize("payload", [{}, {"open": False}, {**OPEN_BREAKER, "open": False}])
def test_closed_breaker_leaves_symbol_decisions_unchanged(payload: dict[str, Any]) -> None:
    for positions, target_dollars in (({}, 100.0), ({"AAPL": 10}, 0.0), ({"AAPL": 5}, 100.0)):
        baseline_processor, baseline_records = _make_processor(positions=dict(positions))
        processor, records = _make_processor(
            drawdown_breaker=dict(payload), positions=dict(positions)
        )

        baseline = _process(baseline_processor, target_dollars=target_dollars)
        result = _process(processor, target_dollars=target_dollars)

        assert result == baseline
        assert [record["gates"] for record in records] == [
            record["gates"] for record in baseline_records
        ]
        assert [record["config_snapshot"] for record in records] == [
            record["config_snapshot"] for record in baseline_records
        ]


def test_kill_switch_keeps_precedence_over_the_breaker() -> None:
    processor, records = _make_processor(drawdown_breaker=dict(OPEN_BREAKER), kill_switch=True)

    _process(processor, target_dollars=100.0)

    assert records[0]["gates"] == ["KILL_SWITCH_BLOCK"]


def test_breaker_gate_cannot_be_auto_disabled() -> None:
    assert "DRAWDOWN_BREAKER_BLOCK" in _NON_DISABLEABLE_GATE_NAMES
    processor, records = _make_processor(
        drawdown_breaker=dict(OPEN_BREAKER),
        ineffective_gate_blocklist={"DRAWDOWN_BREAKER_BLOCK"},
    )

    result = _process(processor, target_dollars=100.0)

    assert result.submitted_increment == 0
    assert records[0]["gates"] == ["DRAWDOWN_BREAKER_BLOCK"]


# ---- the whole netting cycle ---------------------------------------------

_TARGET_DOLLARS = {"AAPL": 1010.0, "MSFT": 0.0}
_POSITIONS = {"MSFT": 15}


class _AllowBreakers:
    def allow(self, dep: str) -> bool:
        return True

    def open_reason(self, dep: str) -> str | None:
        return None

    def record_failure(self, dep: str, error_info: Any) -> None:
        _ = (dep, error_info)

    def record_success(self, dep: str) -> None:
        _ = dep


class _NoBarDedupeDict(dict[str, datetime]):
    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        _ = (key, default)
        return None


def _run_cycle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    breaker: DrawdownCircuitBreaker | None,
    account: Any,
) -> tuple[list[tuple[str, int, str]], dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Run one real netting cycle: AAPL is an entry from flat, MSFT a full exit."""

    cfg = TradingConfig.from_env(allow_missing_drawdown=True).update(
        netting_enabled=True,
        data_contract_enabled=False,
        recon_enabled=False,
        ledger_enabled=False,
        rth_only=False,
        allow_extended=True,
        decision_log_path=None,
    )
    account_reads: list[int] = []

    def _cached_account() -> Any:
        account_reads.append(1)
        return account

    runtime = SimpleNamespace(
        cfg=cfg,
        tickers=["AAPL", "MSFT"],
        universe_tickers=["AAPL", "MSFT"],
        api=None,
        execution_engine=SimpleNamespace(
            _last_submit_outcome={}, _get_account_snapshot=_cached_account
        ),
    )
    if breaker is not None:
        runtime.drawdown_circuit_breaker = breaker
    state = bot_engine.BotState()
    state.last_order_bar_ts = _NoBarDedupeDict()

    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1000],
        },
        index=pd.DatetimeIndex([now - timedelta(minutes=1), now], tz=UTC),
    )

    def _force_target(symbol: str, bar_ts: datetime, proposals: Any, disagree_ratio: Any) -> NettedTarget:
        return NettedTarget(
            symbol=str(symbol),
            bar_ts=bar_ts,
            target_dollars=_TARGET_DOLLARS[str(symbol).upper()],
            target_shares=0.0,
            proposals=list(proposals),
            disagreement_ratio=0.0 if disagree_ratio is None else float(disagree_ratio),
        )

    def _proposal(sleeve: Any, symbol: str, bar_ts: datetime, *_args: Any, **_kwargs: Any) -> SleeveProposal:
        entry = _TARGET_DOLLARS[str(symbol).upper()] > 0
        return SleeveProposal(
            symbol=str(symbol),
            sleeve=sleeve.name,
            bar_ts=bar_ts,
            target_dollars=_TARGET_DOLLARS[str(symbol).upper()],
            expected_edge_bps=8.0,
            expected_cost_bps=2.0,
            score=0.5 if entry else -0.5,
            confidence=0.8,
        )

    submitted: list[tuple[str, int, str]] = []

    def _fake_submit(runtime_obj: Any, symbol: str, qty: Any, side: str, **_kwargs: Any) -> Any:
        submitted.append((str(symbol), int(qty), str(side)))
        runtime_obj.execution_engine._last_submit_outcome = {
            "status": "submitted",
            "symbol": str(symbol),
            "side": str(side),
        }
        return SimpleNamespace(id=f"order-{symbol}", status="submitted", filled_avg_price=None)

    for key, value in {
        "AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE": "0",
        "AI_TRADING_KILL_SWITCH": "0",
        "AI_TRADING_CANARY_SYMBOLS": "AAPL,MSFT",
        "AI_TRADING_WARMUP_MODE": "0",
        "AI_TRADING_EVENT_DRIVEN_NEW_BAR_ONLY": "0",
        "AI_TRADING_NETTING_CYCLE_SLO_LOG_TTL_SEC": "0",
        "AI_TRADING_DERISK_ON_SLO_BREACH_ENABLED": "0",
        "AI_TRADING_LIQ_REGIME_ENABLED": "0",
        "AI_TRADING_CAPACITY_AWARE_THROTTLE_ENABLED": "0",
        "AI_TRADING_EVENT_RISK_BLACKOUT_ENABLED": "0",
        "AI_TRADING_QUARANTINE_ENABLED": "0",
        "AI_TRADING_PRIMARY_FEED_DERISK_ENABLED": "0",
        "AI_TRADING_PARTICIPATION_CAP_ENABLED": "0",
        "AI_TRADING_ALPHA_DECAY_DEWEIGHT_ENABLED": "0",
        "AI_TRADING_EXECUTION_COST_AWARE_ENTRY_GUARD_ENABLED": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        "ai_trading.data.fetch.get_bars_batch",
        lambda symbols, timeframe, start, end: {str(sym).upper(): df for sym in symbols},
    )
    monkeypatch.setattr("ai_trading.core.netting.net_targets_for_symbol", _force_target)
    monkeypatch.setattr(
        "ai_trading.core.horizons.build_sleeve_configs",
        lambda cfg=None: [
            SleeveConfig(
                name="day",
                timeframe="1Min",
                enabled=True,
                entry_threshold=0.2,
                exit_threshold=0.1,
                flip_threshold=0.3,
                reentry_threshold=0.6,
                deadband_dollars=50.0,
                deadband_shares=1.0,
                turnover_cap_dollars=0.0,
                cost_k=1.5,
                edge_scale_bps=20.0,
                max_symbol_dollars=10000.0,
                max_gross_dollars=50000.0,
            )
        ],
    )
    monkeypatch.setattr("ai_trading.core.netting.compute_sleeve_proposal", _proposal)
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda cfg: (False, None))
    monkeypatch.setattr(bot_engine, "_dependency_breakers", lambda _state: _AllowBreakers())
    monkeypatch.setattr(
        "ai_trading.oms.ledger.deterministic_client_order_id",
        lambda **kwargs: (
            f"{kwargs['symbol']}-{kwargs['bar_ts']}-{kwargs['side']}-{int(abs(float(kwargs['qty'])))}"
        ),
    )
    monkeypatch.setattr(
        "ai_trading.oms.ledger.OrderLedger.seen_client_order_id",
        lambda self, client_order_id: False,
    )
    monkeypatch.setattr("ai_trading.oms.ledger.OrderLedger.record", lambda self, entry: None)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda _now=None: True)
    monkeypatch.setattr(bot_engine, "retry_idempotent", lambda fn, **_kwargs: fn())
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda runtime_obj: None)
    monkeypatch.setattr(bot_engine, "compute_current_positions", lambda runtime_obj: dict(_POSITIONS))
    monkeypatch.setattr(bot_engine, "check_daily_loss", lambda runtime_obj, state_obj: False)
    monkeypatch.setattr(bot_engine, "check_weekly_loss", lambda runtime_obj, state_obj: False)
    monkeypatch.setattr(bot_engine, "_run_reconciliation_if_due", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        bot_engine, "_pre_rank_execution_candidates", lambda symbols, runtime=None: list(symbols)
    )
    monkeypatch.setattr(
        bot_engine,
        "approve_execution_candidate",
        lambda policy, candidate: ExecutionApproval(
            True, int(candidate.proposed_delta_shares), 5.0, tuple()
        ),
    )
    for name in (
        "_run_post_trade_learning_update",
        "_run_tca_cost_calibration",
        "_run_replay_governance",
        "_run_walk_forward_governance",
    ):
        monkeypatch.setattr(bot_engine, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bot_engine, "_update_rollout_governance_state", lambda *args, **kwargs: {"capital_ramp": {}}
    )
    monkeypatch.setattr(bot_engine, "_resolve_primary_feed_derisk_state", lambda _runtime: {})
    monkeypatch.setattr(
        bot_engine,
        "enforce_participation_cap",
        lambda **kwargs: (True, float(kwargs["order_qty"]), None),
    )
    monkeypatch.setattr(
        bot_engine,
        "_alpha_decay_entry_guard",
        lambda *args, **kwargs: {"blocked": False, "trades_in_window": 0, "start_trades": 0},
    )
    monkeypatch.setattr(bot_engine, "is_near_event", lambda _symbol, days=0: False)
    monkeypatch.setattr(bot_engine, "_tca_stale_block_reason", lambda _now: None)
    monkeypatch.setattr(
        bot_engine, "safe_validate_pretrade", lambda *args, **kwargs: (True, "OK", {})
    )
    monkeypatch.setattr(bot_engine, "submit_order", _fake_submit)
    written: list[Any] = []
    monkeypatch.setattr(
        bot_engine, "_write_decision_record", lambda record, path: written.append(record)
    )

    bot_engine._run_netting_cycle(state, runtime, "loop", 0.0)

    assert state.halt_trading is False
    gates = {str(record.symbol).upper(): list(record.gates or []) for record in written}
    snapshots = {
        str(record.symbol).upper(): dict(record.config_snapshot or {}) for record in written
    }
    for snapshot in snapshots.values():
        # Carries a process-wide sample counter that grows with every cycle run.
        snapshot.pop("uncertainty_capital_scaling", None)
    return submitted, gates, snapshots


def test_netting_cycle_with_tripped_breaker_blocks_entry_and_still_exits(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    breaker = _breaker()
    breaker.score_account_snapshot({"equity": 100_000.0})

    with caplog.at_level(logging.INFO):
        submitted, gates, snapshots = _run_cycle(
            monkeypatch, breaker=breaker, account=SimpleNamespace(id="acct-1", equity="91000")
        )

    assert len(_messages(caplog, "DRAWDOWN_BREAKER_TRIPPED")) == 1
    assert breaker.state is CircuitBreakerState.OPEN
    assert submitted == [("MSFT", 15, "sell")]
    assert gates["AAPL"][-1] == "DRAWDOWN_BREAKER_BLOCK"
    assert snapshots["AAPL"]["drawdown_breaker"] == {
        "open": True,
        "current_drawdown": pytest.approx(0.09),
        "max_drawdown": 0.08,
        "peak_equity": 100_000.0,
        "equity": 91_000.0,
    }
    assert "DRAWDOWN_BREAKER_BLOCK" not in gates["MSFT"]
    assert "OK_TRADE" in gates["MSFT"]
    assert "drawdown_breaker" not in snapshots["MSFT"]


def test_netting_cycle_decisions_are_unchanged_while_the_breaker_is_closed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    account = SimpleNamespace(id="acct-1", equity="99000")
    breaker = _breaker()
    breaker.score_account_snapshot({"equity": 100_000.0})

    with monkeypatch.context() as patch:
        without_breaker = _run_cycle(patch, breaker=None, account=account)
    with caplog.at_level(logging.INFO), monkeypatch.context() as patch:
        with_breaker = _run_cycle(patch, breaker=breaker, account=account)

    assert with_breaker == without_breaker
    submitted, gates, snapshots = with_breaker
    assert [(symbol, side) for symbol, _qty, side in submitted] == [
        ("AAPL", "buy"),
        ("MSFT", "sell"),
    ]
    assert "OK_TRADE" in gates["AAPL"]
    assert all("drawdown_breaker" not in snapshot for snapshot in snapshots.values())
    assert breaker.state is CircuitBreakerState.CLOSED
    assert breaker.current_drawdown == pytest.approx(0.01)
    assert not _messages(caplog, "DRAWDOWN_BREAKER_TRIPPED")
    assert not _messages(caplog, "DRAWDOWN_BREAKER_FEED_SKIPPED")


def test_netting_cycle_with_unusable_equity_does_not_trip_or_crash(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    breaker = _breaker()
    breaker.score_account_snapshot({"equity": 100_000.0})

    with caplog.at_level(logging.INFO):
        submitted, gates, _snapshots = _run_cycle(
            monkeypatch, breaker=breaker, account=SimpleNamespace(id="acct-1", equity="0")
        )

    assert breaker.state is CircuitBreakerState.CLOSED
    assert breaker.peak_equity == 100_000.0
    assert [(symbol, side) for symbol, _qty, side in submitted] == [
        ("AAPL", "buy"),
        ("MSFT", "sell"),
    ]
    assert "DRAWDOWN_BREAKER_BLOCK" not in gates["AAPL"]
    skipped = _messages(caplog, "DRAWDOWN_BREAKER_FEED_SKIPPED")
    assert len(skipped) == 1
    assert skipped[0].reason == "account_equity_invalid"
