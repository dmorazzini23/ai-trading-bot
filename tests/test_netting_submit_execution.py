from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.core.errors import ErrorCategory
from ai_trading.core.netting_submit_execution import execute_netting_submission


class _BreakersStub:
    def __init__(self) -> None:
        self.successes: list[str] = []
        self.failures: list[tuple[str, object]] = []

    def record_success(self, dep: str) -> None:
        self.successes.append(dep)

    def record_failure(self, dep: str, info: object) -> None:
        self.failures.append((dep, info))

    def open_reason(self, dep: str) -> str | None:
        _ = dep
        return "BREAKER_OPEN"


def _base_kwargs() -> dict[str, Any]:
    return {
        "runtime": SimpleNamespace(),
        "state": SimpleNamespace(),
        "symbol": "AAPL",
        "side": "buy",
        "price": 100.0,
        "delta_shares": 10,
        "now": datetime(2026, 4, 19, 15, 30, tzinfo=UTC),
        "net_target": SimpleNamespace(
            bar_ts=datetime(2026, 4, 19, 15, 29, tzinfo=UTC),
            proposals=[SimpleNamespace(sleeve="alpha", target_dollars=1000.0)],
        ),
        "approval": SimpleNamespace(expected_net_edge_bps=12.5),
        "intent": SimpleNamespace(to_contract=lambda: {"symbol": "AAPL", "side": "buy"}),
        "client_order_id": "cid-1",
        "decision_trace_id_for_order": "trace-1",
        "model_id_for_order": "m1",
        "model_version_for_order": "v1",
        "config_snapshot_hash_for_order": "cfg",
        "dataset_hash_for_order": "data",
        "feature_version_for_order": "feat",
        "model_artifact_hash_for_order": "artifact",
        "policy_hash_for_order": "policy",
        "order_annotations": {"price_source": "nbbo"},
        "order_lineage_metadata": {
            "decision_trace_id": "trace-1",
            "correlation_id": "opp-netting-1",
            "source_timestamp": "2026-04-19T15:29:00+00:00",
            "session_regime": "midday",
        },
        "submit_arrival_price": 100.0,
        "submit_bid_at_arrival": 99.5,
        "submit_ask_at_arrival": 100.5,
        "submit_mid_at_arrival": 100.0,
        "submit_quote_source": "nbbo",
        "candidate_expected_net_edge": {"AAPL": 10.0},
        "candidate_expected_capture": {"AAPL": 8.0},
        "ledger": None,
        "quarantine_enabled": False,
        "quarantine_manager": None,
        "extract_order_value_func": lambda order, *keys: getattr(order, keys[0], None),
        "extract_order_fill_timestamp_func": lambda order: getattr(order, "filled_at", None),
        "normalize_order_status_token_func": lambda status: str(status or "").lower(),
        "safe_float": lambda value: float(value) if value is not None else None,
        "has_persistable_fill_func": lambda **kwargs: True,
        "normalize_submitted_order_func": lambda order, **kwargs: SimpleNamespace(status_text="filled"),
        "record_successful_submission_func": lambda **kwargs: None,
        "build_order_metrics_and_tca_func": lambda **kwargs: ({"pnl": 1.0}, {"tca": True}),
        "submit_order_func": lambda *args, **kwargs: SimpleNamespace(status="filled"),
        "classify_exception_func": lambda exc, **kwargs: SimpleNamespace(
            reason_code="BROKER_SUBMIT_ERROR",
            category=ErrorCategory.ORDER_REJECTED,
        ),
        "handle_error_func": lambda *args, **kwargs: None,
        "trigger_quarantine_func": lambda **kwargs: None,
        "cancel_all_open_orders_oms_func": lambda runtime: SimpleNamespace(cancelled=1, failed=0),
        "resolve_submit_none_reason_func": lambda runtime: "DUPLICATE_INTENT",
        "record_auth_forbidden_cooldown_func": lambda *args, **kwargs: None,
        "get_regime_signal_profile_func": lambda: "normal",
        "normalize_quote_source_token_func": lambda token: token,
        "resolve_quote_proxy_source_func": lambda **kwargs: None,
        "resolved_tca_path_func": lambda: "runtime/tca.jsonl",
        "write_tca_record_func": lambda *args, **kwargs: None,
        "session_bucket_from_ts_func": lambda ts: "rth",
        "compute_attribution_metrics_func": lambda **kwargs: {"edge": 1.0},
        "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
        "breakers": _BreakersStub(),
    }


def test_execute_netting_submission_returns_blocked_on_exception() -> None:
    kwargs = _base_kwargs()

    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    kwargs["submit_order_func"] = _raise
    result = execute_netting_submission(**cast(Any, kwargs))

    assert result.status == "blocked"
    assert result.gates_added == ("BROKER_SUBMIT_ERROR",)
    assert result.attempted_increment == 1
    assert result.order_intent_contract == {"symbol": "AAPL", "side": "buy"}


def test_execute_netting_submission_returns_success_payload() -> None:
    kwargs = _base_kwargs()
    tca_kwargs: dict[str, Any] = {}
    kwargs["build_order_metrics_and_tca_func"] = lambda **call_kwargs: (
        tca_kwargs.update(call_kwargs) or ({"pnl": 1.0}, {"tca": True})
    )
    result = execute_netting_submission(**cast(Any, kwargs))

    assert result.status == "submitted"
    assert result.gates_added == ("OK_TRADE",)
    assert result.attempted_increment == 1
    assert result.submitted_increment == 1
    assert result.order_payload is not None
    assert result.order_payload["client_order_id"] == "cid-1"
    assert result.order_payload["correlation_id"] == "opp-netting-1"
    assert result.order_payload["decision_trace_id"] == "trace-1"
    assert result.metrics == {"pnl": 1.0}
    assert result.tca_record == {"tca": True}
    assert tca_kwargs["order_lineage_metadata"]["decision_trace_id"] == "trace-1"
    assert tca_kwargs["order_lineage_metadata"]["correlation_id"] == "opp-netting-1"
    assert tca_kwargs["order_lineage_metadata"]["expected_net_edge_bps"] == pytest.approx(12.5)


def test_execute_netting_submission_mirrors_expected_edge_evidence() -> None:
    kwargs = _base_kwargs()
    submitted: dict[str, Any] = {}

    def _submit_order(*args: Any, **call_kwargs: Any) -> SimpleNamespace:
        submitted.update(call_kwargs)
        return SimpleNamespace(status="filled")

    kwargs["submit_order_func"] = _submit_order

    result = execute_netting_submission(**cast(Any, kwargs))

    assert result.status == "submitted"
    assert submitted["expected_net_edge_bps"] == pytest.approx(12.5)
    assert submitted["annotations"]["expected_net_edge_bps"] == pytest.approx(12.5)
    assert submitted["metadata"]["expected_net_edge_bps"] == pytest.approx(12.5)


def test_execute_netting_submission_marks_short_cover_reduce_only() -> None:
    kwargs = _base_kwargs()
    submitted: dict[str, Any] = {}
    kwargs["side"] = "buy"
    kwargs["delta_shares"] = 4
    kwargs["net_target"] = SimpleNamespace(
        bar_ts=datetime(2026, 4, 19, 15, 29, tzinfo=UTC),
        target_shares=0,
        proposals=[SimpleNamespace(sleeve="alpha", target_dollars=0.0)],
    )

    def _submit_order(*args: Any, **call_kwargs: Any) -> SimpleNamespace:
        submitted.update(call_kwargs)
        return SimpleNamespace(status="filled")

    kwargs["submit_order_func"] = _submit_order

    result = execute_netting_submission(**cast(Any, kwargs))

    assert result.status == "submitted"
    assert submitted["closing_position"] is True
    assert submitted["reduce_only"] is True


def test_execute_netting_submission_marks_long_reducing_sell_reduce_only() -> None:
    kwargs = _base_kwargs()
    submitted: dict[str, Any] = {}
    kwargs["side"] = "sell"
    kwargs["delta_shares"] = -1
    kwargs["net_target"] = SimpleNamespace(
        bar_ts=datetime(2026, 4, 19, 15, 29, tzinfo=UTC),
        target_shares=0,
        proposals=[SimpleNamespace(sleeve="alpha", target_dollars=0.0)],
    )

    def _submit_order(*args: Any, **call_kwargs: Any) -> SimpleNamespace:
        submitted.update(call_kwargs)
        return SimpleNamespace(status="filled")

    kwargs["submit_order_func"] = _submit_order

    result = execute_netting_submission(**cast(Any, kwargs))

    assert result.status == "submitted"
    assert submitted["closing_position"] is True
    assert submitted["reduce_only"] is True


def test_execute_netting_submission_does_not_count_rejected_order_as_submitted() -> None:
    kwargs = _base_kwargs()
    recorded_success: list[dict[str, Any]] = []
    tca_calls: list[dict[str, Any]] = []
    kwargs["normalize_submitted_order_func"] = lambda order, **kwargs: SimpleNamespace(
        status_text="rejected",
        status_token="rejected",
    )
    kwargs["record_successful_submission_func"] = lambda **record_kwargs: recorded_success.append(
        dict(record_kwargs)
    )
    def _build_order_metrics_and_tca(**call_kwargs: Any) -> tuple[dict[str, str], dict[str, str]]:
        tca_calls.append(dict(call_kwargs))
        return ({"tca_status": "rejected"}, {"status": "rejected"})

    kwargs["build_order_metrics_and_tca_func"] = _build_order_metrics_and_tca

    result = execute_netting_submission(**cast(Any, kwargs))

    assert result.status == "rejected"
    assert result.gates_added == ("BROKER_ORDER_REJECTED",)
    assert result.attempted_increment == 1
    assert result.submitted_increment == 0
    assert result.metrics == {"tca_status": "rejected"}
    assert result.tca_record == {"status": "rejected"}
    assert recorded_success == []
    assert tca_calls
    assert tca_calls[0]["order_lineage_metadata"]["decision_trace_id"] == "trace-1"
    assert tca_calls[0]["order_lineage_metadata"]["expected_net_edge_bps"] == pytest.approx(12.5)


@pytest.mark.parametrize(
    ("status_token", "reason_code"),
    [
        ("failed", "BROKER_ORDER_FAILED"),
        ("error", "BROKER_ORDER_ERROR"),
    ],
)
def test_execute_netting_submission_does_not_count_failed_or_error_order_as_submitted(
    status_token: str,
    reason_code: str,
) -> None:
    kwargs = _base_kwargs()
    recorded_success: list[dict[str, Any]] = []
    cooldowns: list[dict[str, Any]] = []
    kwargs["normalize_submitted_order_func"] = lambda order, **kwargs: SimpleNamespace(
        status_text=status_token,
        status_token=status_token,
    )
    kwargs["record_successful_submission_func"] = lambda **record_kwargs: recorded_success.append(
        dict(record_kwargs)
    )
    kwargs["record_auth_forbidden_cooldown_func"] = (
        lambda *args, **call_kwargs: cooldowns.append(dict(call_kwargs))
    )

    result = execute_netting_submission(**cast(Any, kwargs))

    assert result.status == status_token
    assert result.gates_added == (reason_code,)
    assert result.attempted_increment == 1
    assert result.submitted_increment == 0
    assert recorded_success == []
    assert cooldowns[0]["reason"] == reason_code
