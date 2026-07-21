from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from ai_trading.core import execution_outcome
from ai_trading.core.execution_outcome import SubmittedOrderState


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _env_getter(values: Mapping[str, Any]) -> Any:
    def _get_env(key: str, default: Any = None, *, cast: Any = None) -> Any:
        value = values.get(key, default)
        if cast is bool:
            return bool(value)
        if cast is int:
            return int(value)
        return value

    return _get_env


def test_normalize_submitted_order_defaults_pending_payload() -> None:
    result = execution_outcome.normalize_submitted_order(
        None,
        delta_shares=-7,
        extract_order_value=lambda _obj, *_fields: None,
        extract_order_fill_timestamp=lambda _obj: None,
        normalize_order_status_token=lambda value: str(value or "submitted"),
        safe_float=_safe_float,
        has_persistable_fill=lambda **_kwargs: False,
    )

    assert result.status_text == "submitted"
    assert result.status_token == "submitted"
    assert result.broker_order_id is None
    assert result.filled_qty == 0.0
    assert result.requested_qty == 7.0
    assert result.fill_price is None
    assert result.fill_fees == 0.0
    assert result.persistable_fill is False


def test_normalize_submitted_order_does_not_conflate_client_and_broker_ids() -> None:
    order = SimpleNamespace(client_order_id="client-only", status="accepted")

    result = execution_outcome.normalize_submitted_order(
        order,
        delta_shares=2,
        extract_order_value=lambda obj, *fields: next(
            (
                getattr(obj, field, None)
                for field in fields
                if getattr(obj, field, None) is not None
            ),
            None,
        ),
        extract_order_fill_timestamp=lambda _obj: None,
        normalize_order_status_token=lambda value: str(value or "submitted"),
        safe_float=_safe_float,
        has_persistable_fill=lambda **_kwargs: False,
    )

    assert result.broker_order_id is None


def test_record_successful_submission_without_ledger_or_turnover_target() -> None:
    state = SimpleNamespace(last_order_bar_ts={}, last_order_client_id={}, turnover_dollars={})
    now = datetime(2026, 4, 25, 16, tzinfo=UTC)

    execution_outcome.record_successful_submission(
        ledger=None,
        state=state,
        symbol="AAPL",
        client_order_id="client-1",
        bar_ts=now - timedelta(minutes=1),
        delta_shares=3,
        side="buy",
        price=100.0,
        now=now,
        order_state=SubmittedOrderState(
            status_text="accepted",
            status_token="accepted",
            broker_order_id=None,
            filled_qty=0.0,
            requested_qty=3.0,
            fill_price=None,
            fill_timestamp=None,
            fill_fees=0.0,
            persistable_fill=False,
        ),
        proposals=[SimpleNamespace(sleeve="core", target_dollars=0.0)],
    )

    assert state.last_order_client_id == {"AAPL": "client-1"}
    assert state.turnover_dollars == {}


def test_build_order_metrics_returns_without_tca_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 4, 25, 16, tzinfo=UTC)
    monkeypatch.setattr(
        execution_outcome,
        "get_env",
        _env_getter({"AI_TRADING_TCA_ENABLED": False}),
    )

    metrics, tca_record = execution_outcome.build_order_metrics_and_tca(
        symbol="AAPL",
        side="buy",
        price=100.0,
        delta_shares=5,
        now=now,
        net_target=SimpleNamespace(bar_ts=now - timedelta(minutes=1), proposals=[]),
        order=SimpleNamespace(client_order_id="client-1"),
        order_state=SubmittedOrderState(
            status_text="filled",
            status_token="filled",
            broker_order_id="broker-1",
            filled_qty=5.0,
            requested_qty=5.0,
            fill_price=100.5,
            fill_timestamp=now,
            fill_fees=1.0,
            persistable_fill=True,
        ),
        submit_arrival_price=None,
        submit_bid_at_arrival=99.9,
        submit_ask_at_arrival=100.1,
        submit_mid_at_arrival=100.0,
        submit_quote_source="sip",
        candidate_expected_net_edge={},
        candidate_expected_capture={},
        get_regime_signal_profile_func=lambda: "regular",
        normalize_quote_source_token_func=lambda value: str(value) if value else None,
        resolve_quote_proxy_source_func=lambda *_args, **_kwargs: "last_trade",
        resolved_tca_path_func=lambda: "unused.jsonl",
        write_tca_record_func=lambda _path, _payload: None,
        session_bucket_from_ts_func=lambda _ts: "regular",
        compute_attribution_metrics_func=lambda **_kwargs: {"is_bps": 3.0},
        safe_float=_safe_float,
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    )

    assert metrics == {
        "is_bps": 3.0,
        "evidence_type": "fill_execution",
        "fill_based_evidence": True,
        "promotion_eligible": True,
    }
    assert tca_record is None


def test_build_order_metrics_writes_pending_tca_record(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 4, 25, 16, tzinfo=UTC)
    writes: list[tuple[str, Mapping[str, Any]]] = []
    monkeypatch.setattr(
        execution_outcome,
        "get_env",
        _env_getter(
            {
                "AI_TRADING_TCA_ENABLED": True,
                "AI_TRADING_TCA_ARRIVAL_BENCHMARK": "submit",
                "AI_TRADING_TCA_ALLOW_PROXY_QUOTES": True,
                "AI_TRADING_TCA_PENDING_WRITE_SEC": 12,
                "AI_TRADING_TCA_UPDATE_ON_FILL": True,
                "AI_TRADING_TCA_WRITE_PENDING_EVENTS": True,
                "AI_TRADING_TCA_PATH": "runtime/tca.jsonl",
                "AI_TRADING_TCA_PROXY_MID_SOURCE": "last_trade",
            }
        ),
    )

    metrics, tca_record = execution_outcome.build_order_metrics_and_tca(
        symbol="MSFT",
        side="buy",
        price=100.0,
        delta_shares=10,
        now=now,
        net_target=SimpleNamespace(
            bar_ts=now - timedelta(minutes=1),
            reasons=["EDGE_RANKED"],
            proposals=[SimpleNamespace(sleeve="core")],
        ),
        order=SimpleNamespace(client_order_id="client-2", exchange="iex"),
        order_state=SubmittedOrderState(
            status_text="accepted",
            status_token="accepted",
            broker_order_id="broker-2",
            filled_qty=0.0,
            requested_qty=10.0,
            fill_price=None,
            fill_timestamp=None,
            fill_fees=0.0,
            persistable_fill=False,
        ),
        submit_arrival_price=99.8,
        submit_bid_at_arrival=None,
        submit_ask_at_arrival=None,
        submit_mid_at_arrival=None,
        submit_quote_source=None,
        candidate_expected_net_edge={"MSFT": "7.5"},
        candidate_expected_capture={"MSFT": "2.25"},
        get_regime_signal_profile_func=lambda: "regular",
        normalize_quote_source_token_func=lambda _value: None,
        resolve_quote_proxy_source_func=lambda _order, **_kwargs: "last_trade",
        resolved_tca_path_func=lambda: "resolved/path.jsonl",
        write_tca_record_func=lambda path, payload: writes.append((path, dict(payload))),
        session_bucket_from_ts_func=lambda _ts: "regular",
        compute_attribution_metrics_func=lambda **_kwargs: {"arrival": 99.8},
        safe_float=_safe_float,
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        order_lineage_metadata={
            "model_id": "ml-main",
            "model_version": "v1",
            "config_snapshot_hash": "cfg-1",
            "rank_reasons": ["CAPTURE_OK"],
            "client_order_id": "client-2",
            "broker_order_id": "broker-2",
            "order_type": "market",
            "time_in_force": "day",
            "decision_quote_age_ms": 175.0,
            "decision_spread_bps": 2.5,
            "session_regime": "opening",
            "execution_profile": "balanced",
            "market_regime": "sideways",
            "volatility_regime": "low",
            "trend_regime": "flat",
            "correlation_id": "opp-msft-pending",
            "decision_trace_id": "trace-msft-pending",
            "source_timestamp": (now - timedelta(minutes=1)).isoformat(),
            "quote_timestamp": (now - timedelta(seconds=1)).isoformat(),
            "paper_sampling_reservation_token": "sampling-token-1",
        },
    )

    assert tca_record is not None
    assert tca_record["fill_price"] is None
    assert tca_record["fill_vwap"] is None
    assert tca_record["arrival_benchmark"] == "submit"
    assert tca_record["pending_write_sec"] == 12
    assert tca_record["quote_proxy_source"] == "last_trade"
    assert tca_record["expected_net_edge_bps"] == pytest.approx(7.5)
    assert tca_record["expected_capture_bps"] == pytest.approx(2.25)
    assert tca_record["venue_session"] == "IEX:opening"
    assert tca_record["model_id"] == "ml-main"
    assert tca_record["model_version"] == "v1"
    assert tca_record["config_snapshot_hash"] == "cfg-1"
    assert tca_record["rank_reason"] == "EDGE_RANKED"
    assert tca_record["rank_reasons"] == ["EDGE_RANKED", "CAPTURE_OK"]
    assert tca_record["client_order_id"] == "client-2"
    assert tca_record["broker_order_id"] == "broker-2"
    assert tca_record["order_type"] == "market"
    assert tca_record["time_in_force"] == "day"
    assert tca_record["quote_age_ms"] == pytest.approx(175.0)
    assert tca_record["spread_bps"] == pytest.approx(2.5)
    assert tca_record["session"] == "opening"
    assert tca_record["execution_profile"] == "balanced"
    assert tca_record["regime_profile"] == "regular"
    assert tca_record["market_regime"] == "sideways"
    assert tca_record["volatility_regime"] == "low"
    assert tca_record["trend_regime"] == "flat"
    assert tca_record["correlation_id"] == "opp-msft-pending"
    assert tca_record["decision_trace_id"] == "trace-msft-pending"
    assert tca_record["paper_sampling_reservation_token"] == "sampling-token-1"
    assert tca_record["fill_based_evidence"] is False
    assert tca_record["promotion_eligible"] is False
    assert metrics["tca"] == {
        "is_bps": None,
        "spread_paid_bps": None,
        "fill_latency_ms": None,
    }
    assert writes[0][0] == "resolved/path.jsonl"
    assert writes[0][1]["pending_event"] is True
    assert writes[0][1]["pending_reason"] == "accepted"


def test_build_order_metrics_filled_tca_roles_and_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 4, 25, 16, tzinfo=UTC)
    warnings: list[str] = []
    monkeypatch.setattr(
        execution_outcome,
        "get_env",
        _env_getter(
            {
                "AI_TRADING_TCA_ENABLED": True,
                "AI_TRADING_TCA_ARRIVAL_BENCHMARK": "invalid",
                "AI_TRADING_TCA_ALLOW_PROXY_QUOTES": True,
                "AI_TRADING_TCA_PENDING_WRITE_SEC": 60,
                "AI_TRADING_TCA_UPDATE_ON_FILL": True,
                "AI_TRADING_TCA_WRITE_PENDING_EVENTS": False,
                "AI_TRADING_TCA_PATH": "runtime/tca.jsonl",
            }
        ),
    )

    metrics, tca_record = execution_outcome.build_order_metrics_and_tca(
        symbol="AAPL",
        side="buy",
        price=100.0,
        delta_shares=10,
        now=now,
        net_target=SimpleNamespace(
            bar_ts=now - timedelta(minutes=1),
            proposals=[SimpleNamespace(sleeve="core")],
        ),
        order=SimpleNamespace(client_order_id="client-3", venue="arcx"),
        order_state=SubmittedOrderState(
            status_text="partially_filled",
            status_token="partially_filled",
            broker_order_id="broker-3",
            filled_qty=5.0,
            requested_qty=10.0,
            fill_price=100.002,
            fill_timestamp=now + timedelta(seconds=1),
            fill_fees=0.5,
            persistable_fill=True,
        ),
        submit_arrival_price=100.0,
        submit_bid_at_arrival=99.99,
        submit_ask_at_arrival=100.01,
        submit_mid_at_arrival=100.0,
        submit_quote_source="sip",
        candidate_expected_net_edge={},
        candidate_expected_capture={},
        get_regime_signal_profile_func=lambda: "regular",
        normalize_quote_source_token_func=lambda value: str(value).lower() if value else None,
        resolve_quote_proxy_source_func=lambda *_args, **_kwargs: None,
        resolved_tca_path_func=lambda: "resolved/path.jsonl",
        write_tca_record_func=lambda _path, _payload: (_ for _ in ()).throw(RuntimeError("disk")),
        session_bucket_from_ts_func=lambda _ts: "regular",
        compute_attribution_metrics_func=lambda **_kwargs: {"arrival": 100.0},
        safe_float=_safe_float,
        logger=SimpleNamespace(warning=lambda msg, *_args, **_kwargs: warnings.append(str(msg))),
    )

    assert tca_record is not None
    assert tca_record["arrival_benchmark"] == "decision"
    assert tca_record["liquidity_role"] == "mixed"
    assert tca_record["venue"] == "ARCX"
    assert tca_record["partial_fill"] is True
    assert metrics["tca"]["fill_latency_ms"] == 1000
    assert warnings == ["TCA_WRITE_FAILED path=%s error=%s"]


def test_build_order_metrics_writes_terminal_nonfill_tca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 4, 25, 16, tzinfo=UTC)
    writes: list[tuple[str, Mapping[str, Any]]] = []
    monkeypatch.setattr(
        execution_outcome,
        "get_env",
        _env_getter(
            {
                "AI_TRADING_TCA_ENABLED": True,
                "AI_TRADING_TCA_ARRIVAL_BENCHMARK": "decision",
                "AI_TRADING_TCA_ALLOW_PROXY_QUOTES": True,
                "AI_TRADING_TCA_PENDING_WRITE_SEC": 60,
                "AI_TRADING_TCA_UPDATE_ON_FILL": True,
                "AI_TRADING_TCA_WRITE_PENDING_EVENTS": True,
                "AI_TRADING_TCA_PATH": "runtime/tca.jsonl",
            }
        ),
    )

    metrics, tca_record = execution_outcome.build_order_metrics_and_tca(
        symbol="AAPL",
        side="buy",
        price=100.0,
        delta_shares=4,
        now=now,
        net_target=SimpleNamespace(
            bar_ts=now - timedelta(minutes=1),
            reasons=["EDGE_RANKED"],
            proposals=[SimpleNamespace(sleeve="core")],
        ),
        order=SimpleNamespace(client_order_id="client-reject", id="broker-reject"),
        order_state=SubmittedOrderState(
            status_text="rejected",
            status_token="rejected",
            broker_order_id="broker-reject",
            filled_qty=0.0,
            requested_qty=4.0,
            fill_price=None,
            fill_timestamp=None,
            fill_fees=0.0,
            persistable_fill=False,
        ),
        submit_arrival_price=100.0,
        submit_bid_at_arrival=99.9,
        submit_ask_at_arrival=100.1,
        submit_mid_at_arrival=100.0,
        submit_quote_source="sip",
        candidate_expected_net_edge={"AAPL": 6.0},
        candidate_expected_capture={},
        get_regime_signal_profile_func=lambda: "regular",
        normalize_quote_source_token_func=lambda value: str(value).lower() if value else None,
        resolve_quote_proxy_source_func=lambda *_args, **_kwargs: None,
        resolved_tca_path_func=lambda: "resolved/path.jsonl",
        write_tca_record_func=lambda path, payload: writes.append((path, dict(payload))),
        session_bucket_from_ts_func=lambda _ts: "regular",
        compute_attribution_metrics_func=lambda **_kwargs: {"arrival": 100.0},
        safe_float=_safe_float,
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        order_lineage_metadata={
            "model_id": "ml-main",
            "model_version": "v1",
            "config_snapshot_hash": "cfg-1",
        },
    )

    assert tca_record is not None
    assert metrics["tca"]["is_bps"] is None
    assert writes[0][0] == "resolved/path.jsonl"
    terminal = writes[0][1]
    assert terminal["status"] == "rejected"
    assert terminal["pending_event"] is False
    assert terminal["pending_terminal_nonfill"] is True
    assert terminal["pending_reason"] == "rejected"
    assert terminal["model_id"] == "ml-main"
    assert terminal["config_snapshot_hash"] == "cfg-1"
