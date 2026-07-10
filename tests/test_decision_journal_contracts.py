from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.core.netting import (
    DecisionRecord,
    NettedTarget,
    SleeveProposal,
    build_decision_record,
)
from ai_trading.contracts import Bar, BrokerOrderSnapshot, ExecutionResult, PositionSnapshot, Quote
from ai_trading.oms.pretrade import OrderIntent as PretradeOrderIntent
from ai_trading.strategies.base import StrategySignal


def test_strategy_signal_to_contract_exposes_canonical_fields() -> None:
    signal = StrategySignal(
        "AAPL",
        "BUY",
        strength=0.7,
        confidence=0.8,
        strategy_id="mean_revert_v2",
        timeframe="1m",
        signal_type="mean_reversion",
        metadata={"source": "unit_test"},
    )

    payload = signal.to_contract().to_dict()

    assert payload["symbol"] == "AAPL"
    assert payload["side"] == "buy"
    assert payload["strength"] == 0.7
    assert payload["confidence"] == 0.8
    assert payload["strategy_id"] == "mean_revert_v2"
    assert payload["signal_type"] == "mean_reversion"
    assert payload["metadata"]["source"] == "unit_test"


def test_pretrade_order_intent_to_contract_exposes_canonical_fields() -> None:
    bar_ts = datetime.now(UTC)
    intent = PretradeOrderIntent(
        symbol="MSFT",
        side="buy",
        qty=12,
        notional=3600.0,
        limit_price=300.0,
        bar_ts=bar_ts,
        client_order_id="coid-msft-1",
        last_price=301.0,
        spread=0.02,
        quote_quality_ok=True,
    )

    payload = intent.to_contract().to_dict()

    assert payload["symbol"] == "MSFT"
    assert payload["side"] == "buy"
    assert payload["qty"] == 12.0
    assert payload["notional"] == 3600.0
    assert payload["limit_price"] == 300.0
    assert payload["client_order_id"] == "coid-msft-1"
    assert payload["metadata"]["quote_quality_ok"] is True


def test_pretrade_order_intent_to_contract_preserves_sell_short_side() -> None:
    bar_ts = datetime.now(UTC)
    intent = PretradeOrderIntent(
        symbol="TSLA",
        side="sell_short",
        qty=4,
        notional=1000.0,
        limit_price=250.0,
        bar_ts=bar_ts,
        client_order_id="coid-tsla-short-1",
    )

    payload = intent.to_contract().to_dict()

    assert payload["side"] == "sell_short"
    assert payload["qty"] == 4.0


def test_decision_record_includes_canonical_decision_journal() -> None:
    bar_ts = datetime.now(UTC)
    proposal = SleeveProposal(
        symbol="AAPL",
        sleeve="day",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        expected_edge_bps=18.0,
        expected_cost_bps=4.0,
        score=0.65,
        confidence=0.84,
    )
    net_target = NettedTarget(
        symbol="AAPL",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        target_shares=8.0,
        reasons=["EDGE_OK"],
        proposals=[proposal],
    )
    record = DecisionRecord(
        symbol="AAPL",
        bar_ts=bar_ts,
        sleeves=[proposal],
        net_target=net_target,
        gates=["OK_TRADE"],
        order={
            "client_order_id": "coid-aapl-1",
            "decision_trace_id": "trace-aapl-1",
            "side": "buy",
            "qty": 8,
            "price": 125.5,
            "strategy_id": "mean_revert_v2",
            "status": "accepted",
        },
        metrics={
            "expected_net_edge_bps": 14.2,
            "model_id": "ml-main",
            "model_version": "v2026.04.15",
        },
        config_snapshot={
            "config_snapshot_hash": "cfg-hash-1",
            "effective_policy_hash": "policy-hash-1",
            "dataset_hash": "ds-hash-1",
            "feature_version": "fv-2026.04",
            "model_artifact_hash": "artifact-hash-1",
        },
    )

    payload = record.to_dict()
    journal = payload["decision_journal"]

    assert journal["schema_version"] == "1.0.0"
    assert journal["event"] == "decision_record"
    assert journal["symbol"] == "AAPL"
    assert journal["data_freshness_sec"] is not None
    assert journal["accepted"] is True
    assert journal["submitted"] is True
    assert journal["client_order_id"] == "coid-aapl-1"
    assert journal["decision_trace_id"] == "trace-aapl-1"
    assert journal["config_snapshot_hash"] == "cfg-hash-1"
    assert journal["policy_hash"] == "policy-hash-1"
    assert journal["model_id"] == "ml-main"
    assert journal["model_version"] == "v2026.04.15"
    assert journal["dataset_hash"] == "ds-hash-1"
    assert journal["target_delta_shares"] == 8.0
    assert "OK_TRADE" in journal["reasons"]
    assert journal["signal"]["side"] == "buy"
    assert journal["signal"]["strategy_id"] == "day"
    assert journal["risk_decision"]["accepted"] is True
    assert journal["risk_decision"]["expected_net_edge_bps"] == 14.2
    assert journal["order_intent"]["client_order_id"] == "coid-aapl-1"
    assert journal["order_intent"]["strategy_id"] == "mean_revert_v2"
    assert journal["broker_result"]["submitted"] is True
    assert journal["broker_result"]["accepted"] is True


def test_decision_journal_uses_consistent_lineage_source_precedence() -> None:
    bar_ts = datetime.now(UTC)
    record = build_decision_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=NettedTarget(
            symbol="AAPL",
            bar_ts=bar_ts,
            target_dollars=1000.0,
            target_shares=8.0,
        ),
        gates=["OK_TRADE"],
        metrics={
            "model_id": "metrics-model-id",
            "model_version": "",
        },
        order={
            "side": "buy",
            "qty": 8,
            "status": "accepted",
            "model_id": "order-model-id",
            "model_version": "order-model-version",
            "dataset_hash": "",
        },
        config_snapshot={
            "model_version": "config-model-version",
            "dataset_hash": "config-dataset-hash",
            "feature_version": "",
        },
        tca={
            "dataset_hash": "tca-dataset-hash",
            "feature_version": "tca-feature-version",
            "model_artifact_hash": "tca-artifact-hash",
        },
    )

    journal = record.to_dict()["decision_journal"]
    assert journal["model_id"] == "metrics-model-id"
    assert journal["model_version"] == "order-model-version"
    assert journal["dataset_hash"] == "config-dataset-hash"
    assert journal["feature_version"] == "tca-feature-version"
    assert journal["model_artifact_hash"] == "tca-artifact-hash"


def test_build_decision_record_populates_explicit_canonical_contracts() -> None:
    bar_ts = datetime.now(UTC)
    proposal = SleeveProposal(
        symbol="MSFT",
        sleeve="swing",
        bar_ts=bar_ts,
        target_dollars=-2500.0,
        expected_edge_bps=22.0,
        expected_cost_bps=6.0,
        score=-0.72,
        confidence=0.91,
    )
    net_target = NettedTarget(
        symbol="MSFT",
        bar_ts=bar_ts,
        target_dollars=-2500.0,
        target_shares=-10.0,
        reasons=["REVERSAL_SIGNAL"],
        proposals=[proposal],
    )

    record = build_decision_record(
        symbol="MSFT",
        bar_ts=bar_ts,
        net_target=net_target,
        gates=["PRE_SUBMIT_CHECKS_FAILED"],
        metrics={"expected_net_edge_bps": 9.5},
        config_snapshot={
            "liquidity_regime": "NORMAL",
            "config_snapshot_hash": "cfg-msft",
            "effective_policy_hash": "policy-msft",
        },
        order={
            "client_order_id": "coid-msft-2",
            "decision_trace_id": "trace-msft-2",
            "side": "sell",
            "qty": 10,
            "price": 250.0,
        },
    )

    assert record.signal is not None
    assert record.signal.symbol == "MSFT"
    assert record.signal.side == "sell"
    assert record.signal.strategy_id == "swing"
    assert record.risk_decision is not None
    assert record.risk_decision.accepted is False
    assert record.risk_decision.veto_gate == "PRE_SUBMIT_CHECKS_FAILED"
    assert record.order_intent is not None
    assert record.order_intent.client_order_id == "coid-msft-2"
    assert record.decision_trace_id == "trace-msft-2"


def test_decision_journal_exposes_provider_feed_and_broker_result() -> None:
    bar_ts = datetime.now(UTC)
    proposal = SleeveProposal(
        symbol="NVDA",
        sleeve="intraday",
        bar_ts=bar_ts,
        target_dollars=-900.0,
        expected_edge_bps=11.0,
        expected_cost_bps=3.0,
        score=-0.51,
        confidence=0.88,
    )
    net_target = NettedTarget(
        symbol="NVDA",
        bar_ts=bar_ts,
        target_dollars=-900.0,
        target_shares=-3.0,
        proposals=[proposal],
    )
    record = build_decision_record(
        symbol="NVDA",
        bar_ts=bar_ts,
        net_target=net_target,
        gates=["BROKER_REJECTED"],
        order={
            "client_order_id": "coid-nvda-1",
            "side": "sell",
            "qty": 3,
            "price": 300.0,
            "status": "rejected",
        },
        tca={
            "provider": "alpaca",
            "quote_proxy_source": "iex",
            "venue": "NASDAQ",
            "is_bps": 5.5,
        },
    )

    journal = record.to_dict()["decision_journal"]

    assert journal["provider"] == "alpaca"
    assert journal["feed"] == "iex"
    assert journal["event"] == "decision_record"
    assert journal["data_freshness_sec"] is not None
    assert journal["client_order_id"] == "coid-nvda-1"
    assert journal["target_delta_shares"] == -3.0
    assert "BROKER_REJECTED" in journal["reasons"]
    assert journal["broker_result"]["status"] == "rejected"
    assert journal["broker_result"]["error_reason"] == "BROKER_REJECTED"


def test_decision_journal_preserves_sell_short_intent_with_negative_delta() -> None:
    bar_ts = datetime.now(UTC)
    proposal = SleeveProposal(
        symbol="TSLA",
        sleeve="intraday",
        bar_ts=bar_ts,
        target_dollars=-1000.0,
        expected_edge_bps=12.0,
        expected_cost_bps=3.0,
        score=-0.6,
        confidence=0.9,
    )
    record = build_decision_record(
        symbol="TSLA",
        bar_ts=bar_ts,
        net_target=NettedTarget(
            symbol="TSLA",
            bar_ts=bar_ts,
            target_dollars=-1000.0,
            target_shares=-4.0,
            proposals=[proposal],
        ),
        gates=["OK_TRADE"],
        order={
            "client_order_id": "coid-tsla-short-2",
            "side": "sell_short",
            "qty": 4,
            "price": 250.0,
            "status": "accepted",
        },
    )

    journal = record.to_dict()["decision_journal"]

    assert journal["order_intent"]["side"] == "sell_short"
    assert journal["target_delta_shares"] == -4.0
    assert journal["broker_result"]["broker_order"]["side"] == "sell_short"


def test_decision_journal_preserves_explicit_zero_filled_qty() -> None:
    bar_ts = datetime.now(UTC)
    record = build_decision_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=NettedTarget(
            symbol="AAPL",
            bar_ts=bar_ts,
            target_dollars=1000.0,
            target_shares=8.0,
        ),
        gates=["OK_TRADE"],
        order={
            "client_order_id": "coid-aapl-zero-fill",
            "side": "buy",
            "qty": 8,
            "price": 125.5,
            "status": "accepted",
            "filled_qty": 0,
        },
        tca={
            "total_qty": 8,
            "provider": "alpaca",
        },
    )

    journal = record.to_dict()["decision_journal"]

    assert journal["broker_result"]["filled_qty"] == 0.0
    assert journal["broker_result"]["broker_order"]["filled_qty"] == 0.0


def test_decision_journal_submitted_requires_submission_evidence() -> None:
    bar_ts = datetime.now(UTC)
    record = build_decision_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=NettedTarget(
            symbol="AAPL",
            bar_ts=bar_ts,
            target_dollars=1000.0,
            target_shares=8.0,
        ),
        gates=["OK_TRADE"],
        order={
            "client_order_id": "coid-aapl-intent-only",
            "side": "buy",
            "qty": 8,
            "price": 125.5,
        },
    )

    journal = record.to_dict()["decision_journal"]

    assert journal["order_intent"] is not None
    assert journal["submitted"] is False
    assert journal["broker_result"]["submitted"] is False


def test_decision_journal_uses_explicit_event_and_freshness_metrics() -> None:
    bar_ts = datetime.now(UTC)
    record = build_decision_record(
        symbol="TSLA",
        bar_ts=bar_ts,
        net_target=NettedTarget(
            symbol="TSLA",
            bar_ts=bar_ts,
            target_dollars=0.0,
            target_shares=0.0,
        ),
        gates=["LOW_OR_NO_SIGNAL"],
        metrics={
            "event": "low_signal_hold",
            "data_freshness_sec": 17.5,
        },
    )

    journal = record.to_dict()["decision_journal"]

    assert journal["event"] == "low_signal_hold"
    assert journal["data_freshness_sec"] == 17.5


def test_market_and_execution_contracts_serialize() -> None:
    ts = datetime.now(UTC)
    bar = Bar.from_mapping(
        {
            "symbol": "AAPL",
            "ts": ts.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000,
            "provider": "alpaca",
            "feed": "iex",
        }
    )
    quote = Quote(
        symbol="AAPL",
        ts=ts,
        bid=100.0,
        ask=100.1,
        mid=100.05,
        last=100.08,
        provider="alpaca",
        feed="iex",
    )
    broker_order = BrokerOrderSnapshot(
        client_order_id="coid-aapl-9",
        broker_order_id="boid-1",
        side="buy",
        qty=5.0,
        filled_qty=5.0,
        limit_price=100.0,
        fill_price=100.02,
        status="filled",
        venue="NASDAQ",
        ts=ts,
    )
    result = ExecutionResult(
        submitted=True,
        accepted=True,
        status="filled",
        provider="alpaca",
        venue="NASDAQ",
        broker_order=broker_order,
        fill_count=1,
        filled_qty=5.0,
        realized_slippage_bps=2.0,
        fees=0.1,
    )
    position = PositionSnapshot(
        symbol="AAPL",
        qty=5.0,
        market_value=500.0,
        avg_entry_price=100.0,
        provider="alpaca",
        ts=ts,
    )

    assert bar.to_dict()["provider"] == "alpaca"
    assert quote.to_dict()["feed"] == "iex"
    assert result.to_dict()["broker_order"]["venue"] == "NASDAQ"
    assert position.to_dict()["qty"] == 5.0
