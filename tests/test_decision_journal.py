from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.contracts import Bar
from ai_trading.core.decision_journal import DecisionJournalRecorder


def test_decision_journal_recorder_emits_canonical_journal() -> None:
    captured: dict[str, object] = {}

    def _write(record, path):
        captured["path"] = path
        captured["payload"] = record.to_dict()

    recorder = DecisionJournalRecorder(
        path="runtime/decision_records.jsonl",
        write_impl=_write,
    )
    ts = datetime(2026, 4, 19, 15, 30, tzinfo=UTC)

    recorder.record(
        symbol="AAPL",
        bar_ts=ts,
        signal_side="buy",
        final_score=0.72,
        confidence=0.88,
        strategy_id="momentum",
        accepted=True,
        gates=["OK_TRADE"],
        reasons=["EDGE_OK"],
        target_delta_shares=5.0,
        submitted=True,
        client_order_id="coid-aapl-1",
        broker_order_id="oid-aapl-1",
        broker_status="accepted",
        provider="alpaca",
        feed="iex",
        reference_price=100.0,
        event="order_submitted",
        data_freshness_sec=12.5,
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    journal = payload["decision_journal"]
    assert journal["event"] == "order_submitted"
    assert journal["provider"] == "alpaca"
    assert journal["feed"] == "iex"
    assert journal["data_freshness_sec"] == 12.5
    assert journal["accepted"] is True
    assert journal["submitted"] is True
    assert journal["target_delta_shares"] == 5.0
    assert journal["client_order_id"] == "coid-aapl-1"
    assert journal["broker_result"]["accepted"] is True


def test_decision_journal_recorder_emits_block_without_order() -> None:
    captured: dict[str, object] = {}

    recorder = DecisionJournalRecorder(
        path=None,
        write_impl=lambda record, path: captured.setdefault("payload", record.to_dict()),
    )
    ts = datetime(2026, 4, 19, 15, 45, tzinfo=UTC)

    recorder.record(
        symbol="MSFT",
        bar_ts=ts,
        signal_side="hold",
        final_score=0.0,
        confidence=0.0,
        strategy_id=None,
        accepted=False,
        gates=["LOW_OR_NO_SIGNAL"],
        reasons=["LOW_OR_NO_SIGNAL"],
        event="low_signal_hold",
        data_freshness_sec=3.0,
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    journal = payload["decision_journal"]
    assert journal["event"] == "low_signal_hold"
    assert journal["accepted"] is False
    assert journal["submitted"] is False
    assert journal["order_intent"] is None
    assert journal["broker_result"] is None
    assert journal["reasons"] == ["LOW_OR_NO_SIGNAL"]


def test_decision_journal_uses_market_bar_contract_defaults() -> None:
    captured: dict[str, object] = {}
    ts = datetime(2026, 4, 19, 15, 50, tzinfo=UTC)
    market_bar = Bar(
        symbol="QQQ",
        ts=ts,
        open=499.5,
        high=500.2,
        low=498.9,
        close=500.0,
        volume=2500.0,
        timeframe="1Min",
        provider="alpaca",
        feed="sip",
    )

    recorder = DecisionJournalRecorder(
        path=None,
        write_impl=lambda record, path: captured.setdefault("payload", record.to_dict()),
    )

    recorder.record(
        symbol="QQQ",
        market_bar=market_bar,
        bar_ts=None,
        signal_side="buy",
        final_score=0.5,
        confidence=0.7,
        strategy_id="non_netting",
        accepted=False,
        gates=["ENTRY_BLOCKED_FEED_RELIABILITY"],
        reasons=["ENTRY_BLOCKED_FEED_RELIABILITY"],
        event="feed_reliability_block",
        data_freshness_sec=4.0,
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    journal = payload["decision_journal"]
    assert journal["provider"] == "alpaca"
    assert journal["feed"] == "sip"
    assert journal["bar_ts"] == ts.isoformat()
    assert journal["metadata"]["market_bar"]["close"] == 500.0


def test_decision_journal_correlation_is_pre_submit_and_order_independent() -> None:
    payloads: list[dict[str, object]] = []
    recorder = DecisionJournalRecorder(
        path=None,
        write_impl=lambda record, _path: payloads.append(record.to_dict()),
    )
    ts = datetime(2026, 7, 21, 14, 31, tzinfo=UTC)

    recorder.record(
        symbol="AAPL",
        bar_ts=ts,
        signal_side="buy",
        final_score=0.6,
        confidence=0.8,
        strategy_id="day",
        accepted=False,
        gates=["METRICS_IMPROVEMENT_CONTROLLED_SKIP"],
        event="metrics_improvement_controlled_skip",
        reference_price=100.0,
    )
    recorder.record(
        symbol="AAPL",
        bar_ts=ts,
        signal_side="buy",
        final_score=0.6,
        confidence=0.8,
        strategy_id="day",
        accepted=True,
        gates=["OK_TRADE"],
        submitted=True,
        client_order_id="replacement-child-order",
        reference_price=100.0,
    )

    blocked = payloads[0]
    submitted = payloads[1]
    correlation_id = blocked["correlation_id"]
    assert isinstance(correlation_id, str)
    assert correlation_id.startswith("opp_")
    assert submitted["correlation_id"] == correlation_id
    assert blocked["decision_journal"]["correlation_id"] == correlation_id
    assert submitted["decision_journal"]["order_intent"]["correlation_id"] == correlation_id
    assert submitted["decision_journal"]["client_order_id"] == "replacement-child-order"


def test_decision_journal_preserves_opportunity_snapshot_metadata() -> None:
    captured: dict[str, object] = {}
    recorder = DecisionJournalRecorder(
        path=None,
        write_impl=lambda record, _path: captured.setdefault("payload", record.to_dict()),
    )
    source_ts = datetime(2026, 7, 21, 15, 0, tzinfo=UTC)
    decision_ts = datetime(2026, 7, 21, 15, 0, 1, tzinfo=UTC)
    quote_ts = datetime(2026, 7, 21, 15, 0, 0, 750000, tzinfo=UTC)

    recorder.record(
        symbol="MSFT",
        bar_ts=source_ts,
        signal_side="sell",
        final_score=-0.7,
        confidence=0.9,
        strategy_id="intraday",
        accepted=False,
        gates=["NET_EDGE_FLOOR_GATE"],
        decision_ts=decision_ts,
        source_timestamp=source_ts,
        quote_timestamp=quote_ts,
        quote_age_ms=250.0,
        spread_bps=3.5,
        order_type="limit",
        session="midday",
        market_regime="sideways",
        volatility_regime="normal",
        trend_regime="flat",
        execution_profile="passive",
        reference_price=410.0,
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    journal = payload["decision_journal"]
    assert journal["decision_ts"] == decision_ts.isoformat()
    assert journal["source_timestamp"] == source_ts.isoformat()
    assert journal["quote_timestamp"] == quote_ts.isoformat()
    metadata = journal["metadata"]
    assert metadata["opportunity_eligible"] is True
    assert metadata["reference_price"] == 410.0
    assert metadata["quote_age_ms"] == 250.0
    assert metadata["spread_bps"] == 3.5
    assert metadata["order_type"] == "limit"
    assert metadata["session_regime"] == "midday"
    assert metadata["market_regime"] == "sideways"
    assert metadata["execution_profile"] == "passive"
