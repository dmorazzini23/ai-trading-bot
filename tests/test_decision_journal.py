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
