"""Order diagnostics must preserve stage identities and evidence gaps."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ai_trading.tools.order_funnel import build_funnel, configured_sqlite_path, read_session, session_diagnostic


def _decision(**lineage):
    return {
        "decision_uuid": "decision-1", "symbol": "AAPL",
        "decision_ts": "2026-09-30T14:00:00+00:00",  # Bar timestamp, not decision time.
        "created_at": "2026-09-30T14:06:00+00:00",
        "context_json": json.dumps({"gates": ["OK_TRADE"], "order_side": "buy", "lineage": {
            "decision_trace_id": "trace-1", "decision_ts": "2026-09-30T14:05:02+00:00",
            "account_id": "paper-1", **lineage,
        }}),
    }


def _intent(**metadata):
    return {
        "intent_id": "intent-1", "symbol": "AAPL", "side": "buy", "status": "REJECTED",
        "created_at": "2026-09-30T14:05:03+00:00", "submit_attempts": 1,
        "broker_order_id": None, "last_error": "runtime_gonogo_gate",
        "metadata_json": json.dumps({"decision_trace_id": "trace-1", "account_id": "paper-1", **metadata}),
    }


def test_claim_is_not_submission_and_rejection_has_final_blocker():
    report = build_funnel([_decision()], [_intent()], [], [])
    assert report["counts"] == {
        "durable_decisions": 1, "final_accepted_decisions": 1,
        "pre_submit_qualified_decisions": None, "durable_intents": 1,
        "intents_with_submit_claim": 1, "intents_with_durable_ack": 0,
        "intents_with_recorded_fill": 0,
    }
    row = report["intents"][0]
    assert row["decision_uuid"] == "decision-1"
    assert row["final_blocker"] == "runtime_gonogo_gate"
    assert report["decisions"][0]["recorded_at"] != report["decisions"][0]["decision_ts"]


@pytest.mark.parametrize(("lineage", "metadata", "reason"), [
    ({"decision_ts": None}, {}, "causal_timestamp_unverified"),
    ({"decision_ts": "2026-09-30T14:07:00+00:00"}, {}, "causal_timestamp_unverified"),
    ({}, {"account_id": "another-account"}, "account_conflict"),
    ({}, {"decision_trace_id": None}, "missing_trace_id"),
])
def test_no_guessed_join(lineage, metadata, reason):
    report = build_funnel([_decision(**lineage)], [_intent(**metadata)], [], [])
    assert report["intents"][0]["decision_uuid"] is None
    assert report["intents"][0]["linkage"] == reason
    assert report["status"] == "evidence_gaps"


def test_ambiguous_trace_and_missing_account_are_visible():
    other = {**_decision(), "decision_uuid": "decision-2"}
    report = build_funnel([_decision(), other], [_intent(account_id=None)], [], [])
    assert report["intents"][0]["linkage"] == "ambiguous_trace_id"
    assert "account_identity_unverified" in report["gaps"]


def test_ack_needs_matching_broker_identity_and_fills_remain_local_records():
    intent = {**_intent(), "broker_order_id": "broker-1", "status": "FILLED"}
    event = {"intent_id": "intent-1", "event_id": 1, "event_type": "SUBMIT_ACK", "broker_order_id": "wrong-id"}
    fill = {"intent_id": "intent-1", "fill_qty": 2}
    assert build_funnel([_decision()], [intent], [event], [fill])["counts"]["intents_with_durable_ack"] == 0
    event["broker_order_id"] = "broker-1"
    report = build_funnel([_decision()], [intent], [event], [fill])
    assert report["counts"]["intents_with_durable_ack"] == 1
    assert report["intents"][0]["fill_quantity"] == 2
    assert report["orders_sent"] is False


def test_duplicate_decision_uuid_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build_funnel([_decision(), _decision()], [], [], [])


def test_reader_is_read_only_and_handles_offsets_and_day_boundary(tmp_path: Path):
    database = tmp_path / "oms.db"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE decision_events (decision_uuid TEXT, symbol TEXT, created_at TEXT, context_json TEXT);
            CREATE TABLE intents (intent_id TEXT, created_at TEXT);
            CREATE TABLE oms_events (intent_id TEXT);
            CREATE TABLE intent_fills (intent_id TEXT);
        """)
        for uid, ts in (("include", "2026-09-30T23:30:00-04:00"), ("exclude", "2026-10-01T04:00:00+00:00")):
            connection.execute("INSERT INTO decision_events VALUES (?, 'AAPL', ?, '{}')", (uid, ts))
    before = database.read_bytes()
    report = read_session(database, "2026-09-30")
    assert report["counts"]["durable_decisions"] == 1
    assert report["decisions"][0]["decision_uuid"] == "include"
    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO decision_events VALUES ('unknown', 'AAPL', 'not-a-time', '{}')")
    report = read_session(database, "2026-09-30")
    assert report["status"] == "evidence_gaps"
    assert report["unassigned_capture_timestamps"] == {"decisions": 1}
    assert "rows_with_unknown_capture_session" in report["gaps"]
    with pytest.raises(FileNotFoundError):
        read_session(tmp_path / "missing.db", "2026-09-30")
    assert not (tmp_path / "missing.db").exists()


def test_configured_database_precedence_and_unavailable_read(monkeypatch, tmp_path):
    values = {"DATABASE_URL": "postgresql://redacted.invalid/oms", "AI_TRADING_OMS_INTENT_STORE_PATH": str(tmp_path / "wrong.db")}
    monkeypatch.setattr("ai_trading.tools.order_funnel.get_env", lambda name, default: values.get(name, default))
    assert configured_sqlite_path() is None
    assert session_diagnostic("2026-09-30")["gaps"] == ["configured_sqlite_oms_required"]
    values["DATABASE_URL"] = ""
    assert session_diagnostic("2026-09-30")["gaps"] == ["oms_snapshot_read_failed"]
    assert not (tmp_path / "wrong.db").exists()
