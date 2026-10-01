"""Read-only, identity-preserving diagnostic of a paper OMS session.

Counts are separate cohorts, not conversion rates. Local submit claims and
durable acknowledgements are not proof of broker activity or profitability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import UTC, date, datetime, time, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else None


def _object(value: Any) -> dict[str, Any]:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("OMS metadata must be a JSON object")
    return parsed


def build_funnel(
    decisions: list[dict[str, Any]], intents: list[dict[str, Any]],
    events: list[dict[str, Any]], fills: list[dict[str, Any]],
) -> dict[str, Any]:
    """Join only unique explicit trace identities with causal decision times."""
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    blockers: Counter[str] = Counter()
    candidate_rows = []
    ids = [row["decision_uuid"] for row in decisions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate durable decision UUID")
    for row in decisions:
        context = _object(row.get("context_json"))
        lineage = context.get("lineage", {})
        if not isinstance(lineage, dict):
            raise ValueError("decision lineage must be an object")
        gates = context.get("gates", [])
        if not isinstance(gates, list):
            raise ValueError("decision gates must be a list")
        accepted = "OK_TRADE" in gates
        # Keep all reported reasons; do not guess precedence among multiple gates.
        reasons = [str(gate) for gate in gates if gate != "OK_TRADE"]
        if not accepted:
            blockers.update(set(reasons or ["acceptance_evidence_missing"]))
        candidate = {
            "decision_uuid": row["decision_uuid"], "symbol": row["symbol"],
            "side": context.get("order_side"),
            "account_id": lineage.get("account_id"),
            "decision_trace_id": lineage.get("decision_trace_id"),
            # The decision table's timestamp can be a bar start. Never substitute it.
            "decision_ts": lineage.get("decision_ts"),
            "source_timestamp": lineage.get("source_timestamp"),
            "recorded_at": row.get("created_at"), "final_accepted": accepted,
            "reported_gate_reasons": reasons,
            "final_blocker": context.get("terminal_reason"),
        }
        candidate_rows.append(candidate)
        if candidate["decision_trace_id"]:
            by_trace[str(candidate["decision_trace_id"])].append(candidate)

    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fill_by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_intent[str(event.get("intent_id"))].append(event)
    for fill in fills:
        fill_by_intent[str(fill.get("intent_id"))].append(fill)
    intent_rows = []
    linkage: Counter[str] = Counter()
    for intent in intents:
        metadata = _object(intent.get("metadata_json"))
        trace = metadata.get("decision_trace_id")
        candidates = by_trace.get(str(trace), []) if trace else []
        reason = "missing_trace_id" if not trace else "decision_not_recorded"
        matched = None
        if len(candidates) > 1:
            reason = "ambiguous_trace_id"
        elif len(candidates) == 1:
            candidate = candidates[0]
            decision_ts = _timestamp(candidate["decision_ts"])
            created_at = _timestamp(intent.get("created_at"))
            if candidate["symbol"] != intent["symbol"] or candidate["side"] not in (None, intent["side"]):
                reason = "symbol_or_side_conflict"
            elif candidate["account_id"] and metadata.get("account_id") and candidate["account_id"] != metadata["account_id"]:
                reason = "account_conflict"
            elif decision_ts is None or created_at is None or decision_ts > created_at:
                reason = "causal_timestamp_unverified"
            else:
                matched, reason = candidate["decision_uuid"], "linked_local_trace"
        linkage[reason] += 1
        timeline = sorted(by_intent[str(intent["intent_id"])], key=lambda row: row.get("event_id", 0))
        acknowledgements = [event for event in timeline if event.get("event_type") == "SUBMIT_ACK" and event.get("broker_order_id") == intent.get("broker_order_id") and intent.get("broker_order_id")]
        final_rejections = [event for event in timeline if event.get("event_type") == "SUBMIT_REJECT"]
        intent_rows.append({
            "intent_id": intent["intent_id"], "symbol": intent["symbol"],
            "side": intent["side"], "status": intent["status"],
            "decision_trace_id": trace, "decision_uuid": matched,
            "linkage": reason, "account_id": metadata.get("account_id"),
            "account_identity_consistent": bool(metadata.get("account_id") and candidates and candidates[0]["account_id"] == metadata["account_id"] and matched),
            "created_at": intent.get("created_at"), "decision_ts": intent.get("decision_ts"),
            "broker_order_id": intent.get("broker_order_id"),
            "submit_claims": intent.get("submit_attempts", 0),
            "durable_acknowledgement": bool(acknowledgements),
            "fill_records": len(fill_by_intent[str(intent["intent_id"])]),
            "fill_quantity": sum(float(fill["fill_qty"]) for fill in fill_by_intent[str(intent["intent_id"])]),
            "final_blocker": intent.get("last_error") or (final_rejections[-1].get("error_code") if final_rejections else None),
            "events": [{key: event.get(key) for key in ("event_uuid", "event_type", "event_ts", "created_at", "error_code", "broker_order_id", "fill_id")} for event in timeline],
        })
    gaps = []
    if any(row["linkage"] != "linked_local_trace" for row in intent_rows):
        gaps.append("decision_intent_linkage_incomplete")
    if any(not row["account_identity_consistent"] for row in intent_rows):
        gaps.append("account_identity_unverified")
    return {
        "status": "evidence_gaps" if gaps else "local_diagnostic_complete",
        "gaps": gaps,
        "counts": {
            "durable_decisions": len(candidate_rows),
            "final_accepted_decisions": sum(row["final_accepted"] for row in candidate_rows),
            "pre_submit_qualified_decisions": None,
            "durable_intents": len(intent_rows),
            "intents_with_submit_claim": sum(row["submit_claims"] > 0 for row in intent_rows),
            "intents_with_durable_ack": sum(row["durable_acknowledgement"] for row in intent_rows),
            "intents_with_recorded_fill": sum(row["fill_records"] > 0 for row in intent_rows),
        },
        "linkage_counts": dict(linkage), "decision_gate_reasons": dict(blockers),
        "decisions": candidate_rows, "intents": intent_rows,
        "count_semantics": "distinct stage cohorts; blocker counts overlap; submit claims are not broker requests; matching local account identities and acknowledgements are not an independent broker audit",
        "qualification_evidence": "pre-submit qualification is not independently recorded; OK_TRADE is final acceptance; reported gate reasons can include nonblocking scales",
        "promotion_authority": False, "orders_sent": False,
    }


def read_session(database: Path, session: str) -> dict[str, Any]:
    """Use a coherent read transaction; never initialize or modify an OMS."""
    day = date.fromisoformat(session)
    start = datetime.combine(day, time.min, ZoneInfo("America/New_York")).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), time.min, ZoneInfo("America/New_York")).astimezone(UTC)
    if not database.is_file():
        raise FileNotFoundError(database)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        # Filter parsed capture times, rather than assuming every offset sorts lexically.
        unassigned: Counter[str] = Counter()
        def captured_rows(query: str, label: str) -> list[dict[str, Any]]:
            rows = []
            for row in connection.execute(query):
                ts = _timestamp(row["created_at"])
                if ts is None:
                    unassigned[label] += 1
                elif start <= ts < end:
                    rows.append(dict(row))
            return rows
        decisions = captured_rows("SELECT * FROM decision_events", "decisions")
        intents = captured_rows("SELECT * FROM intents", "intents")
        events: list[dict[str, Any]] = []
        fills: list[dict[str, Any]] = []
        for intent in intents:
            events.extend(dict(row) for row in connection.execute("SELECT * FROM oms_events WHERE intent_id = ?", (intent["intent_id"],)))
            fills.extend(dict(row) for row in connection.execute("SELECT * FROM intent_fills WHERE intent_id = ?", (intent["intent_id"],)))
        report = build_funnel(decisions, intents, events, fills)
        report["unassigned_capture_timestamps"] = dict(unassigned)
        if unassigned:
            report["gaps"].append("rows_with_unknown_capture_session")
            report["status"] = "evidence_gaps"
    report.update({"session_date": session, "generated_at": datetime.now(UTC).isoformat(), "database": str(database.resolve()), "cohort": "decisions and intents captured during NY calendar day; all lifecycle events/fills for those intents as of snapshot"})
    return report


def configured_sqlite_path() -> Path | None:
    """Honor DATABASE_URL precedence without initializing storage or exposing it."""
    url = str(get_env("DATABASE_URL", "") or "").strip()
    if url:
        return Path(url[len("sqlite:///"):]).expanduser() if url.startswith("sqlite:///") and "?" not in url else None
    path = str(get_env("AI_TRADING_OMS_INTENT_STORE_PATH", "") or "").strip()
    return Path(path).expanduser() if path and "://" not in path else None


def session_diagnostic(session: str, database: Path | None = None) -> dict[str, Any]:
    path = database if database is not None else configured_sqlite_path()
    if path is None:
        return {"status": "unavailable", "gaps": ["configured_sqlite_oms_required"], "orders_sent": False}
    try:
        return read_session(path, session)
    except (OSError, ValueError, sqlite3.Error) as exc:
        return {"status": "unavailable", "gaps": ["oms_snapshot_read_failed"], "error_type": type(exc).__name__, "orders_sent": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = read_session(args.database, args.session)
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    get_logger(__name__).info("ORDER_FUNNEL_COMPLETE", extra={"status": report["status"], "counts": report["counts"]})


if __name__ == "__main__":
    main()
