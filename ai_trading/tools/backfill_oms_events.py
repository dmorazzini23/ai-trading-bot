from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import sys
from typing import Any, cast

from ai_trading.config.management import get_env
from ai_trading.oms.event_store import EventStore
from ai_trading.oms.event_types import OmsEvent, OmsEventType
from ai_trading.oms.intent_store import FillRecord, IntentRecord, IntentStore
from ai_trading.oms.statuses import is_terminal_intent_status, normalize_intent_status


def _event_key(*parts: Any) -> str:
    material = "|".join(str(part) for part in parts if part not in (None, ""))
    if not material:
        material = "oms-backfill"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _terminal_event_type(final_status: str) -> str:
    normalized = normalize_intent_status(final_status)
    if normalized == "FILLED":
        return "ORDER_FILLED"
    if normalized in {"CANCELED", "CANCELLED", "EXPIRED", "DONE_FOR_DAY"}:
        return "ORDER_CANCELED"
    if normalized in {"FAILED", "REJECTED"}:
        return "ORDER_FAILED"
    return "INTENT_CLOSED"


def _build_intent_events(intent: IntentRecord, fills: list[FillRecord]) -> list[OmsEvent]:
    events: list[OmsEvent] = []
    source = "backfill_intent_store"

    events.append(
        OmsEvent(
            event_type="INTENT_CREATED",
            event_source=source,
            idempotency_key=_event_key(
                "INTENT_CREATED",
                intent.intent_id,
                intent.idempotency_key,
            ),
            intent_id=intent.intent_id,
            event_ts=intent.created_at,
            payload={
                "symbol": intent.symbol,
                "side": intent.side,
                "quantity": intent.quantity,
                "status": intent.status,
                "strategy_id": intent.strategy_id,
                "expected_edge_bps": intent.expected_edge_bps,
                "regime": intent.regime,
                "decision_ts": intent.decision_ts,
            },
        )
    )

    if int(intent.submit_attempts or 0) > 0:
        submit_attempts = int(intent.submit_attempts)
        events.append(
            OmsEvent(
                event_type="SUBMIT_CLAIMED",
                event_source=source,
                idempotency_key=_event_key("SUBMIT_CLAIMED", intent.intent_id, submit_attempts),
                intent_id=intent.intent_id,
                event_ts=intent.updated_at,
                payload={"submit_attempts": submit_attempts},
            )
        )
        events.append(
            OmsEvent(
                event_type="SUBMIT_ATTEMPTED",
                event_source=source,
                idempotency_key=_event_key("SUBMIT_ATTEMPTED", intent.intent_id, submit_attempts),
                intent_id=intent.intent_id,
                event_ts=intent.updated_at,
                payload={"submit_attempts": submit_attempts},
            )
        )

    broker_order_id = str(intent.broker_order_id or "").strip()
    if broker_order_id:
        events.append(
            OmsEvent(
                event_type="SUBMIT_ACK",
                event_source=source,
                idempotency_key=_event_key("SUBMIT_ACK", intent.intent_id, broker_order_id),
                intent_id=intent.intent_id,
                event_ts=intent.updated_at,
                broker_order_id=broker_order_id,
                payload={"broker_order_id": broker_order_id, "status": "SUBMITTED"},
            )
        )

    for fill in fills:
        fill_event_type = cast(
            OmsEventType,
            (
                "ORDER_FILLED"
                if normalize_intent_status(intent.status) == "FILLED"
                else "ORDER_PARTIALLY_FILLED"
            ),
        )
        events.append(
            OmsEvent(
                event_type=fill_event_type,
                event_source=source,
                idempotency_key=_event_key("FILL", intent.intent_id, fill.fill_id),
                intent_id=intent.intent_id,
                event_ts=fill.fill_ts,
                fill_id=str(fill.fill_id),
                payload={
                    "fill_id": fill.fill_id,
                    "fill_qty": fill.fill_qty,
                    "fill_price": fill.fill_price,
                    "fee": fill.fee,
                    "liquidity_flag": fill.liquidity_flag,
                    "fill_ts": fill.fill_ts,
                },
            )
        )

    if is_terminal_intent_status(intent.status):
        mapped_terminal = cast(OmsEventType, _terminal_event_type(intent.status))
        events.append(
            OmsEvent(
                event_type=mapped_terminal,
                event_source=source,
                idempotency_key=_event_key(
                    mapped_terminal,
                    intent.intent_id,
                    normalize_intent_status(intent.status),
                ),
                intent_id=intent.intent_id,
                event_ts=intent.updated_at,
                error_code=(str(intent.last_error)[:64] if intent.last_error else None),
                payload={
                    "final_status": normalize_intent_status(intent.status),
                    "last_error": intent.last_error,
                },
            )
        )
        events.append(
            OmsEvent(
                event_type="INTENT_CLOSED",
                event_source=source,
                idempotency_key=_event_key(
                    "INTENT_CLOSED",
                    intent.intent_id,
                    normalize_intent_status(intent.status),
                ),
                intent_id=intent.intent_id,
                event_ts=intent.updated_at,
                error_code=(str(intent.last_error)[:64] if intent.last_error else None),
                payload={
                    "final_status": normalize_intent_status(intent.status),
                    "last_error": intent.last_error,
                },
            )
        )

    return events


def backfill_oms_events(
    *,
    database_url: str | None = None,
    intent_store_path: str | None = None,
    limit: int = 5000,
    dry_run: bool = False,
) -> dict[str, Any]:
    store = IntentStore(path=intent_store_path, url=database_url)
    event_store = EventStore(path=intent_store_path, url=database_url)
    scanned_intents = 0
    scanned_fills = 0
    generated_events = 0
    inserted_events = 0
    by_type: Counter[str] = Counter()

    try:
        intents = store.list_intents(limit=max(1, int(limit)))
        for intent in intents:
            fills = store.list_fills(intent.intent_id)
            scanned_intents += 1
            scanned_fills += len(fills)
            built = _build_intent_events(intent, fills)
            generated_events += len(built)
            for evt in built:
                by_type[evt.event_type] += 1
            if not dry_run:
                inserted_events += event_store.append_batch(built)
    finally:
        store.close()
        event_store.close()

    duplicate_events = generated_events - inserted_events if not dry_run else 0
    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "database_url": event_store.database_url,
        "scanned_intents": int(scanned_intents),
        "scanned_fills": int(scanned_fills),
        "generated_events": int(generated_events),
        "inserted_events": int(inserted_events),
        "duplicate_events": int(max(0, duplicate_events)),
        "events_by_type": dict(sorted(by_type.items())),
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill immutable oms_events rows from existing intents/intent_fills lifecycle state."
        ),
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=str(get_env("DATABASE_URL", "", cast=str, resolve_aliases=False) or ""),
        help="Optional explicit DB URL override. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--intent-store-path",
        type=str,
        default=str(
            get_env(
                "AI_TRADING_OMS_INTENT_STORE_PATH",
                "runtime/oms_intents.db",
                cast=str,
                resolve_aliases=False,
            )
            or "runtime/oms_intents.db"
        ),
        help="SQLite path fallback when DATABASE_URL is not configured.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of intents to scan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report backfill plan without writing events.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = backfill_oms_events(
        database_url=(str(args.database_url).strip() or None),
        intent_store_path=(str(args.intent_store_path).strip() or None),
        limit=max(1, int(args.limit)),
        dry_run=bool(args.dry_run),
    )
    sys.stdout.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
