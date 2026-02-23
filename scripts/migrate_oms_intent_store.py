#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.oms.intent_store import IntentStore
from ai_trading.oms.statuses import TERMINAL_INTENT_STATUSES, normalize_intent_status


_TERMINAL_STATUSES: frozenset[str] = TERMINAL_INTENT_STATUSES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate OMS intent/fill state from legacy sqlite intent store to "
            "DATABASE_URL-backed SQLAlchemy store."
        )
    )
    parser.add_argument(
        "--source-sqlite",
        type=Path,
        default=Path(str(get_env("AI_TRADING_OMS_INTENT_STORE_PATH", "runtime/oms_intents.db"))),
        help="Path to source sqlite DB file (legacy intent store).",
    )
    parser.add_argument(
        "--target-url",
        type=str,
        default=str(get_env("DATABASE_URL", "") or "").strip(),
        help="Target SQLAlchemy DB URL (defaults to DATABASE_URL).",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only inspect source data and print migration plan.",
    )
    return parser


def _safe_json_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_source_rows(source_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not source_path.exists():
        raise FileNotFoundError(f"Source sqlite DB not found: {source_path}")
    conn = sqlite3.connect(str(source_path))
    conn.row_factory = sqlite3.Row
    try:
        intents = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM intents ORDER BY created_at ASC, intent_id ASC"
            ).fetchall()
        ]
        fills = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM intent_fills ORDER BY fill_id ASC"
            ).fetchall()
        ]
    finally:
        conn.close()
    return intents, fills


def _migrate(
    *,
    source_sqlite: Path,
    target_url: str,
    dry_run: bool,
) -> dict[str, Any]:
    intents, fills = _load_source_rows(source_sqlite)
    fills_by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fill in fills:
        fills_by_intent[str(fill.get("intent_id", ""))].append(fill)

    summary: dict[str, Any] = {
        "source_sqlite": str(source_sqlite),
        "target_url_configured": bool(target_url),
        "intents_in_source": len(intents),
        "fills_in_source": len(fills),
        "dry_run": bool(dry_run),
        "migrated_intents": 0,
        "migrated_fills": 0,
        "deduped_intents": 0,
    }
    if dry_run:
        return summary
    if not target_url:
        raise RuntimeError(
            "Missing target DB URL. Set DATABASE_URL or pass --target-url."
        )

    target = IntentStore(url=target_url)
    old_to_new_intent_id: dict[str, str] = {}
    for row in intents:
        old_intent_id = str(row.get("intent_id", ""))
        status = normalize_intent_status(
            str(row.get("status", "PENDING_SUBMIT") or "PENDING_SUBMIT"),
            default="PENDING_SUBMIT",
        )
        record, created = target.create_intent(
            intent_id=old_intent_id,
            idempotency_key=str(row.get("idempotency_key", old_intent_id)),
            symbol=str(row.get("symbol", "")).upper(),
            side=str(row.get("side", "buy")).lower(),
            quantity=float(row.get("quantity", 0.0) or 0.0),
            decision_ts=str(row.get("decision_ts") or ""),
            strategy_id=(str(row["strategy_id"]) if row.get("strategy_id") else None),
            expected_edge_bps=(
                float(row["expected_edge_bps"])
                if row.get("expected_edge_bps") is not None
                else None
            ),
            regime=(str(row["regime"]) if row.get("regime") else None),
            metadata=_safe_json_dict(row.get("metadata_json")),
            status=status,
        )
        old_to_new_intent_id[old_intent_id] = record.intent_id
        if created:
            summary["migrated_intents"] += 1
        else:
            summary["deduped_intents"] += 1

        broker_order_id = row.get("broker_order_id")
        if broker_order_id:
            target.mark_submitted(record.intent_id, str(broker_order_id))

        for fill in fills_by_intent.get(old_intent_id, ()):
            target.record_fill(
                record.intent_id,
                fill_qty=float(fill.get("fill_qty", 0.0) or 0.0),
                fill_price=(
                    float(fill["fill_price"])
                    if fill.get("fill_price") is not None
                    else None
                ),
                fee=float(fill.get("fee", 0.0) or 0.0),
                liquidity_flag=(
                    str(fill["liquidity_flag"]) if fill.get("liquidity_flag") else None
                ),
                fill_ts=str(fill.get("fill_ts") or ""),
            )
            summary["migrated_fills"] += 1

        last_error = str(row.get("last_error") or "").strip() or None
        if status in _TERMINAL_STATUSES:
            target.close_intent(
                record.intent_id,
                final_status=status,
                last_error=last_error,
            )
        elif status == "PENDING_SUBMIT" and last_error:
            target.record_submit_error(record.intent_id, last_error)

    summary["intent_id_map_size"] = len(old_to_new_intent_id)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = _migrate(
        source_sqlite=Path(args.source_sqlite),
        target_url=str(args.target_url or "").strip(),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
