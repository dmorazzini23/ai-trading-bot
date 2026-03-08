from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import uuid

from ai_trading.config.management import (
    clear_runtime_env_override,
    get_env,
    set_runtime_env_override,
)
from ai_trading.logging import get_logger
from ai_trading.oms.intent_store import IntentStore

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a live cutover drill for startup validation and OMS durability checks.",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="live",
        help="Mode used for startup validation check (live or paper).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="AAPL",
        help="Symbol used for synthetic drill intent.",
    )
    parser.add_argument(
        "--qty",
        type=float,
        default=1.0,
        help="Quantity for synthetic drill intent.",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=100.0,
        help="Synthetic fill price for drill fill event.",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="cutover-drill",
        help="Prefix used for drill intent/idempotency identifiers.",
    )
    parser.add_argument(
        "--intent-store-path",
        type=str,
        default="",
        help="Optional explicit sqlite path override for intent store.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="",
        help="Optional explicit DATABASE_URL override for intent store.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser


def _normalize_database_url(raw: str) -> str:
    value = str(raw or "").strip()
    if value.startswith("postgres://"):
        return f"postgresql+psycopg://{value[len('postgres://') :]}"
    if value.startswith("postgresql://") and "+" not in value.split("://", 1)[0]:
        return f"postgresql+psycopg://{value[len('postgresql://') :]}"
    return value


def _run_startup_validation(execution_mode: str) -> tuple[bool, str | None]:
    from ai_trading.__main__ import _validate_startup_config

    prior_mode = get_env("EXECUTION_MODE", None, cast=str, resolve_aliases=False)
    set_runtime_env_override("EXECUTION_MODE", execution_mode)
    try:
        _validate_startup_config()
        return True, None
    except SystemExit as exc:
        return False, str(exc)
    finally:
        if prior_mode in (None, ""):
            clear_runtime_env_override("EXECUTION_MODE")
        else:
            set_runtime_env_override("EXECUTION_MODE", prior_mode)


def _run_oms_durability_drill(
    *,
    symbol: str,
    qty: float,
    price: float,
    id_prefix: str,
    intent_store_path: str,
    database_url: str,
) -> dict[str, Any]:
    now_iso = datetime.now(UTC).isoformat()
    suffix = uuid.uuid4().hex[:12]
    intent_id = f"{id_prefix}-{suffix}"
    idempotency_key = f"{id_prefix}|{suffix}"
    store = IntentStore(
        path=(intent_store_path or None),
        url=(database_url or None),
    )

    record, created = store.create_intent(
        intent_id=intent_id,
        idempotency_key=idempotency_key,
        symbol=symbol.upper(),
        side="buy",
        quantity=max(0.0001, float(qty)),
        decision_ts=now_iso,
        metadata={"source": "live_cutover_drill", "created_at": now_iso},
        status="PENDING_SUBMIT",
    )
    claimed = bool(store.claim_for_submit(record.intent_id))
    broker_order_id = f"{id_prefix}-broker-{suffix}"
    store.mark_submitted(record.intent_id, broker_order_id)
    store.record_fill(
        record.intent_id,
        fill_qty=max(0.0001, float(qty)),
        fill_price=max(0.01, float(price)),
        fill_ts=now_iso,
    )
    store.close_intent(record.intent_id, final_status="FILLED")
    persisted = store.get_intent(record.intent_id)
    fills = store.list_fills(record.intent_id)

    return {
        "created": bool(created),
        "claimed_for_submit": claimed,
        "intent_id": record.intent_id,
        "broker_order_id": broker_order_id,
        "database_url": getattr(store, "database_url", ""),
        "database_scheme": urlparse(getattr(store, "database_url", "")).scheme,
        "final_status": str(getattr(persisted, "status", "") or ""),
        "fill_count": len(fills),
        "durability_ok": bool(persisted is not None and len(fills) > 0),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    execution_mode = str(args.execution_mode or "").strip().lower() or "live"
    startup_ok, startup_error = _run_startup_validation(execution_mode)

    effective_database_url = _normalize_database_url(
        str(
            args.database_url
            or get_env("DATABASE_URL", "", cast=str, resolve_aliases=False)
        )
    )
    parsed_database_url = urlparse(effective_database_url) if effective_database_url else None
    non_sqlite_database = bool(
        parsed_database_url is not None
        and bool(parsed_database_url.scheme)
        and parsed_database_url.scheme != "sqlite"
    )
    if execution_mode == "live" and not non_sqlite_database:
        return {
            "status": "failed",
            "execution_mode": execution_mode,
            "startup_validation_ok": startup_ok,
            "startup_validation_error": startup_error,
            "database_url_ok": False,
            "database_url_error": "live mode requires non-sqlite DATABASE_URL",
            "oms_drill": None,
        }

    oms_result = _run_oms_durability_drill(
        symbol=str(args.symbol or "AAPL"),
        qty=float(args.qty),
        price=float(args.price),
        id_prefix=str(args.id_prefix or "cutover-drill").strip() or "cutover-drill",
        intent_store_path=str(args.intent_store_path or "").strip(),
        database_url=effective_database_url,
    )
    ok = bool(startup_ok and oms_result.get("durability_ok"))
    return {
        "status": "ok" if ok else "failed",
        "execution_mode": execution_mode,
        "startup_validation_ok": startup_ok,
        "startup_validation_error": startup_error,
        "database_url_ok": True,
        "database_url_error": None,
        "oms_drill": oms_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = _run(args)

    level = "info" if payload.get("status") == "ok" else "error"
    log_fn = getattr(logger, level, logger.info)
    log_fn("LIVE_CUTOVER_DRILL_RESULT", extra=payload)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
