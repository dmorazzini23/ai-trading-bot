from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.oms.event_store import EventStore
from ai_trading.oms.intent_store import IntentStore
from ai_trading.oms.simulated_lifecycle import SimulatedLifecycleDriver


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixture_root_must_be_object")
    scenarios_raw = payload.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError("fixture_scenarios_missing")
    scenarios: list[dict[str, Any]] = []
    for index, raw in enumerate(scenarios_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"fixture_scenario_invalid:{index}")
        scenarios.append(dict(raw))
    return scenarios


def _event_type_stream(store: EventStore, intent_id: str) -> list[str]:
    rows = store.list_oms_events(intent_id=intent_id, limit=5000)
    return [str(row.get("event_type") or "").strip().upper() for row in rows]


def _run_live(
    *,
    database_url: str,
    intent_store_path: str,
    scenario_name: str,
    symbol: str,
    side: str,
    quantity: float,
    fill_qty: float,
    fill_price: float | None,
    terminal_status: str,
    last_error: str | None = None,
) -> str:
    store = IntentStore(url=database_url, path=intent_store_path)
    try:
        intent_id = f"replay-live-{scenario_name}"
        record, _created = store.create_intent(
            intent_id=intent_id,
            idempotency_key=f"replay-live-key-{scenario_name}",
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            status="PENDING_SUBMIT",
            decision_ts=datetime.now(UTC).isoformat(),
        )
        store.claim_for_submit(record.intent_id)
        store.mark_submitted(record.intent_id, f"replay-live-broker-{scenario_name}")
        if float(fill_qty) > 0.0:
            store.record_fill(
                record.intent_id,
                fill_qty=float(fill_qty),
                fill_price=fill_price,
                fee=0.0,
            )
        store.close_intent(
            record.intent_id,
            final_status=str(terminal_status),
            last_error=(str(last_error) if last_error not in (None, "") else None),
        )
    finally:
        store.close()
    return intent_id


def _run_simulated(
    *,
    database_url: str,
    intent_store_path: str,
    scenario_name: str,
    symbol: str,
    side: str,
    quantity: float,
    fill_qty: float,
    fill_price: float | None,
    terminal_status: str,
    last_error: str | None = None,
) -> str:
    lifecycle = SimulatedLifecycleDriver(
        enabled=True,
        source="replay_harness",
        database_url=database_url,
        intent_store_path=intent_store_path,
    )
    try:
        ref = lifecycle.open_submitted_intent(
            intent_id=f"replay-sim-{scenario_name}",
            idempotency_key=f"replay-sim-key-{scenario_name}",
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            decision_ts=datetime.now(UTC).isoformat(),
            broker_order_id=f"replay-sim-broker-{scenario_name}",
            strategy_id="replay_harness",
            metadata={"fixture_scenario": scenario_name},
        )
        if ref is None:
            raise RuntimeError(f"sim_open_failed:{scenario_name}")
        ok = lifecycle.record_fill_and_close_intent(
            intent_id=ref.intent_id,
            fill_qty=float(fill_qty),
            fill_price=fill_price,
            fee=0.0,
            fill_ts=datetime.now(UTC).isoformat(),
            terminal_status=str(terminal_status),
            last_error=(str(last_error) if last_error not in (None, "") else None),
            liquidity_flag="SIMULATED",
        )
        if not ok:
            raise RuntimeError(f"sim_close_failed:{scenario_name}")
        return str(ref.intent_id)
    finally:
        lifecycle.close()


def replay_lifecycle_parity(
    *,
    fixture_path: str,
    database_url: str | None = None,
    intent_store_path: str | None = None,
) -> dict[str, Any]:
    fixture = Path(str(fixture_path)).expanduser()
    scenarios = _load_fixture(fixture)
    resolved_database_url = str(
        database_url
        or get_env("DATABASE_URL", "", cast=str, resolve_aliases=False)
        or ""
    ).strip()
    resolved_intent_store_path = str(
        intent_store_path
        or get_env(
            "AI_TRADING_OMS_INTENT_STORE_PATH",
            "runtime/oms_intents.db",
            cast=str,
            resolve_aliases=False,
        )
        or "runtime/oms_intents.db"
    ).strip()
    if not resolved_database_url:
        resolved_database_url = f"sqlite:///{resolved_intent_store_path}"

    event_store = EventStore(
        url=resolved_database_url,
        path=resolved_intent_store_path,
    )
    comparisons: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    try:
        for index, scenario in enumerate(scenarios):
            scenario_name = str(scenario.get("name") or f"scenario-{index + 1}")
            symbol = str(scenario.get("symbol") or "AAPL").strip().upper()
            side = str(scenario.get("side") or "buy").strip().lower()
            quantity = float(scenario.get("quantity") or 1.0)
            fill_qty = float(scenario.get("fill_qty") or 0.0)
            fill_price_raw = scenario.get("fill_price")
            fill_price = (
                float(fill_price_raw)
                if fill_price_raw not in (None, "")
                else None
            )
            terminal_status = str(scenario.get("terminal_status") or "FILLED").strip().upper()
            last_error = (
                str(scenario.get("last_error"))
                if scenario.get("last_error") not in (None, "")
                else None
            )

            live_intent_id = _run_live(
                database_url=resolved_database_url,
                intent_store_path=resolved_intent_store_path,
                scenario_name=scenario_name,
                symbol=symbol,
                side=side,
                quantity=quantity,
                fill_qty=fill_qty,
                fill_price=fill_price,
                terminal_status=terminal_status,
                last_error=last_error,
            )
            simulated_intent_id = _run_simulated(
                database_url=resolved_database_url,
                intent_store_path=resolved_intent_store_path,
                scenario_name=scenario_name,
                symbol=symbol,
                side=side,
                quantity=quantity,
                fill_qty=fill_qty,
                fill_price=fill_price,
                terminal_status=terminal_status,
                last_error=last_error,
            )
            live_stream = _event_type_stream(event_store, live_intent_id)
            simulated_stream = _event_type_stream(event_store, simulated_intent_id)
            parity_ok = live_stream == simulated_stream
            comparison = {
                "name": scenario_name,
                "terminal_status": terminal_status,
                "live_intent_id": live_intent_id,
                "simulated_intent_id": simulated_intent_id,
                "live_stream": live_stream,
                "simulated_stream": simulated_stream,
                "parity_ok": bool(parity_ok),
            }
            comparisons.append(comparison)
            if not parity_ok:
                mismatches.append(comparison)
    finally:
        event_store.close()

    return {
        "ok": len(mismatches) == 0,
        "fixture_path": str(fixture),
        "database_url": resolved_database_url,
        "intent_store_path": resolved_intent_store_path,
        "scenario_count": len(comparisons),
        "mismatch_count": len(mismatches),
        "comparisons": comparisons,
        "mismatches": mismatches,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one lifecycle fixture through live and simulated OMS paths and fail on parity mismatches."
        ),
    )
    parser.add_argument(
        "--fixture",
        required=True,
        type=str,
        help="Path to replay fixture JSON containing scenario list.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="",
        help="Optional DB URL override. Defaults to DATABASE_URL or sqlite intent store path.",
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
        help="SQLite fallback path used when database URL is not configured.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = replay_lifecycle_parity(
        fixture_path=str(args.fixture),
        database_url=(str(args.database_url).strip() or None),
        intent_store_path=(str(args.intent_store_path).strip() or None),
    )
    sys.stdout.write(json.dumps(payload, sort_keys=True, indent=2))
    sys.stdout.write("\n")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
