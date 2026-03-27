#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from typing import Any
import uuid

import pandas as pd


@dataclass(slots=True)
class _DeltaRow:
    symbol: str
    broker_qty: float
    reconstructed_qty: float
    delta_qty: float
    reason: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair open-position reconciliation drift by appending synthetic "
            "reconcile_backfill fill rows to trade_history."
        )
    )
    parser.add_argument(
        "--runtime-report",
        type=Path,
        default=Path("/var/lib/ai-trading-bot/runtime/runtime_performance_report_latest.json"),
        help="Path to runtime performance report JSON.",
    )
    parser.add_argument(
        "--trade-history",
        type=Path,
        default=Path("/var/lib/ai-trading-bot/runtime/trade_history.parquet"),
        help="Path to canonical trade history artifact.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run).",
    )
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid report payload in {path}")
    return payload


def _build_deltas(payload: dict[str, Any]) -> list[_DeltaRow]:
    trade = payload.get("trade_history", {})
    if not isinstance(trade, dict):
        raise ValueError("trade_history missing from runtime report")
    broker_raw = trade.get("broker_open_positions", {})
    reconstructed_raw = trade.get("reconstructed_open_positions", {})
    if not isinstance(broker_raw, dict) or not isinstance(reconstructed_raw, dict):
        raise ValueError("open position maps missing from runtime report")

    broker = {
        str(symbol).strip().upper(): float(qty)
        for symbol, qty in broker_raw.items()
        if str(symbol).strip()
    }
    reconstructed = {
        str(symbol).strip().upper(): float(qty)
        for symbol, qty in reconstructed_raw.items()
        if str(symbol).strip()
    }

    rows: list[_DeltaRow] = []
    for symbol in sorted(set(broker) | set(reconstructed)):
        broker_qty = float(broker.get(symbol, 0.0))
        reconstructed_qty = float(reconstructed.get(symbol, 0.0))
        delta_qty = broker_qty - reconstructed_qty
        if abs(delta_qty) <= 1e-9:
            continue
        reason = "quantity_mismatch"
        if symbol not in broker:
            reason = "missing_in_broker"
        elif symbol not in reconstructed:
            reason = "missing_in_reconstructed"
        rows.append(
            _DeltaRow(
                symbol=symbol,
                broker_qty=broker_qty,
                reconstructed_qty=reconstructed_qty,
                delta_qty=delta_qty,
                reason=reason,
            )
        )
    rows.sort(key=lambda item: abs(item.delta_qty), reverse=True)
    return rows


def _is_parquet_path(path: Path) -> bool:
    return path.suffix.lower() in {".parquet", ".pq"}


def _pickle_sidecar_path(path: Path) -> Path:
    suffix = path.suffix if path.suffix else ""
    return path.with_suffix(f"{suffix}.pkl")


def _normalize_trade_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if normalized.empty:
        return normalized
    for column in (
        "entry_time",
        "exit_time",
        "timestamp",
        "filled_at",
        "executed_at",
        "updated_at",
        "ts",
    ):
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(
                normalized[column],
                errors="coerce",
                utc=True,
            )
    for column in (
        "qty",
        "fill_qty",
        "entry_price",
        "fill_price",
        "expected_price",
        "slippage_bps",
        "fee_amount",
        "fee_bps",
        "confidence",
    ):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def _load_trade_history(path: Path) -> tuple[pd.DataFrame, str, str]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path), "parquet", "parquet"
        except Exception:
            sidecar = _pickle_sidecar_path(path)
            if sidecar.exists():
                return pd.read_pickle(sidecar), "pickle_sidecar", "parquet"
            return pd.read_pickle(path), "pickle_alias", "parquet"
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path), "pickle", "pickle"
    if suffix == ".csv":
        return pd.read_csv(path), "csv", "csv"
    raise ValueError(f"Unsupported trade history format: {path}")


def _last_price_by_symbol(df: pd.DataFrame) -> dict[str, float]:
    if "symbol" not in df.columns:
        return {}
    price_keys = (
        "fill_price",
        "entry_price",
        "price",
        "expected_price",
        "reward",
    )
    out: dict[str, float] = {}
    for row in df.iloc[::-1].to_dict(orient="records"):
        symbol_raw = row.get("symbol")
        symbol = str(symbol_raw or "").strip().upper()
        if not symbol or symbol in out:
            continue
        price_val: float | None = None
        for key in price_keys:
            raw = row.get(key)
            if raw in (None, ""):
                continue
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                continue
            if parsed > 0 and parsed == parsed:
                price_val = parsed
                break
        if price_val is not None:
            out[symbol] = price_val
    return out


def _repair_rows(deltas: list[_DeltaRow], last_price: dict[str, float]) -> list[dict[str, Any]]:
    now_iso = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for delta in deltas:
        qty = abs(float(delta.delta_qty))
        if qty <= 1e-9:
            continue
        side = "buy" if delta.delta_qty > 0 else "sell"
        price = float(last_price.get(delta.symbol, 100.0))
        repair_id = f"recon-repair-{delta.symbol.lower()}-{uuid.uuid4().hex[:16]}"
        rows.append(
            {
                "symbol": delta.symbol,
                "entry_time": now_iso,
                "timestamp": now_iso,
                "ts": now_iso,
                "entry_price": price,
                "fill_price": price,
                "expected_price": price,
                "qty": qty,
                "fill_qty": qty,
                "side": side,
                "status": "filled",
                "strategy": "reconciliation_repair",
                "signal_tags": "reconciliation_repair",
                "classification": "reconciliation_repair",
                "confidence": 1.0,
                "order_id": repair_id,
                "fill_id": f"{repair_id}:{qty}:filled",
                "client_order_id": repair_id[:32],
                "source": "reconcile_backfill",
                "fee_amount": 0.0,
                "fee_bps": 0.0,
                "slippage_bps": 0.0,
                "exit_price": None,
            }
        )
    return rows


def _write_trade_history(path: Path, fmt: str, frame: pd.DataFrame) -> None:
    normalized = _normalize_trade_history_frame(frame)
    if fmt == "parquet":
        normalized.to_parquet(path, index=False)  # pragma: no cover (best effort)
        return
    if fmt == "pickle":
        normalized.to_pickle(path)
        return
    if fmt == "csv":
        normalized.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported write format: {fmt}")


def main() -> int:
    args = _parse_args()
    report_path: Path = args.runtime_report
    trade_history_path: Path = args.trade_history

    report_payload = _load_report(report_path)
    deltas = _build_deltas(report_payload)
    summary: dict[str, Any] = {
        "runtime_report": str(report_path),
        "trade_history": str(trade_history_path),
        "apply": bool(args.apply),
        "mismatch_count": len(deltas),
        "top_mismatches": [
            {
                "symbol": row.symbol,
                "broker_qty": row.broker_qty,
                "reconstructed_qty": row.reconstructed_qty,
                "delta_qty": row.delta_qty,
                "reason": row.reason,
            }
            for row in deltas[:20]
        ],
    }
    if not deltas:
        summary["status"] = "no_op"
        output = json.dumps(summary, indent=2, sort_keys=True)
        print(output)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(output, encoding="utf-8")
        return 0

    frame, loaded_fmt, write_fmt = _load_trade_history(trade_history_path)
    price_map = _last_price_by_symbol(frame)
    repair_rows = _repair_rows(deltas, price_map)
    summary["repair_rows"] = len(repair_rows)
    summary["loaded_format"] = loaded_fmt
    summary["write_format"] = write_fmt
    summary["status"] = "dry_run"

    if args.apply:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_path = Path(f"{trade_history_path}.bak.{stamp}")
        shutil.copy2(trade_history_path, backup_path)
        summary["backup_path"] = str(backup_path)
        if repair_rows:
            frame_out = pd.concat(
                [frame, pd.DataFrame(repair_rows)],
                ignore_index=True,
                sort=False,
            )
        else:
            frame_out = frame
        if write_fmt == "parquet":
            try:
                _write_trade_history(trade_history_path, "parquet", frame_out)
            except Exception as exc:
                sidecar = _pickle_sidecar_path(trade_history_path)
                _write_trade_history(sidecar, "pickle", frame_out)
                summary["write_fallback"] = "pickle_sidecar"
                summary["write_fallback_path"] = str(sidecar)
                summary["write_fallback_cause"] = str(exc)
        else:
            _write_trade_history(trade_history_path, write_fmt, frame_out)
        summary["status"] = "applied"
        summary["rows_before"] = int(len(frame))
        summary["rows_after"] = int(len(frame_out))

    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
