"""Deterministic replay helpers for failed/bad live sessions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _coerce_timestamp(payload: dict[str, Any]) -> str:
    direct_fields = (
        payload.get("timestamp"),
        payload.get("ts"),
        payload.get("decision_ts"),
        payload.get("submit_ts"),
        payload.get("first_fill_ts"),
    )
    for value in direct_fields:
        text = str(value or "").strip()
        if text:
            return text
    benchmark = payload.get("benchmark")
    if isinstance(benchmark, dict):
        for key in ("decision_ts", "submit_ts", "first_fill_ts"):
            text = str(benchmark.get(key) or "").strip()
            if text:
                return text
    return ""


def _coerce_price(payload: dict[str, Any]) -> float | None:
    for key in (
        "price",
        "fill_price",
        "fill_vwap",
        "arrival_price",
        "decision_price",
        "submit_price_reference",
        "close",
    ):
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _parse_event_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    symbol = str(payload.get("symbol") or "").strip().upper()
    timestamp = _coerce_timestamp(payload)
    price = _coerce_price(payload)
    if not symbol or not timestamp:
        return None
    if price is None:
        return None
    volume_raw = payload.get("volume")
    if volume_raw in (None, ""):
        volume_raw = payload.get("qty")
    try:
        volume = float(volume_raw or 0.0)
    except (TypeError, ValueError):
        volume = 0.0
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "price": price,
        "volume": volume,
    }


def _load_jsonl_payload_rows(log_path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL rows as ``{'line_no': int, 'payload': dict}`` entries."""

    path = Path(log_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line_no, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            text = raw_line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            rows.append({"line_no": int(line_no), "payload": payload})
    except OSError as exc:
        logger.error("BAD_SESSION_LOG_READ_FAILED", extra={"path": str(path), "error": str(exc)})
        return []
    return rows


def canonical_bad_session_events(log_path: str | Path) -> list[dict[str, Any]]:
    """Load and normalize bad-session events from JSONL logs."""
    payload_rows = _load_jsonl_payload_rows(log_path)
    events: list[dict[str, Any]] = []
    for row in payload_rows:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        parsed = _parse_event_payload(payload)
        if parsed is not None:
            events.append(parsed)
    events.sort(key=lambda item: (str(item["timestamp"]), str(item["symbol"])))
    return events


def deterministic_replay_fingerprint(log_path: str | Path, *, seed: int = 42) -> str:
    """Return deterministic fingerprint for replay reproducibility checks."""
    events = canonical_bad_session_events(log_path)
    payload = {"seed": int(seed), "events": events}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _build_incident_bundle(
    payload_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Write deterministic incident artifacts for replay diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    decision_rows: list[dict[str, Any]] = []
    intent_rows: list[dict[str, Any]] = []
    broker_rows: list[dict[str, Any]] = []

    for row in payload_rows:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        line_no = int(row.get("line_no", 0) or 0)
        ts = _coerce_timestamp(payload)
        symbol_raw = str(payload.get("symbol") or "").strip().upper()
        symbol = symbol_raw if symbol_raw else None
        msg = str(payload.get("msg") or payload.get("event") or "").strip()
        msg_upper = msg.upper()
        normalized = {
            "line_no": line_no,
            "timestamp": ts or None,
            "symbol": symbol,
            "msg": msg or None,
            "payload": payload,
        }

        if msg_upper == "DECISION_RECORD" or "decision" in payload:
            decision_rows.append(normalized)

        if (
            "intent_id" in payload
            or "client_order_id" in payload
            or "idempotency_key" in payload
            or "ORDER_SUBMIT" in msg_upper
            or "INTENT" in msg_upper
        ):
            intent_rows.append(normalized)

        if (
            msg_upper.startswith("BROKER_")
            or msg_upper.startswith("ALPACA_ORDER")
            or "ORDER_ACK" in msg_upper
            or msg_upper.startswith("API_GET_ORDER")
            or msg_upper.startswith("API_CANCEL_ORDER")
            or "BROKER_STATE" in msg_upper
        ):
            broker_rows.append(normalized)

    def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda item: (
                str(item.get("timestamp") or ""),
                int(item.get("line_no") or 0),
                str(item.get("msg") or ""),
            ),
        )

    decision_rows = _sort_rows(decision_rows)
    intent_rows = _sort_rows(intent_rows)
    broker_rows = _sort_rows(broker_rows)

    def _write_jsonl(filename: str, rows: list[dict[str, Any]]) -> str:
        target = output_dir / filename
        lines = [json.dumps(item, sort_keys=True) for item in rows]
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return str(target)

    return {
        "decisions": _write_jsonl("incident_decisions.jsonl", decision_rows),
        "intents": _write_jsonl("incident_intents.jsonl", intent_rows),
        "broker": _write_jsonl("incident_broker.jsonl", broker_rows),
        "counts": {
            "decisions": int(len(decision_rows)),
            "intents": int(len(intent_rows)),
            "broker": int(len(broker_rows)),
        },
    }


def build_replay_dataset_from_bad_session(
    log_path: str | Path,
    *,
    output_dir: str | Path = "runtime/replay_bad_session",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Build deterministic per-symbol replay CSVs from bad-session logs.

    The output format is compatible with ``ai_trading.tools.offline_replay``.
    """
    payload_rows = _load_jsonl_payload_rows(log_path)
    events = canonical_bad_session_events(log_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event["symbol"]), []).append(event)

    written: dict[str, str] = {}
    for symbol, rows in grouped.items():
        target = out_dir / f"{symbol}.csv"
        lines = ["timestamp,open,high,low,close,volume"]
        for row in rows:
            ts = str(row["timestamp"])
            px = float(row["price"])
            volume = float(row.get("volume", 0.0))
            # Deterministic synthetic OHLC from single print.
            lines.append(f"{ts},{px},{px},{px},{px},{volume}")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[symbol] = str(target)

    fingerprint = deterministic_replay_fingerprint(log_path, seed=seed)
    bundle = _build_incident_bundle(payload_rows, output_dir=out_dir)
    meta_path = out_dir / "replay_manifest.json"
    manifest = {
        "source_log": str(log_path),
        "seed": int(seed),
        "fingerprint": fingerprint,
        "symbols": sorted(written),
        "files": written,
        "incident_bundle": bundle,
    }
    meta_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "fingerprint": fingerprint,
        "manifest": str(meta_path),
        "manifest_path": str(meta_path),
        "files": written,
        "symbols": sorted(written),
        "events": int(len(events)),
        "bundle": bundle,
    }


__all__ = [
    "build_replay_dataset_from_bad_session",
    "canonical_bad_session_events",
    "deterministic_replay_fingerprint",
]
