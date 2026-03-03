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


def _parse_log_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
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


def canonical_bad_session_events(log_path: str | Path) -> list[dict[str, Any]]:
    """Load and normalize bad-session events from JSONL logs."""
    path = Path(log_path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_log_line(raw_line)
            if parsed is not None:
                events.append(parsed)
    except OSError as exc:
        logger.error("BAD_SESSION_LOG_READ_FAILED", extra={"path": str(path), "error": str(exc)})
        return []
    events.sort(key=lambda item: (str(item["timestamp"]), str(item["symbol"])))
    return events


def deterministic_replay_fingerprint(log_path: str | Path, *, seed: int = 42) -> str:
    """Return deterministic fingerprint for replay reproducibility checks."""
    events = canonical_bad_session_events(log_path)
    payload = {"seed": int(seed), "events": events}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


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
    meta_path = out_dir / "replay_manifest.json"
    manifest = {
        "source_log": str(log_path),
        "seed": int(seed),
        "fingerprint": fingerprint,
        "symbols": sorted(written),
        "files": written,
    }
    meta_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"fingerprint": fingerprint, "manifest": str(meta_path), "files": written}


__all__ = [
    "build_replay_dataset_from_bad_session",
    "canonical_bad_session_events",
    "deterministic_replay_fingerprint",
]
