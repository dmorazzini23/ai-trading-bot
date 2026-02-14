"""Persistent OMS ledger for idempotent client order IDs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class LedgerEntry:
    client_order_id: str
    symbol: str
    bar_ts: str
    qty: float
    side: str
    limit_price: float | None
    ts: str
    broker_order_id: str | None = None
    status: str | None = None


class OrderLedger:
    """Append-only JSONL ledger for client order idempotency."""

    def __init__(self, path: str, lookback_hours: float = 24.0) -> None:
        self._path = Path(path)
        self._lock = Lock()
        self._seen: set[str] = set()
        self._load_recent(lookback_hours)

    def _load_recent(self, lookback_hours: float) -> None:
        if not self._path.exists():
            return
        cutoff = datetime.now(UTC) - timedelta(hours=max(0.0, float(lookback_hours)))
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    client_order_id = entry.get("client_order_id")
                    ts_raw = entry.get("ts")
                    if not client_order_id or not ts_raw:
                        continue
                    try:
                        ts = datetime.fromisoformat(str(ts_raw))
                    except ValueError:
                        continue
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    if ts < cutoff:
                        continue
                    self._seen.add(str(client_order_id))
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning("LEDGER_LOAD_FAILED", extra={"error": str(exc)})

    def seen_client_order_id(self, client_order_id: str) -> bool:
        return client_order_id in self._seen

    def record(self, entry: LedgerEntry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "client_order_id": entry.client_order_id,
            "symbol": entry.symbol,
            "bar_ts": entry.bar_ts,
            "qty": entry.qty,
            "side": entry.side,
            "limit_price": entry.limit_price,
            "ts": entry.ts,
            "broker_order_id": entry.broker_order_id,
            "status": entry.status,
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True))
                fh.write("\n")
            self._seen.add(entry.client_order_id)


def deterministic_client_order_id(
    *,
    salt: str,
    symbol: str,
    bar_ts: str,
    side: str,
    qty: float,
    limit_price: float | None,
    length: int = 16,
) -> str:
    price_bucket = 0 if limit_price is None else int(round(float(limit_price) * 100))
    payload = f"{salt}|{symbol}|{bar_ts}|{side}|{abs(qty)}|{price_bucket}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
