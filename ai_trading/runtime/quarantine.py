"""Deterministic sleeve/symbol quarantine controls."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class QuarantineEntry:
    start_ts: datetime
    end_ts: datetime
    trigger_reason: str
    metrics_snapshot: dict[str, Any]


class QuarantineManager:
    def __init__(self) -> None:
        self.sleeves: dict[str, QuarantineEntry] = {}
        self.symbols: dict[str, QuarantineEntry] = {}

    def quarantine_sleeve(
        self,
        sleeve: str,
        *,
        duration: timedelta,
        trigger_reason: str,
        metrics_snapshot: Mapping[str, Any],
    ) -> None:
        now = datetime.now(UTC)
        self.sleeves[str(sleeve)] = QuarantineEntry(
            start_ts=now,
            end_ts=now + duration,
            trigger_reason=trigger_reason,
            metrics_snapshot=dict(metrics_snapshot),
        )

    def quarantine_symbol(
        self,
        symbol: str,
        *,
        duration: timedelta,
        trigger_reason: str,
        metrics_snapshot: Mapping[str, Any],
    ) -> None:
        now = datetime.now(UTC)
        self.symbols[str(symbol).upper()] = QuarantineEntry(
            start_ts=now,
            end_ts=now + duration,
            trigger_reason=trigger_reason,
            metrics_snapshot=dict(metrics_snapshot),
        )

    def _active(self, entry: QuarantineEntry | None, now: datetime) -> bool:
        if entry is None:
            return False
        return now < entry.end_ts

    def is_quarantined(
        self,
        *,
        sleeve: str | None = None,
        symbol: str | None = None,
        now: datetime | None = None,
    ) -> tuple[bool, str | None]:
        ts = now or datetime.now(UTC)
        if sleeve:
            entry = self.sleeves.get(str(sleeve))
            if self._active(entry, ts):
                return True, "SLEEVE_QUARANTINED"
        if symbol:
            entry = self.symbols.get(str(symbol).upper())
            if self._active(entry, ts):
                return True, "SYMBOL_QUARANTINED"
        return False, None

    def to_dict(self) -> dict[str, Any]:
        def _encode(entries: dict[str, QuarantineEntry]) -> dict[str, Any]:
            payload: dict[str, Any] = {}
            for key, entry in entries.items():
                payload[key] = {
                    "start_ts": entry.start_ts.isoformat(),
                    "end_ts": entry.end_ts.isoformat(),
                    "trigger_reason": entry.trigger_reason,
                    "metrics_snapshot": dict(entry.metrics_snapshot),
                }
            return payload

        return {
            "sleeves": _encode(self.sleeves),
            "symbols": _encode(self.symbols),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuarantineManager":
        manager = cls()
        for key, entries in (("sleeves", manager.sleeves), ("symbols", manager.symbols)):
            raw_map = payload.get(key)
            if not isinstance(raw_map, dict):
                continue
            for name, raw in raw_map.items():
                if not isinstance(raw, dict):
                    continue
                try:
                    start_ts = datetime.fromisoformat(str(raw.get("start_ts")))
                    end_ts = datetime.fromisoformat(str(raw.get("end_ts")))
                except (TypeError, ValueError):
                    continue
                if start_ts.tzinfo is None:
                    start_ts = start_ts.replace(tzinfo=UTC)
                if end_ts.tzinfo is None:
                    end_ts = end_ts.replace(tzinfo=UTC)
                entries[str(name)] = QuarantineEntry(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    trigger_reason=str(raw.get("trigger_reason", "")),
                    metrics_snapshot=dict(raw.get("metrics_snapshot", {})),
                )
        return manager


def save_quarantine_state(path: str, manager: QuarantineManager) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manager.to_dict(), sort_keys=True), encoding="utf-8")


def load_quarantine_state(path: str) -> QuarantineManager:
    src = Path(path)
    if not src.exists():
        return QuarantineManager()
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return QuarantineManager()
    if not isinstance(payload, dict):
        return QuarantineManager()
    return QuarantineManager.from_dict(payload)
