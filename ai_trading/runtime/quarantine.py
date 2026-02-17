"""Deterministic sleeve/symbol quarantine controls."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger

logger = get_logger(__name__)


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


def _resolve_state_path(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target

    data_root_raw = str(get_env("AI_TRADING_DATA_DIR", "") or "").strip()
    if data_root_raw:
        data_root = Path(data_root_raw.split(":")[0]).expanduser()
        if data_root.is_absolute():
            return (data_root / target).resolve()

    state_dir_raw = str(os.getenv("STATE_DIRECTORY", "") or "").strip()
    if state_dir_raw:
        state_root = Path(state_dir_raw.split(":")[0]).expanduser()
        if state_root.is_absolute():
            return (state_root / target).resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / target).resolve()


def save_quarantine_state(path: str, manager: QuarantineManager) -> None:
    dest = _resolve_state_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manager.to_dict(), sort_keys=True), encoding="utf-8")


def load_quarantine_state(path: str) -> QuarantineManager:
    src = _resolve_state_path(path)
    if not src.exists():
        return QuarantineManager()
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return QuarantineManager()
    except OSError as exc:
        logger.warning("QUARANTINE_STATE_READ_FAILED", extra={"error": str(exc), "path": str(src)})
        return QuarantineManager()
    if not isinstance(payload, dict):
        return QuarantineManager()
    return QuarantineManager.from_dict(payload)
