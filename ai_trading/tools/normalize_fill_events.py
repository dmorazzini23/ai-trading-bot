from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return float(parsed)


def _resolve_path(configured: str | None) -> Path:
    raw = str(
        configured
        or get_env("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl", cast=str)
        or "runtime/fill_events.jsonl"
    )
    return resolve_runtime_artifact_path(raw, default_relative="runtime/fill_events.jsonl")


def normalize_fill_events_file(path: Path, *, backup: bool) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "reason": "missing_path", "path": str(path)}

    rows: list[str] = []
    scanned = 0
    fill_rows = 0
    updated_rows = 0
    missing_rows = 0
    malformed_rows = 0

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            rows.append("")
            continue
        if not stripped.startswith("{"):
            rows.append(raw_line)
            continue
        scanned += 1
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            malformed_rows += 1
            rows.append(raw_line)
            continue
        if not isinstance(row, dict):
            rows.append(raw_line)
            continue
        if str(row.get("event") or "").strip().lower() != "fill_recorded":
            rows.append(json.dumps(row, sort_keys=True, default=str))
            continue
        fill_rows += 1
        before_price = row.get("fill_price")
        before_qty = row.get("fill_qty")

        fill_price = _safe_float(row.get("fill_price"))
        if fill_price is None:
            for key in ("entry_price", "price", "avg_fill_price", "filled_avg_price"):
                fill_price = _safe_float(row.get(key))
                if fill_price is not None:
                    break
        fill_qty = _safe_float(row.get("fill_qty"))
        if fill_qty is None:
            for key in ("qty", "filled_qty", "quantity"):
                fill_qty = _safe_float(row.get(key))
                if fill_qty is not None:
                    break

        if fill_price is not None:
            row["fill_price"] = float(fill_price)
        if fill_qty is not None:
            row["fill_qty"] = float(fill_qty)

        if row.get("fill_price") in (None, "") or row.get("fill_qty") in (None, ""):
            missing_rows += 1
        if row.get("fill_price") != before_price or row.get("fill_qty") != before_qty:
            updated_rows += 1
        rows.append(json.dumps(row, sort_keys=True, default=str))

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "path": str(path),
        "scanned_rows": int(scanned),
        "fill_rows": int(fill_rows),
        "updated_rows": int(updated_rows),
        "missing_fill_fields": int(missing_rows),
        "malformed_rows": int(malformed_rows),
        "backup_written": bool(backup),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize canonical fill_price/fill_qty in fill_events.jsonl")
    parser.add_argument("--path", type=str, default=None, help="Optional explicit fill_events.jsonl path")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable .bak backup before rewrite",
    )
    args = parser.parse_args()

    path = _resolve_path(args.path)
    summary = normalize_fill_events_file(path, backup=not bool(args.no_backup))
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0 if bool(summary.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
