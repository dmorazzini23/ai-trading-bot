from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_trading.analytics.tca import finalize_stale_pending_tca
from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _resolve_path(value: str | None, *, env_key: str, default_relative: str) -> Path:
    raw = str(value or get_env(env_key, default_relative, cast=str) or default_relative)
    return resolve_runtime_artifact_path(raw, default_relative=default_relative)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact matched pending TCA rows into resolved superseded entries."
    )
    parser.add_argument("--tca-path", type=str, default=None)
    parser.add_argument("--fill-events-path", type=str, default=None)
    parser.add_argument("--stale-after-sec", type=float, default=21600.0)
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--fill-match-window-sec", type=float, default=86400.0)
    parser.add_argument("--fill-qty-tolerance", type=float, default=0.05)
    parser.add_argument("--fill-events-max-rows", type=int, default=300000)
    parser.add_argument(
        "--source",
        type=str,
        default="manual_compaction",
        help="pending_resolved_source marker for compacted/finalized rows",
    )
    args = parser.parse_args()

    tca_path = _resolve_path(
        args.tca_path,
        env_key="AI_TRADING_TCA_PATH",
        default_relative="runtime/tca_records.jsonl",
    )
    fill_events_path = _resolve_path(
        args.fill_events_path,
        env_key="AI_TRADING_FILL_EVENTS_PATH",
        default_relative="runtime/fill_events.jsonl",
    )

    summary = finalize_stale_pending_tca(
        str(tca_path),
        stale_after_seconds=float(args.stale_after_sec),
        max_records=int(max(args.max_records, 0)),
        source=str(args.source or "manual_compaction"),
        fill_events_path=str(fill_events_path),
        fill_match_window_seconds=float(args.fill_match_window_sec),
        fill_qty_tolerance_ratio=float(args.fill_qty_tolerance),
        fill_events_max_records=int(max(args.fill_events_max_rows, 1)),
        compact_matched_pending=True,
    )
    sys.stdout.write(json.dumps(summary, sort_keys=True, default=str) + "\n")
    return 0 if bool(summary.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
