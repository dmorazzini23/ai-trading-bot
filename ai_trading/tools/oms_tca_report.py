from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ai_trading.config.management import get_env
from ai_trading.tca.event_analytics import summarize_oms_event_tca


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate event-driven TCA summary from immutable OMS events.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=str(get_env("DATABASE_URL", "", cast=str, resolve_aliases=False) or ""),
        help="Optional DB URL override; defaults to DATABASE_URL.",
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
        help="SQLite fallback path when DATABASE_URL is unset.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Optional lookback window in days.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Maximum rows scanned from event tables.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write JSON summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = summarize_oms_event_tca(
        database_url=(str(args.database_url or "").strip() or None),
        intent_store_path=(str(args.intent_store_path or "").strip() or None),
        lookback_days=(int(args.lookback_days) if args.lookback_days is not None else None),
        limit=max(1, int(args.limit)),
    )
    encoded = json.dumps(payload, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    sys.stdout.write(encoded + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
