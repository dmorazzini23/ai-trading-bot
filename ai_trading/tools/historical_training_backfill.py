"""CLI for governed, research-only Alpaca one-minute historical backfill."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Sequence

from ai_trading.alpaca_api import get_data_client_cls
from ai_trading.config.management import get_env
from ai_trading.config.managed_secrets import hydrate_managed_secrets
from ai_trading.data.historical_backfill import (
    AlpacaHistoricalBarFetcher,
    HistoricalBackfillIncompleteError,
    HistoricalBackfillSpec,
    materialize_historical_backfill,
    normalize_governed_symbols,
)
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text


logger = get_logger(__name__)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _build_alpaca_historical_client():
    hydrate_managed_secrets(
        required_keys=(
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY",
        )
    )
    client_cls = get_data_client_cls()
    oauth = str(get_env("ALPACA_OAUTH", "", cast=str) or "").strip()
    api_key = str(get_env("ALPACA_API_KEY", "", cast=str) or "").strip()
    secret_key = str(get_env("ALPACA_SECRET_KEY", "", cast=str) or "").strip()
    if oauth and (api_key or secret_key):
        raise RuntimeError(
            "Provide either ALPACA_OAUTH or ALPACA_API_KEY/ALPACA_SECRET_KEY, not both"
        )
    if oauth:
        return client_cls(oauth_token=oauth)
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials for historical backfill"
        )
    return client_cls(api_key=api_key, secret_key=secret_key)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize governed Alpaca one-minute bars for research-only training",
    )
    parser.add_argument("--symbols", default="AAPL,AMZN,MSFT")
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/historical_training"),
    )
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--adjustment", default="raw")
    parser.add_argument("--window-sessions", type=int, default=10)
    parser.add_argument("--request-limit", type=int, default=10_000)
    parser.add_argument("--max-missing-ratio", type=float, default=0.02)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Materialize and report incomplete data without a failing exit status",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the structured backfill result for downstream research automation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        symbols = normalize_governed_symbols(str(args.symbols))
        spec = HistoricalBackfillSpec(
            symbols=symbols,
            start_date=args.start,
            end_date=args.end,
            output_root=args.output_dir,
            feed=str(args.feed),
            adjustment=str(args.adjustment),
            window_sessions=int(args.window_sessions),
            request_limit=int(args.request_limit),
            max_missing_ratio=float(args.max_missing_ratio),
            fail_on_incomplete=not bool(args.allow_incomplete),
        )
    except ValueError as exc:
        parser.error(str(exc))
    fetcher = AlpacaHistoricalBarFetcher(_build_alpaca_historical_client())
    try:
        result = materialize_historical_backfill(spec, fetcher=fetcher)
    except HistoricalBackfillIncompleteError as exc:
        if args.output_json is not None:
            atomic_write_text(
                args.output_json,
                json.dumps(exc.result.as_dict(), indent=2, sort_keys=True),
            )
        logger.error(
            "HISTORICAL_BACKFILL_INCOMPLETE",
            extra=exc.result.as_dict(),
        )
        return 2
    if args.output_json is not None:
        atomic_write_text(
            args.output_json,
            json.dumps(result.as_dict(), indent=2, sort_keys=True),
        )
    logger.info("HISTORICAL_BACKFILL_COMPLETE", extra=result.as_dict())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
