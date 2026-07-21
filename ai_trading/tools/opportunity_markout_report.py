"""Build governed 1/3/5-bar shadow markouts from decision-journal records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd

from ai_trading.analytics.opportunity_markouts import (
    DEFAULT_GOVERNED_SYMBOLS,
    DEFAULT_MARKOUT_HORIZONS,
    resolve_opportunity_markouts,
)
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _timestamp(row: Mapping[str, Any]) -> str:
    journal = row.get("decision_journal")
    journal_map = journal if isinstance(journal, Mapping) else {}
    return str(
        journal_map.get("source_timestamp")
        or journal_map.get("decision_ts")
        or row.get("bar_ts")
        or journal_map.get("bar_ts")
        or ""
    )


def read_decision_jsonl(
    path: Path,
    *,
    report_date: str | None = None,
) -> list[dict[str, Any]]:
    """Read valid decision objects, optionally scoped by source date."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if report_date and not _timestamp(payload).startswith(report_date):
                continue
            rows.append(payload)
    return rows


def load_governed_bars(
    bars_dir: Path,
    *,
    symbols: Sequence[str],
    timestamp_column: str = "timestamp",
) -> dict[str, pd.DataFrame]:
    """Load one CSV per governed symbol without fetching or mutating data."""

    frames: dict[str, pd.DataFrame] = {}
    for symbol_raw in symbols:
        symbol = str(symbol_raw or "").strip().upper()
        if not symbol:
            continue
        path = bars_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if timestamp_column not in frame.columns or "close" not in frame.columns:
            continue
        frame.index = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        frames[symbol] = frame
    return frames


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_historical_bars_artifact(
    artifact_path: Path,
    *,
    symbols: Sequence[str],
) -> tuple[Path, dict[str, Any]]:
    """Verify a quality-passed research-only backfill before using its bars."""

    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("historical backfill artifact is unreadable") from exc
    if not isinstance(payload, Mapping) or not payload.get("quality_passed"):
        raise ValueError("historical backfill artifact did not pass data quality")
    authority = payload.get("authority")
    authority_map = authority if isinstance(authority, Mapping) else {}
    if not (
        authority_map.get("research_only") is True
        and authority_map.get("evidence_type") == "historical_research"
        and authority_map.get("promotion_eligible") is False
        and authority_map.get("promotion_authority") is False
        and authority_map.get("live_money_authority") is False
        and authority_map.get("runtime_fill_authority") is False
    ):
        raise ValueError("historical backfill authority is not research-only")
    dataset_dir = Path(str(payload.get("dataset_dir") or "")).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise ValueError("historical backfill dataset directory is missing")
    rows = payload.get("symbols")
    if not isinstance(rows, list):
        raise ValueError("historical backfill symbol manifest is missing")
    by_symbol = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in rows
        if isinstance(row, Mapping)
    }
    requested = tuple(sorted({str(symbol).strip().upper() for symbol in symbols}))
    for symbol in requested:
        row = by_symbol.get(symbol)
        if row is None or row.get("quality_passed") is not True:
            raise ValueError(f"historical backfill is not quality-passed for {symbol}")
        csv_path = Path(str(row.get("csv_path") or "")).expanduser().resolve()
        try:
            csv_path.relative_to(dataset_dir)
        except ValueError as exc:
            raise ValueError(f"historical bar path escapes dataset for {symbol}") from exc
        expected_hash = str(row.get("content_sha256") or "").strip().lower()
        if not expected_hash or not csv_path.is_file():
            raise ValueError(f"historical bar artifact is missing for {symbol}")
        if _sha256_file(csv_path) != expected_hash:
            raise ValueError(f"historical bar artifact hash mismatch for {symbol}")
    return dataset_dir, {
        "source": "quality_verified_historical_backfill",
        "artifact_path": str(artifact_path.resolve()),
        "dataset_dir": str(dataset_dir),
        "cache_key": payload.get("cache_key"),
        "quality_passed": True,
        "research_only": True,
        "promotion_eligible": False,
        "promotion_authority": False,
        "runtime_authority": False,
        "runtime_fill_authority": False,
        "live_money_authority": False,
    }


def build_opportunity_markout_report(
    *,
    report_date: str,
    decisions: Sequence[Mapping[str, Any]],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    governed_symbols: Sequence[str] = DEFAULT_GOVERNED_SYMBOLS,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    bars_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = resolve_opportunity_markouts(
        decisions,
        bars_by_symbol,
        governed_symbols=governed_symbols,
        horizons=DEFAULT_MARKOUT_HORIZONS,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    report["report_date"] = str(report_date)
    report["decision_rows_scanned"] = len(decisions)
    report["bars_provenance"] = dict(bars_provenance or {})
    return report


def _default_output_path(report_date: str) -> Path:
    root = resolve_runtime_artifact_path(
        "runtime/reports",
        default_relative="runtime/reports",
        for_write=True,
    )
    return root / f"opportunity_markouts_{report_date.replace('-', '')}.json"


def _parse_symbols(raw: str) -> list[str]:
    return sorted(
        {
            token.strip().upper()
            for token in str(raw or "").split(",")
            if token.strip()
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--decisions-jsonl", type=Path, required=True)
    bars_source = parser.add_mutually_exclusive_group(required=True)
    bars_source.add_argument("--bars-dir", type=Path)
    bars_source.add_argument("--historical-backfill-json", type=Path)
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_GOVERNED_SYMBOLS),
        help="Comma-separated governed symbols (default: AAPL,AMZN,MSFT)",
    )
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)

    symbols = _parse_symbols(args.symbols)
    decisions = read_decision_jsonl(
        args.decisions_jsonl,
        report_date=str(args.report_date),
    )
    bars_provenance: dict[str, Any]
    if args.historical_backfill_json is not None:
        try:
            bars_dir, bars_provenance = resolve_historical_bars_artifact(
                args.historical_backfill_json,
                symbols=symbols,
            )
        except ValueError as exc:
            parser.error(str(exc))
    else:
        bars_dir = args.bars_dir
        if bars_dir is None:
            parser.error("a bars source is required")
        bars_provenance = {
            "source": "explicit_bars_directory",
            "dataset_dir": str(bars_dir.resolve()),
            "quality_passed": None,
            "research_only": True,
            "promotion_eligible": False,
            "runtime_authority": False,
        }
    bars = load_governed_bars(
        bars_dir,
        symbols=symbols,
        timestamp_column=str(args.timestamp_column),
    )
    report = build_opportunity_markout_report(
        report_date=str(args.report_date),
        decisions=decisions,
        bars_by_symbol=bars,
        governed_symbols=symbols,
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        bars_provenance=bars_provenance,
    )
    output_path = args.output_json or _default_output_path(str(args.report_date))
    latest_path = args.latest_json or output_path.with_name("opportunity_markouts_latest.json")
    for path in {output_path, latest_path}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    sys.stdout.write(
        json.dumps(
            {
                "path": str(output_path),
                "eligible_opportunities": report["eligible_opportunities"],
                "outcomes_emitted": report["outcomes_emitted"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_opportunity_markout_report",
    "load_governed_bars",
    "main",
    "read_decision_jsonl",
    "resolve_historical_bars_artifact",
]
