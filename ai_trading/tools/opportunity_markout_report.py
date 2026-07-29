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
from ai_trading.runtime.atomic_io import atomic_write_text
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
    bar_source_raw = report.get("bar_source")
    bar_source = bar_source_raw if isinstance(bar_source_raw, Mapping) else {}
    max_timestamp = str(bar_source.get("max_timestamp") or "").strip()
    available_symbols_raw = bar_source.get("symbols_available")
    available_symbols = (
        {
            str(symbol).strip().upper()
            for symbol in available_symbols_raw
            if str(symbol).strip()
        }
        if isinstance(available_symbols_raw, Sequence)
        and not isinstance(available_symbols_raw, (str, bytes))
        else set()
    )
    governed = {
        str(symbol).strip().upper()
        for symbol in governed_symbols
        if str(symbol).strip()
    }
    outcome_rows = report.get("outcomes")
    required_symbols = (
        {
            str(row.get("symbol") or "").strip().upper()
            for row in outcome_rows
            if isinstance(row, Mapping) and str(row.get("symbol") or "").strip()
        }
        if isinstance(outcome_rows, list)
        else set()
    )
    symbol_source_raw = bar_source.get("by_symbol")
    symbol_source = (
        symbol_source_raw if isinstance(symbol_source_raw, Mapping) else {}
    )
    current_session_symbols = {
        symbol
        for symbol in available_symbols
        if str(
            (
                symbol_source.get(symbol)
                if isinstance(symbol_source.get(symbol), Mapping)
                else {}
            ).get("max_timestamp")
            or ""
        ).startswith(str(report_date))
    }
    one_minute_symbols = {
        symbol
        for symbol in available_symbols
        if (
            symbol_source.get(symbol)
            if isinstance(symbol_source.get(symbol), Mapping)
            else {}
        ).get("one_minute_cadence")
        is True
    }
    freshness_scope = required_symbols or available_symbols
    source_available = bool(bar_source.get("available"))
    source_fresh = bool(
        source_available
        and freshness_scope
        and freshness_scope.issubset(current_session_symbols)
        and freshness_scope.issubset(one_minute_symbols)
    )
    source_reason = (
        "current_session_one_minute_bars_available"
        if source_fresh
        else "bars_missing"
        if not source_available
        else "required_symbol_bars_missing"
        if freshness_scope.difference(available_symbols)
        else "source_date_mismatch"
        if freshness_scope.difference(current_session_symbols)
        else "one_minute_cadence_not_verified"
    )
    source_freshness = {
        "available": source_available,
        "fresh": source_fresh,
        "reason": source_reason,
        "report_date": str(report_date),
        "max_timestamp": max_timestamp or None,
        "available_symbols": sorted(available_symbols),
        "missing_governed_symbols": sorted(governed.difference(available_symbols)),
        "required_symbols": sorted(required_symbols),
        "current_session_symbols": sorted(current_session_symbols),
        "one_minute_symbols": sorted(one_minute_symbols),
        "missing_required_symbols": sorted(
            required_symbols.difference(available_symbols)
        ),
    }
    provenance = dict(bars_provenance or {})
    provenance.update(
        {
            "research_only": True,
            "promotion_eligible": False,
            "runtime_authority": False,
            "runtime_fill_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "schema_verified": source_available,
            "one_minute_cadence_verified": bool(
                freshness_scope
                and freshness_scope.issubset(one_minute_symbols)
            ),
            "source_freshness": source_freshness,
        }
    )
    report["bars_provenance"] = provenance
    report["source_freshness"] = source_freshness
    research_replay_raw = report.get("research_replay")
    research_replay = (
        dict(research_replay_raw)
        if isinstance(research_replay_raw, Mapping)
        else {}
    )
    research_replay["source_fresh"] = source_fresh
    research_replay["source_freshness_reason"] = source_reason
    research_replay["source_max_timestamp"] = max_timestamp or None
    if not source_available:
        research_replay["status"] = "unavailable"
        research_replay["reason"] = "bars_missing"
    elif not source_fresh:
        research_replay["status"] = "source_stale"
        research_replay["reason"] = source_reason
    report["research_replay"] = research_replay
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
            "runtime_fill_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
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
        atomic_write_text(
            path,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
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
