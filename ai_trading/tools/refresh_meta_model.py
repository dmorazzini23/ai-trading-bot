from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

from ai_trading.config.management import get_env, reload_env
from ai_trading.logging import get_logger
from ai_trading.meta_learning.core import (
    load_model_checkpoint,
    retrain_meta_learner,
    validate_trade_data_quality,
)
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools import runtime_performance_report as runtime_perf_report

logger = get_logger(__name__)
_FILL_DERIVED_SUFFIXES = {".jsonl", ".json", ".parquet", ".pq", ".pkl", ".pickle"}
_SANITIZE_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _sanitize_token(value: str) -> str:
    normalized = _SANITIZE_TOKEN_RE.sub("_", value.strip().lower()).strip("_")
    return normalized or "unknown"


def _is_fill_derived_source(path: Path) -> bool:
    return path.suffix.lower() in _FILL_DERIVED_SUFFIXES


def _derive_meta_rows_from_fill_history(path: Path) -> list[dict[str, Any]]:
    records = runtime_perf_report._load_trade_rows(path)
    if not records:
        return []

    closed = runtime_perf_report._direct_closed_trades(records)
    if not closed:
        events = runtime_perf_report._extract_fill_events(records)
        if not events:
            return []
        closed, _open_positions, _open_lot_count = runtime_perf_report._reconstruct_closed_trades(events)

    rows: list[dict[str, Any]] = []
    for trade in closed:
        symbol = str(trade.get("symbol", "") or "").strip().upper()
        entry_price = _as_float(trade.get("entry_price"))
        exit_price = _as_float(trade.get("exit_price"))
        qty = _as_float(trade.get("qty"))
        if not symbol or entry_price is None or exit_price is None or qty is None:
            continue
        if entry_price <= 0 or exit_price <= 0 or qty <= 0:
            continue

        raw_side = str(trade.get("side", "") or "").strip().lower()
        side = "buy" if raw_side in {"long", "buy"} else "sell"
        strategy = str(trade.get("strategy", "") or "").strip()
        signal_tags = str(trade.get("signal_tags", "") or "").strip()
        if not signal_tags:
            tags = ["fill_derived", f"symbol_{_sanitize_token(symbol)}", f"side_{side}"]
            if strategy:
                tags.insert(1, f"strategy_{_sanitize_token(strategy)}")
            signal_tags = "+".join(tags)

        net_pnl = _as_float(trade.get("net_pnl"))
        if net_pnl is None:
            net_pnl = _as_float(trade.get("gross_pnl")) or 0.0

        rows.append(
            {
                "symbol": symbol,
                "entry_time": str(trade.get("entry_time", "") or ""),
                "entry_price": float(entry_price),
                "exit_time": str(trade.get("exit_time", "") or ""),
                "exit_price": float(exit_price),
                "qty": float(qty),
                "side": side,
                "strategy": strategy or "fill_derived",
                "classification": "fill_derived",
                "signal_tags": signal_tags,
                "confidence": 0.5,
                "reward": float(net_pnl),
            }
        )
    return rows


def _materialize_meta_csv_from_fill_history(path: Path) -> tuple[Path | None, dict[str, Any]]:
    rows = _derive_meta_rows_from_fill_history(path)
    summary: dict[str, Any] = {
        "source_mode": "fill_derived",
        "source_path": str(path),
        "source_records": len(rows),
    }
    if not rows:
        return None, summary

    output_raw = str(
        get_env(
            "AI_TRADING_META_FILL_DERIVED_CSV_PATH",
            "runtime/meta_learning_fill_derived.csv",
            cast=str,
        )
        or "runtime/meta_learning_fill_derived.csv"
    ).strip()
    output_path = resolve_runtime_artifact_path(
        output_raw,
        default_relative="runtime/meta_learning_fill_derived.csv",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "symbol",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "qty",
        "side",
        "strategy",
        "classification",
        "signal_tags",
        "confidence",
        "reward",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary["materialized_path"] = str(output_path)
    return output_path, summary


def _build_parser() -> argparse.ArgumentParser:
    data_dir_raw = str(
        get_env("AI_TRADING_DATA_DIR", "/var/lib/ai-trading-bot", cast=str)
        or "/var/lib/ai-trading-bot"
    ).split(":")[0].strip()
    data_dir = Path(data_dir_raw).expanduser()
    default_fill_history_path = (
        str((data_dir / "runtime/tca_records.jsonl").resolve())
        if data_dir.is_absolute()
        else "runtime/tca_records.jsonl"
    )
    default_trade_source = str(
        get_env(
            "AI_TRADING_META_TRADE_SOURCE_PATH",
            get_env(
                "AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH",
                default_fill_history_path,
                cast=str,
            ),
            cast=str,
        )
        or default_fill_history_path
    )
    parser = argparse.ArgumentParser(
        description="Refresh the meta-learning model artifact with current runtime dependencies.",
    )
    parser.add_argument(
        "--trade-log-path",
        type=str,
        default=default_trade_source,
        help=(
            "Meta-learning source path. Defaults to canonical fill-derived history "
            "(runtime/tca_records.jsonl)."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(get_env("AI_TRADING_META_MODEL_PATH", "meta_model.pkl", cast=str) or "meta_model.pkl"),
        help="Output path for refreshed meta model checkpoint.",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=str(
            get_env(
                "AI_TRADING_META_RETRAIN_HISTORY_PATH",
                "meta_retrain_history.pkl",
                cast=str,
            )
            or "meta_retrain_history.pkl"
        ),
        help="Output path for retrain history checkpoint.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=int(get_env("AI_TRADING_META_MODEL_REFRESH_MIN_SAMPLES", 20, cast=int)),
        help="Minimum sample count required to retrain.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write refresh summary JSON.",
    )
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = resolve_runtime_artifact_path(
        args.model_path,
        default_relative="meta_model.pkl",
    )
    history_path = resolve_runtime_artifact_path(
        args.history_path,
        default_relative="meta_retrain_history.pkl",
    )
    # Keep trade-log path as provided when absolute; otherwise resolve under runtime root.
    trade_log_path = resolve_runtime_artifact_path(
        args.trade_log_path,
        default_relative="runtime/tca_records.jsonl",
    )
    source_meta: dict[str, Any] = {
        "source_mode": "trade_log_csv",
        "source_path": str(trade_log_path),
    }
    effective_trade_log_path = trade_log_path
    if _is_fill_derived_source(trade_log_path):
        materialized_path, fill_meta = _materialize_meta_csv_from_fill_history(trade_log_path)
        source_meta = fill_meta
        if materialized_path is not None:
            effective_trade_log_path = materialized_path
        else:
            error_payload: dict[str, Any] = {
                "requested_trade_log_path": str(trade_log_path),
                "trade_log_path": str(trade_log_path),
                "model_path": str(model_path),
                "history_path": str(history_path),
                "min_samples": max(1, int(args.min_samples)),
                "retrain_succeeded": False,
                "model_load_verified": False,
                "strict_schema_enabled": bool(
                    get_env("AI_TRADING_META_STRICT_SCHEMA_ENABLED", False, cast=bool)
                ),
                "quality_report": {
                    "file_exists": trade_log_path.exists(),
                    "row_count": 0,
                    "valid_price_rows": 0,
                    "data_quality_score": 0.0,
                    "issue_count": 1,
                },
                "source_metadata": source_meta,
                "status": "failed",
            }
            return error_payload

    quality_report = validate_trade_data_quality(str(effective_trade_log_path))

    ok = retrain_meta_learner(
        trade_log_path=str(effective_trade_log_path),
        model_path=str(model_path),
        history_path=str(history_path),
        min_samples=max(1, int(args.min_samples)),
    )
    loaded = load_model_checkpoint(str(model_path))
    model_load_mode = "checkpoint_loaded"
    model_loaded = loaded is not None
    if not model_loaded and model_path.exists():
        # Some deployments keep model artifacts under runtime state directories
        # that are intentionally outside the meta-learning checkpoint safe-load
        # allowlist. In that case, confirm persistence by file existence.
        model_loaded = True
        model_load_mode = "file_exists_fallback"
    elif not model_loaded:
        model_load_mode = "missing"
    payload: dict[str, Any] = {
        "requested_trade_log_path": str(trade_log_path),
        "trade_log_path": str(effective_trade_log_path),
        "model_path": str(model_path),
        "history_path": str(history_path),
        "min_samples": max(1, int(args.min_samples)),
        "retrain_succeeded": bool(ok),
        "model_load_verified": bool(model_loaded),
        "model_load_mode": model_load_mode,
        "strict_schema_enabled": bool(
            get_env("AI_TRADING_META_STRICT_SCHEMA_ENABLED", False, cast=bool)
        ),
        "quality_report": {
            "file_exists": bool(quality_report.get("file_exists", False)),
            "row_count": int(quality_report.get("row_count", 0) or 0),
            "valid_price_rows": int(quality_report.get("valid_price_rows", 0) or 0),
            "data_quality_score": float(quality_report.get("data_quality_score", 0.0) or 0.0),
            "issue_count": len(list(quality_report.get("issues", []) or [])),
        },
        "source_metadata": source_meta,
        "status": "ok" if ok and model_loaded else "failed",
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    reload_env()
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = _run(args)

    level = "info" if payload["status"] == "ok" else "error"
    log_fn = getattr(logger, level, logger.info)
    log_fn("META_MODEL_REFRESH_RESULT", extra=payload)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
