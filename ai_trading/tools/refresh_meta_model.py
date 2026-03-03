from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.meta_learning.core import (
    load_model_checkpoint,
    retrain_meta_learner,
    validate_trade_data_quality,
)
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh the meta-learning model artifact with current runtime dependencies.",
    )
    parser.add_argument(
        "--trade-log-path",
        type=str,
        default=str(get_env("TRADE_LOG_PATH", "trades.csv", cast=str) or "trades.csv"),
        help="CSV trade history path used for retraining.",
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
        default_relative="trades.csv",
    )
    quality_report = validate_trade_data_quality(str(trade_log_path))

    ok = retrain_meta_learner(
        trade_log_path=str(trade_log_path),
        model_path=str(model_path),
        history_path=str(history_path),
        min_samples=max(1, int(args.min_samples)),
    )
    loaded = load_model_checkpoint(str(model_path))
    model_loaded = loaded is not None
    payload: dict[str, Any] = {
        "trade_log_path": str(trade_log_path),
        "model_path": str(model_path),
        "history_path": str(history_path),
        "min_samples": max(1, int(args.min_samples)),
        "retrain_succeeded": bool(ok),
        "model_load_verified": bool(model_loaded),
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
        "status": "ok" if ok and model_loaded else "failed",
    }
    return payload


def main(argv: list[str] | None = None) -> int:
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
