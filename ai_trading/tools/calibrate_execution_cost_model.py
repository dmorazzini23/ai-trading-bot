from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_trading.logging import get_logger
from ai_trading.tca.rollups import calibrate_cost_model_from_tca

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate execution cost model from TCA records (with slippage fallback).",
    )
    parser.add_argument("--tca-path", type=str, default="", help="Optional TCA JSONL path override.")
    parser.add_argument("--model-path", type=str, default="", help="Optional model JSON path override.")
    parser.add_argument("--lookback-days", type=int, default=None, help="Optional lookback window in days.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = calibrate_cost_model_from_tca(
        tca_path=(str(args.tca_path).strip() or None),
        model_path=(str(args.model_path).strip() or None),
        lookback_days=args.lookback_days,
    )
    logger.info(
        "EXEC_COST_MODEL_CALIBRATE_RESULT",
        extra={
            "records": int(result.get("records", 0) or 0),
            "tca_records": int(result.get("tca_records", 0) or 0),
            "slippage_records": int(result.get("slippage_records", 0) or 0),
            "model_path": str(result.get("model_path", "")),
        },
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
