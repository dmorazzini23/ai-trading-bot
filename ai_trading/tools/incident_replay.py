from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.replay.bad_session import build_replay_dataset_from_bad_session
from ai_trading.tools.offline_replay import run_replay

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic incident replay artifacts from runtime logs.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("runtime/decision_records.jsonl"),
        help="JSONL runtime log path to mine into a replay dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime/replay_bad_session"),
        help="Output directory for generated replay dataset and manifests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed embedded into replay fingerprint metadata.",
    )
    parser.add_argument(
        "--run-offline-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Immediately run offline replay on the generated incident dataset.",
    )
    parser.add_argument(
        "--offline-output-json",
        type=Path,
        default=None,
        help="Optional path for offline replay summary JSON.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional output path for this command's summary payload.",
    )
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run_incident_replay(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = build_replay_dataset_from_bad_session(
        args.log_path,
        output_dir=args.output_dir,
        seed=int(args.seed),
    )
    payload: dict[str, Any] = {
        "status": "ok",
        "source_log": str(args.log_path),
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "dataset": report,
    }
    if bool(args.run_offline_replay):
        replay_args: list[str] = ["--data-dir", str(args.output_dir)]
        if args.offline_output_json is not None:
            replay_args.extend(["--output-json", str(args.offline_output_json)])
        replay_payload = run_replay(replay_args)
        payload["offline_replay"] = replay_payload

    summary_target = args.summary_json
    if summary_target is None:
        summary_target = Path(args.output_dir) / "incident_replay_summary.json"
    _write_json(Path(summary_target), payload)
    payload["summary_path"] = str(summary_target)
    logger.info(
        "INCIDENT_REPLAY_COMPLETE",
        extra={
            "source_log": str(args.log_path),
            "output_dir": str(args.output_dir),
            "summary_path": str(summary_target),
            "symbols": int(len(report.get("symbols", []))),
            "events": int(report.get("events", 0) or 0),
        },
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        run_incident_replay(argv)
    except Exception as exc:  # pragma: no cover - CLI defensive branch
        logger.error("INCIDENT_REPLAY_FAILED", extra={"error": str(exc)}, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
