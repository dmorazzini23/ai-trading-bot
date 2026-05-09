from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import argparse
import json
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.replay.bad_session import build_replay_dataset_from_bad_session
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
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


def _resolve_artifact_path(path: Path, *, default_relative: str, for_write: bool) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target
    return resolve_runtime_artifact_path(
        target,
        default_relative=default_relative,
        for_write=for_write,
    )


def run_incident_replay(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    log_path = _resolve_artifact_path(
        args.log_path,
        default_relative=str(args.log_path),
        for_write=False,
    )
    output_dir = _resolve_artifact_path(
        args.output_dir,
        default_relative=str(args.output_dir),
        for_write=True,
    )
    offline_output_json = (
        _resolve_artifact_path(
            args.offline_output_json,
            default_relative=str(args.offline_output_json),
            for_write=True,
        )
        if args.offline_output_json is not None
        else None
    )

    report = build_replay_dataset_from_bad_session(
        log_path,
        output_dir=output_dir,
        seed=int(args.seed),
    )
    payload: dict[str, Any] = {
        "status": "ok",
        "authority": {
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "research_only": True,
            "timestamp_authoritative": True,
        },
        "source_log": str(log_path),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "dataset": report,
    }
    if bool(args.run_offline_replay):
        if int(report.get("events", 0) or 0) <= 0:
            raise ValueError("incident replay requested but source log produced zero replay events")
        replay_args: list[str] = ["--data-dir", str(output_dir)]
        symbols = [
            str(symbol).strip().upper()
            for symbol in report.get("symbols", [])
            if str(symbol).strip()
        ]
        if symbols:
            replay_args.extend(["--symbols", ",".join(symbols)])
        if offline_output_json is not None:
            replay_args.extend(["--output-json", str(offline_output_json)])
        replay_payload = run_replay(replay_args)
        payload["offline_replay"] = replay_payload

    summary_target = args.summary_json
    if summary_target is None:
        summary_target = output_dir / "incident_replay_summary.json"
    summary_target = _resolve_artifact_path(
        Path(summary_target),
        default_relative=str(summary_target),
        for_write=True,
    )
    _write_json(summary_target, payload)
    payload["summary_path"] = str(summary_target)
    logger.info(
        "INCIDENT_REPLAY_COMPLETE",
        extra={
            "source_log": str(log_path),
            "output_dir": str(output_dir),
            "summary_path": str(summary_target),
            "symbols": int(len(report.get("symbols", []))),
            "events": int(report.get("events", 0) or 0),
        },
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        run_incident_replay(argv)
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - CLI defensive branch
        logger.error("INCIDENT_REPLAY_FAILED", extra={"error": str(exc)}, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
