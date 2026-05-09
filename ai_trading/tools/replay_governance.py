from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.config.management import (
    clear_runtime_env_overrides,
    get_env,
    set_runtime_env_override,
)
from ai_trading.core import bot_engine
from ai_trading.env import ensure_dotenv_loaded
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run replay governance once in maintenance mode.",
    )
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="Override replay timestamp (ISO-8601). Defaults to current UTC time.",
    )
    parser.add_argument(
        "--market-open-now",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set market-open context passed to governance scheduler checks.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bypass schedule gating and run replay immediately.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional output path for command summary JSON payload.",
    )
    parser.add_argument("--replay-data-dir", type=str, default=None)
    parser.add_argument("--replay-output-dir", type=str, default=None)
    parser.add_argument("--replay-seed", type=int, default=None)
    parser.add_argument("--replay-symbols", type=str, default=None)
    parser.add_argument("--replay-timeframes", type=str, default=None)
    parser.add_argument("--replay-start-date", type=str, default=None)
    parser.add_argument("--replay-end-date", type=str, default=None)
    parser.add_argument(
        "--simulate-fills",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enforce-oms-gates",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--require-non-regression",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--clip-intents-to-caps",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--replay-max-symbol-notional", type=float, default=None)
    parser.add_argument("--replay-max-gross-notional", type=float, default=None)
    return parser


def _parse_now(raw: str | None) -> datetime:
    if raw is None:
        return datetime.now(UTC)
    text = str(raw or "").strip()
    if not text:
        return datetime.now(UTC)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _resolve_replay_output_path(now: datetime) -> Path:
    output_dir_raw = str(
        get_env("AI_TRADING_REPLAY_OUTPUT_DIR", "runtime/replay_outputs", cast=str)
        or ""
    ).strip()
    output_dir = resolve_runtime_artifact_path(
        output_dir_raw or "runtime/replay_outputs",
        default_relative="runtime/replay_outputs",
        for_write=True,
    )
    return output_dir / f"replay_hash_{now.strftime('%Y%m%d')}.json"


def _collect_replay_snapshot(path: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return snapshot
    payload = json.loads(path.read_text(encoding="utf-8"))
    violations = payload.get("violations", [])
    counterfactual = payload.get("counterfactual")
    counterfactual_mapping = counterfactual if isinstance(counterfactual, dict) else {}
    counterfactual_passed = counterfactual_mapping.get("passed")
    snapshot.update(
        {
            "ts": payload.get("ts"),
            "rows": int(payload.get("rows", 0) or 0),
            "orders_submitted": int(payload.get("orders_submitted", 0) or 0),
            "fill_events": int(payload.get("fill_events", 0) or 0),
            "cap_adjustments_count": int(payload.get("cap_adjustments_count", 0) or 0),
            "violations_count": int(len(violations) if isinstance(violations, list) else 0),
            "violations_by_code": dict(payload.get("violations_by_code", {}) or {}),
            "counterfactual_passed": counterfactual_passed is True,
            "counterfactual_available": "passed" in counterfactual_mapping,
            "counterfactual_reason": counterfactual_mapping.get("reason"),
        }
    )
    live_cost_alignment = payload.get("live_cost_alignment")
    if isinstance(live_cost_alignment, dict):
        snapshot["live_cost_alignment"] = dict(live_cost_alignment)
    return snapshot


def _artifact_mtime_ns(path: Path) -> int | None:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return None


def _apply_runtime_overrides(args: argparse.Namespace) -> list[str]:
    overrides: dict[str, str] = {}

    string_overrides = {
        "AI_TRADING_REPLAY_DATA_DIR": args.replay_data_dir,
        "AI_TRADING_REPLAY_OUTPUT_DIR": args.replay_output_dir,
        "AI_TRADING_REPLAY_SYMBOLS": args.replay_symbols,
        "AI_TRADING_REPLAY_TIMEFRAMES": args.replay_timeframes,
        "AI_TRADING_REPLAY_START_DATE": args.replay_start_date,
        "AI_TRADING_REPLAY_END_DATE": args.replay_end_date,
    }
    for key, value in string_overrides.items():
        if value is not None:
            overrides[key] = str(value)

    int_overrides = {
        "AI_TRADING_REPLAY_SEED": args.replay_seed,
    }
    for key, value in int_overrides.items():
        if value is not None:
            overrides[key] = str(int(value))

    float_overrides = {
        "AI_TRADING_REPLAY_MAX_SYMBOL_NOTIONAL": args.replay_max_symbol_notional,
        "AI_TRADING_REPLAY_MAX_GROSS_NOTIONAL": args.replay_max_gross_notional,
    }
    for key, value in float_overrides.items():
        if value is not None:
            overrides[key] = str(float(value))

    bool_overrides = {
        "AI_TRADING_REPLAY_SIMULATE_FILLS": args.simulate_fills,
        "AI_TRADING_REPLAY_ENFORCE_OMS_GATES": args.enforce_oms_gates,
        "AI_TRADING_REPLAY_REQUIRE_NON_REGRESSION": args.require_non_regression,
        "AI_TRADING_REPLAY_CLIP_INTENTS_TO_CAPS": args.clip_intents_to_caps,
    }
    for key, value in bool_overrides.items():
        if value is not None:
            overrides[key] = "1" if bool(value) else "0"

    applied_keys: list[str] = []
    for key, value in overrides.items():
        set_runtime_env_override(key, value)
        applied_keys.append(key)
    return applied_keys


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
    )


def run_replay_governance(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    ensure_dotenv_loaded()
    now = _parse_now(args.now)
    override_keys = _apply_runtime_overrides(args)
    try:
        replay_path = _resolve_replay_output_path(now)
        before_mtime_ns = _artifact_mtime_ns(replay_path)
        started_mono = time.monotonic()
        state = bot_engine.BotState()
        try:
            bot_engine._run_replay_governance(
                state,
                now=now,
                market_open_now=bool(args.market_open_now),
                force=bool(args.force),
            )
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            replay_snapshot = _collect_replay_snapshot(replay_path)
            error_text = str(exc)
            status = (
                "blocked"
                if error_text == "REPLAY_POLICY_NON_REGRESSION_FAILED"
                else "failed"
            )
            blocked_payload: dict[str, Any] = {
                "status": status,
                "reason": error_text or type(exc).__name__,
                "now": now.isoformat(),
                "force": bool(args.force),
                "market_open_now": bool(args.market_open_now),
                "fresh_artifact": False,
                "elapsed_sec": max(0.0, time.monotonic() - started_mono),
                "last_replay_run_date": (
                    state.last_replay_run_date.isoformat()
                    if getattr(state, "last_replay_run_date", None) is not None
                    else None
                ),
                "replay": replay_snapshot,
                "error": {
                    "type": type(exc).__name__,
                    "message": error_text,
                },
            }
            if args.summary_json is not None:
                _write_summary(Path(args.summary_json), blocked_payload)
                blocked_payload["summary_path"] = str(args.summary_json)
            logger.info(
                "REPLAY_GOVERNANCE_TOOL_BLOCKED"
                if status == "blocked"
                else "REPLAY_GOVERNANCE_TOOL_FAILED",
                extra={
                    "force": bool(args.force),
                    "market_open_now": bool(args.market_open_now),
                    "replay_path": str(replay_path),
                    "status": status,
                    "reason": blocked_payload["reason"],
                },
            )
            return blocked_payload
        replay_snapshot = _collect_replay_snapshot(replay_path)
        after_mtime_ns = _artifact_mtime_ns(replay_path)
        fresh_artifact = bool(
            replay_snapshot.get("exists")
            and after_mtime_ns is not None
            and (before_mtime_ns is None or after_mtime_ns > before_mtime_ns)
        )
        counterfactual_passed = bool(replay_snapshot.get("counterfactual_passed"))
        counterfactual_available = bool(replay_snapshot.get("counterfactual_available"))
        no_baseline_counterfactual = (
            str(replay_snapshot.get("counterfactual_reason") or "") == "no_baseline_summary"
        )
        blocked_counterfactual = bool(
            fresh_artifact
            and (not counterfactual_passed or not counterfactual_available or no_baseline_counterfactual)
        )
        payload: dict[str, Any] = {
            "status": "blocked" if blocked_counterfactual else ("ok" if fresh_artifact else "failed"),
            "now": now.isoformat(),
            "force": bool(args.force),
            "market_open_now": bool(args.market_open_now),
            "fresh_artifact": fresh_artifact,
            "last_replay_run_date": (
                state.last_replay_run_date.isoformat()
                if getattr(state, "last_replay_run_date", None) is not None
                else None
            ),
            "replay": replay_snapshot,
        }
        if not fresh_artifact:
            payload["reason"] = "missing_fresh_replay_artifact"
            payload["elapsed_sec"] = max(0.0, time.monotonic() - started_mono)
        elif blocked_counterfactual:
            payload["reason"] = (
                "counterfactual_no_baseline"
                if no_baseline_counterfactual
                else "counterfactual_non_regression_insufficient"
            )
            payload["elapsed_sec"] = max(0.0, time.monotonic() - started_mono)
        if args.summary_json is not None:
            _write_summary(Path(args.summary_json), payload)
            payload["summary_path"] = str(args.summary_json)
        logger.info(
            "REPLAY_GOVERNANCE_TOOL_COMPLETE",
            extra={
                "force": bool(args.force),
                "market_open_now": bool(args.market_open_now),
                "replay_path": replay_snapshot.get("path"),
                "status": payload["status"],
                "replay_exists": bool(replay_snapshot.get("exists")),
                "fresh_artifact": fresh_artifact,
                "violations_count": int(replay_snapshot.get("violations_count", 0) or 0),
                "cap_adjustments_count": int(replay_snapshot.get("cap_adjustments_count", 0) or 0),
            },
        )
        return payload
    finally:
        clear_runtime_env_overrides(override_keys)


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run_replay_governance(argv)
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - defensive CLI branch
        logger.error("REPLAY_GOVERNANCE_TOOL_FAILED", extra={"error": str(exc)}, exc_info=True)
        return 1
    sys.stdout.write(f"{json.dumps(payload, sort_keys=True, indent=2, default=str)}\n")
    if payload.get("status") == "ok":
        return 0
    if payload.get("status") == "blocked":
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
