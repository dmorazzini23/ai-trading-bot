from __future__ import annotations

import argparse
import json
import sys
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
    snapshot.update(
        {
            "ts": payload.get("ts"),
            "rows": int(payload.get("rows", 0) or 0),
            "orders_submitted": int(payload.get("orders_submitted", 0) or 0),
            "fill_events": int(payload.get("fill_events", 0) or 0),
            "cap_adjustments_count": int(payload.get("cap_adjustments_count", 0) or 0),
            "violations_count": int(len(violations) if isinstance(violations, list) else 0),
            "violations_by_code": dict(payload.get("violations_by_code", {}) or {}),
            "counterfactual_passed": bool(
                ((payload.get("counterfactual") or {}).get("passed", True))
            ),
        }
    )
    return snapshot


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def run_replay_governance(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    ensure_dotenv_loaded()
    now = _parse_now(args.now)
    override_keys = _apply_runtime_overrides(args)
    try:
        state = bot_engine.BotState()
        bot_engine._run_replay_governance(
            state,
            now=now,
            market_open_now=bool(args.market_open_now),
            force=bool(args.force),
        )
        replay_path = _resolve_replay_output_path(now)
        payload: dict[str, Any] = {
            "status": "ok",
            "now": now.isoformat(),
            "force": bool(args.force),
            "market_open_now": bool(args.market_open_now),
            "last_replay_run_date": (
                state.last_replay_run_date.isoformat()
                if getattr(state, "last_replay_run_date", None) is not None
                else None
            ),
            "replay": _collect_replay_snapshot(replay_path),
        }
        if args.summary_json is not None:
            _write_summary(Path(args.summary_json), payload)
            payload["summary_path"] = str(args.summary_json)
        logger.info(
            "REPLAY_GOVERNANCE_TOOL_COMPLETE",
            extra={
                "force": bool(args.force),
                "market_open_now": bool(args.market_open_now),
                "replay_path": payload["replay"]["path"],
                "replay_exists": bool(payload["replay"]["exists"]),
                "violations_count": int(payload["replay"].get("violations_count", 0) or 0),
                "cap_adjustments_count": int(payload["replay"].get("cap_adjustments_count", 0) or 0),
            },
        )
        return payload
    finally:
        clear_runtime_env_overrides(override_keys)


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run_replay_governance(argv)
    except Exception as exc:  # pragma: no cover - defensive CLI branch
        logger.error("REPLAY_GOVERNANCE_TOOL_FAILED", extra={"error": str(exc)}, exc_info=True)
        return 1
    sys.stdout.write(f"{json.dumps(payload, sort_keys=True, indent=2, default=str)}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
