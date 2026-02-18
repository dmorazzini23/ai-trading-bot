#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_ENV_PATH = Path(".env")
DEFAULT_TCA_PATH = Path("runtime/tca_records.jsonl")
DEFAULT_STATE_DIR = Path("runtime/ab_pacing")
DEFAULT_OUTPUT_PATH = Path("runtime/ab_compare.json")
DEFAULT_SERVICE = "ai-trading.service"
PACING_LOG_KEY = "ORDER_PACING_CAP_HIT"


@dataclass(frozen=True)
class Window:
    arm: str
    start: datetime
    end: datetime


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_iso(raw: str) -> datetime:
    text = str(raw).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_env_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _upsert_env(path: Path, key: str, value: str) -> None:
    pattern = re.compile(rf"^\s*{re.escape(key)}=")
    lines = _load_env_lines(path)
    replaced = False
    for idx, line in enumerate(lines):
        if pattern.match(line):
            lines[idx] = f"{key}={value}"
            replaced = True
    if not replaced:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _backup_env(path: Path) -> Path:
    stamp = _now_utc().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    shutil.copy2(path, backup)
    return backup


def _restart_service(service: str) -> None:
    subprocess.run(["sudo", "systemctl", "restart", service], check=True)


def _read_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _load_env_lines(path):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _write_stamp(path: Path, ts: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_utc_iso(ts) + "\n", encoding="utf-8")


def _read_stamp(path: Path) -> datetime:
    return _parse_iso(path.read_text(encoding="utf-8").strip())


def _arm_stamp_path(state_dir: Path, arm: str, kind: str) -> Path:
    return state_dir / f"{arm}_{kind}_utc.txt"


def _as_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(1.0, pct / 100.0)) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _max_drawdown_bps(edges_bps: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for edge in edges_bps:
        equity += edge
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
    return out


def _rows_in_window(rows: list[dict[str, Any]], window: Window) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for row in rows:
        ts_raw = row.get("ts")
        if not ts_raw:
            continue
        try:
            ts = _parse_iso(str(ts_raw))
        except (TypeError, ValueError):
            continue
        if window.start <= ts < window.end:
            picked.append(row)
    return picked


def _count_pacing_hits(service: str, window: Window) -> int | None:
    cmd = [
        "journalctl",
        "-u",
        service,
        "--since",
        _utc_iso(window.start),
        "--until",
        _utc_iso(window.end),
        "--no-pager",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return sum(1 for line in proc.stdout.splitlines() if PACING_LOG_KEY in line)


def _summarize_arm(
    arm: str,
    rows: list[dict[str, Any]],
    window: Window,
    *,
    service: str,
) -> dict[str, Any]:
    filled = 0
    partial = 0
    rejects = 0
    cancels = 0
    is_bps_vals: list[float] = []
    latency_vals: list[float] = []

    for row in rows:
        status = str(row.get("status", "")).lower()
        if status in {"filled", "partially_filled"}:
            filled += 1
        if status == "partially_filled":
            partial += 1
        if status == "rejected":
            rejects += 1
        if status == "canceled":
            cancels += 1

        is_bps = _as_float(row.get("is_bps"))
        if is_bps is not None:
            is_bps_vals.append(is_bps)

        latency_ms = _as_float(row.get("fill_latency_ms"))
        if latency_ms is not None:
            latency_vals.append(latency_ms)

    edges_bps = [-value for value in is_bps_vals]
    expectancy_bps = float(mean(edges_bps)) if edges_bps else None
    max_dd_bps = _max_drawdown_bps(edges_bps) if edges_bps else None
    pacing_hits = _count_pacing_hits(service, window)

    return {
        "arm": arm,
        "start_utc": _utc_iso(window.start),
        "end_utc": _utc_iso(window.end),
        "records": len(rows),
        "net_expectancy_bps": expectancy_bps,
        "max_drawdown_bps": max_dd_bps,
        "pacing_cap_hits": pacing_hits,
        "fill_quality": {
            "fill_rate": float(filled) / float(len(rows)) if rows else None,
            "partial_fill_rate": float(partial) / float(len(rows)) if rows else None,
            "reject_rate": float(rejects) / float(len(rows)) if rows else None,
            "cancel_rate": float(cancels) / float(len(rows)) if rows else None,
            "mean_fill_latency_ms": float(mean(latency_vals)) if latency_vals else None,
            "p90_is_bps": _percentile(is_bps_vals, 90.0),
            "p95_is_bps": _percentile(is_bps_vals, 95.0),
        },
    }


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def _compare(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    bq = baseline.get("fill_quality", {})
    vq = variant.get("fill_quality", {})
    return {
        "expectancy_bps_delta_variant_minus_baseline": _delta(
            baseline.get("net_expectancy_bps"),
            variant.get("net_expectancy_bps"),
        ),
        "max_drawdown_bps_delta_variant_minus_baseline": _delta(
            baseline.get("max_drawdown_bps"),
            variant.get("max_drawdown_bps"),
        ),
        "fill_rate_delta_variant_minus_baseline": _delta(
            bq.get("fill_rate"),
            vq.get("fill_rate"),
        ),
        "mean_fill_latency_ms_delta_variant_minus_baseline": _delta(
            bq.get("mean_fill_latency_ms"),
            vq.get("mean_fill_latency_ms"),
        ),
        "p95_is_bps_delta_variant_minus_baseline": _delta(
            bq.get("p95_is_bps"),
            vq.get("p95_is_bps"),
        ),
        "pacing_cap_hits_delta_variant_minus_baseline": _delta(
            baseline.get("pacing_cap_hits"),
            variant.get("pacing_cap_hits"),
        ),
    }


def _window_from_state(state_dir: Path, arm: str) -> Window:
    start_path = _arm_stamp_path(state_dir, arm, "start")
    end_path = _arm_stamp_path(state_dir, arm, "end")
    if not start_path.exists():
        raise FileNotFoundError(f"Missing {start_path}")
    if not end_path.exists():
        raise FileNotFoundError(f"Missing {end_path}")
    return Window(
        arm=arm,
        start=_read_stamp(start_path),
        end=_read_stamp(end_path),
    )


def _cmd_arm(args: argparse.Namespace) -> int:
    env_path = Path(args.env_file)
    if not env_path.exists():
        raise FileNotFoundError(f"Missing env file: {env_path}")
    backup = _backup_env(env_path)
    _upsert_env(env_path, "EXECUTION_MAX_NEW_ORDERS_PER_CYCLE", str(args.max_new_orders))
    _write_stamp(
        _arm_stamp_path(Path(args.state_dir), args.arm, "start"),
        _now_utc(),
    )
    if not args.no_restart:
        _restart_service(args.service)
    print(
        json.dumps(
            {
                "status": "armed",
                "arm": args.arm,
                "max_new_orders_per_cycle": args.max_new_orders,
                "env_file": str(env_path),
                "env_backup": str(backup),
                "service_restarted": bool(not args.no_restart),
                "start_stamp": str(_arm_stamp_path(Path(args.state_dir), args.arm, "start")),
            },
            indent=2,
        )
    )
    return 0


def _cmd_close(args: argparse.Namespace) -> int:
    stamp = _arm_stamp_path(Path(args.state_dir), args.arm, "end")
    _write_stamp(stamp, _now_utc())
    print(
        json.dumps(
            {
                "status": "closed",
                "arm": args.arm,
                "end_stamp": str(stamp),
            },
            indent=2,
        )
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    env_values = _read_env_values(Path(args.env_file))
    tca_path = Path(
        args.tca_path
        or env_values.get("AI_TRADING_TCA_PATH")
        or str(DEFAULT_TCA_PATH)
    )

    baseline_window = _window_from_state(Path(args.state_dir), "baseline")
    variant_window = _window_from_state(Path(args.state_dir), "variant")

    tca_rows = _read_jsonl(tca_path)
    baseline_rows = _rows_in_window(tca_rows, baseline_window)
    variant_rows = _rows_in_window(tca_rows, variant_window)

    baseline = _summarize_arm(
        "baseline",
        baseline_rows,
        baseline_window,
        service=args.service,
    )
    variant = _summarize_arm(
        "variant",
        variant_rows,
        variant_window,
        service=args.service,
    )

    report = {
        "generated_at_utc": _utc_iso(_now_utc()),
        "tca_path": str(tca_path),
        "service": args.service,
        "baseline": baseline,
        "variant": variant,
        "delta": _compare(baseline, variant),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote report: {output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "A/B compare pacing cap settings using runtime TCA records. "
            "Run 'arm baseline' and 'arm variant' at matching market windows, "
            "close each arm, then run 'report'."
        )
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        help="Path to .env file (default: .env).",
    )
    parser.add_argument(
        "--state-dir",
        default=str(DEFAULT_STATE_DIR),
        help="Directory for baseline/variant start/end stamps.",
    )
    parser.add_argument(
        "--service",
        default=DEFAULT_SERVICE,
        help="Systemd service name for restart and log counting.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    arm = sub.add_parser("arm", help="Set pacing cap, stamp start time, and restart service.")
    arm.add_argument("--arm", choices=("baseline", "variant"), required=True)
    arm.add_argument("--max-new-orders", type=int, required=True)
    arm.add_argument(
        "--no-restart",
        action="store_true",
        help="Do not restart service after editing .env.",
    )
    arm.set_defaults(func=_cmd_arm)

    close = sub.add_parser("close", help="Stamp end time for an arm.")
    close.add_argument("--arm", choices=("baseline", "variant"), required=True)
    close.set_defaults(func=_cmd_close)

    report = sub.add_parser("report", help="Build A/B report from stamped windows.")
    report.add_argument(
        "--tca-path",
        default="",
        help="Optional explicit TCA JSONL path. Defaults to AI_TRADING_TCA_PATH from env.",
    )
    report.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON path (default: runtime/ab_compare.json).",
    )
    report.set_defaults(func=_cmd_report)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
