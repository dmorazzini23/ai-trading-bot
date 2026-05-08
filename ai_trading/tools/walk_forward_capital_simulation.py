"""Build a research-only walk-forward capital simulation artifact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai_trading.config.launch_profiles import launch_profile_payload, resolve_launch_profile
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _parse_ts(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _read_jsonl(path: Path | None, *, max_rows: int = 250_000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "path": str(path) if path is not None else None,
        "exists": bool(path is not None and path.exists()),
        "rows_read": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
    }
    if path is None or not path.exists():
        return [], stats
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        stats["read_error"] = True
        return [], stats
    with handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            stats["rows_read"] = int(stats["rows_read"]) + 1
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                stats["invalid_rows"] = int(stats["invalid_rows"]) + 1
                continue
            if not isinstance(parsed, dict):
                stats["invalid_rows"] = int(stats["invalid_rows"]) + 1
                continue
            rows.append(parsed)
            stats["valid_rows"] = int(stats["valid_rows"]) + 1
            if len(rows) > max(1, int(max_rows)):
                rows.pop(0)
    return rows, stats


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _side(row: Mapping[str, Any]) -> str:
    token = str(row.get("side") or row.get("order_side") or "buy").strip().lower()
    if token in {"short", "sell_short", "sellshort"}:
        return "sell_short"
    if token in {"buy", "sell"}:
        return token
    return "unknown"


def _notional(row: Mapping[str, Any], *, default_notional: float) -> float:
    explicit = _first_float(row, "notional", "requested_notional", "order_notional", "gross_notional")
    if explicit is not None:
        return max(0.0, explicit)
    quantity = _first_float(row, "qty", "quantity", "shares")
    price = _first_float(row, "price", "fill_price", "decision_price")
    if quantity is not None and price is not None:
        return max(0.0, abs(quantity * price))
    return max(0.0, float(default_notional))


def _pnl(row: Mapping[str, Any], *, filled_notional: float, requested_notional: float) -> float:
    explicit = _first_float(row, "realized_pnl", "pnl", "net_pnl", "profit_loss")
    if explicit is not None:
        if requested_notional > 0.0 and filled_notional < requested_notional:
            return float(explicit) * (filled_notional / requested_notional)
        return float(explicit)
    return_bps = _first_float(
        row,
        "realized_return_bps",
        "return_bps",
        "realized_net_edge_bps",
        "net_edge_bps",
    )
    if return_bps is None:
        return 0.0
    return float(filled_notional) * (float(return_bps) / 10_000.0)


def _percentile(values: Iterable[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    raw = max(0.0, min(float(q), 1.0)) * (len(clean) - 1)
    lo = int(math.floor(raw))
    hi = int(math.ceil(raw))
    if lo == hi:
        return clean[lo]
    return float(clean[lo] + ((clean[hi] - clean[lo]) * (raw - lo)))


def _max_drawdown(path: list[dict[str, Any]]) -> dict[str, Any]:
    peak = None
    max_abs = 0.0
    max_pct = 0.0
    trough_at = None
    for point in path:
        capital = float(point["capital"])
        if peak is None or capital > peak:
            peak = capital
        drawdown = max(0.0, float(peak) - capital)
        pct = 0.0 if not peak else drawdown / float(peak)
        if drawdown > max_abs:
            max_abs = drawdown
            max_pct = pct
            trough_at = point.get("ts")
    return {"amount": max_abs, "pct": max_pct, "trough_at": trough_at}


def build_walk_forward_capital_simulation(
    *,
    rows: Sequence[Mapping[str, Any]],
    initial_capital: float = 100_000.0,
    default_order_notional: float = 1_000.0,
    launch_profile_name: str | None = None,
    max_rows: int = 250_000,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Estimate a constrained capital path without enabling live trading."""

    profile = resolve_launch_profile(launch_profile_name)
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    capital = max(0.0, float(initial_capital))
    starting_capital = capital
    accepted = 0
    blocked = 0
    scaled = 0
    path: list[dict[str, Any]] = [
        {"ts": generated.isoformat().replace("+00:00", "Z"), "capital": capital, "event": "start"}
    ]
    blocked_reasons: Counter[str] = Counter()
    usage_values: list[float] = []
    daily_orders: defaultdict[date, int] = defaultdict(int)
    daily_pnl: defaultdict[date, float] = defaultdict(float)
    symbol_usage: defaultdict[tuple[date, str], float] = defaultdict(float)
    gross_usage: defaultdict[date, float] = defaultdict(float)
    decisions = []

    ordered_rows = sorted(
        rows[-max(1, int(max_rows)) :],
        key=lambda row: _parse_ts(row.get("ts") or row.get("timestamp") or row.get("decision_ts"))
        or datetime.min.replace(tzinfo=UTC),
    )
    for index, row in enumerate(ordered_rows):
        ts = _parse_ts(row.get("ts") or row.get("timestamp") or row.get("decision_ts")) or generated
        session_day = ts.date()
        symbol = _symbol(row)
        side = _side(row)
        requested_notional = _notional(row, default_notional=default_order_notional)
        reasons: list[str] = []
        if requested_notional <= 0.0:
            reasons.append("non_positive_notional")
        if profile.allowed_symbols and symbol not in profile.allowed_symbols:
            reasons.append("symbol_not_allowed_by_launch_profile")
        if side == "sell_short" and not profile.shorts_allowed:
            reasons.append("shorts_not_allowed_by_launch_profile")
        if daily_orders[session_day] >= int(profile.max_order_count):
            reasons.append("daily_order_count_limit")
        if profile.max_daily_loss is not None and daily_pnl[session_day] <= -float(profile.max_daily_loss):
            reasons.append("daily_loss_limit_reached")

        gross_cap = max(0.0, capital * float(profile.max_gross_exposure))
        symbol_cap = max(0.0, capital * float(profile.max_symbol_exposure))
        gross_remaining = max(0.0, gross_cap - gross_usage[session_day])
        symbol_remaining = max(0.0, symbol_cap - symbol_usage[(session_day, symbol)])
        size_cap = min(gross_remaining, symbol_remaining)
        if profile.max_notional_per_order is not None:
            size_cap = min(size_cap, max(0.0, float(profile.max_notional_per_order)))
        filled_notional = min(requested_notional, size_cap)
        if not reasons and filled_notional <= 0.0:
            reasons.append("capital_usage_limit")

        if reasons:
            blocked += 1
            blocked_reasons.update(reasons)
            decisions.append(
                {
                    "index": index,
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "symbol": symbol,
                    "side": side,
                    "requested_notional": requested_notional,
                    "accepted": False,
                    "blocked_reasons": reasons,
                }
            )
            continue

        pnl = _pnl(row, filled_notional=filled_notional, requested_notional=requested_notional)
        capital = max(0.0, capital + pnl)
        accepted += 1
        daily_orders[session_day] += 1
        daily_pnl[session_day] += pnl
        gross_usage[session_day] += filled_notional
        symbol_usage[(session_day, symbol)] += filled_notional
        usage = 0.0 if starting_capital <= 0.0 else filled_notional / starting_capital
        usage_values.append(usage)
        if filled_notional < requested_notional:
            scaled += 1
        path.append(
            {
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "capital": capital,
                "pnl": pnl,
                "notional": filled_notional,
                "symbol": symbol,
                "side": side,
            }
        )
        decisions.append(
            {
                "index": index,
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "symbol": symbol,
                "side": side,
                "requested_notional": requested_notional,
                "filled_notional": filled_notional,
                "accepted": True,
                "pnl": pnl,
                "scaled_by_constraints": filled_notional < requested_notional,
            }
        )

    drawdown = _max_drawdown(path)
    return {
        "schema_version": "1.0.0",
        "artifact_type": "walk_forward_capital_simulation",
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "mode": "research_shadow",
        "live_enabled": False,
        "status": "completed" if accepted else "no_accepted_orders",
        "launch_profile": launch_profile_payload(profile),
        "input": {
            "row_count": len(rows),
            "simulated_row_count": len(ordered_rows),
            "initial_capital": starting_capital,
            "default_order_notional": float(default_order_notional),
        },
        "summary": {
            "starting_capital": starting_capital,
            "ending_capital": capital,
            "net_pnl": capital - starting_capital,
            "accepted_orders": accepted,
            "blocked_orders": blocked,
            "scaled_orders": scaled,
            "max_drawdown": drawdown,
            "max_capital_usage_pct": max(usage_values) if usage_values else 0.0,
            "p90_capital_usage_pct": _percentile(usage_values, 0.90) or 0.0,
        },
        "constraints": {
            "blocked_reason_counts": dict(sorted(blocked_reasons.items())),
            "daily_order_counts": {day.isoformat(): count for day, count in sorted(daily_orders.items())},
            "daily_pnl": {day.isoformat(): pnl for day, pnl in sorted(daily_pnl.items())},
        },
        "capital_path": path,
        "decisions": decisions,
        "operator_note": "Research artifact only; it does not enable or approve live capital.",
    }


def _default_output() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/walk_forward_capital_simulation_latest.json",
        default_relative="runtime/walk_forward_capital_simulation_latest.json",
        for_write=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-jsonl", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--default-order-notional", type=float, default=1_000.0)
    parser.add_argument("--launch-profile", default=None)
    parser.add_argument("--max-rows", type=int, default=250_000)
    args = parser.parse_args(argv)

    rows, diagnostics = _read_jsonl(args.events_jsonl, max_rows=max(1, int(args.max_rows)))
    report = build_walk_forward_capital_simulation(
        rows=rows,
        initial_capital=max(0.0, float(args.initial_capital)),
        default_order_notional=max(0.0, float(args.default_order_notional)),
        launch_profile_name=args.launch_profile,
        max_rows=max(1, int(args.max_rows)),
    )
    report["sources"] = {"events_jsonl": diagnostics}
    output = args.output_json or _default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output), "status": report["status"]}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
