"""Build daily ML shadow evaluation reports from runtime JSONL telemetry."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from ai_trading.data.historical_bars import load_historical_bars
from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _load_shadow_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "ML_SHADOW_REPORT_INVALID_JSONL_ROW",
                    extra={"path": str(path), "line_number": line_number},
                )
                continue
            if not isinstance(payload, Mapping):
                continue
            if str(payload.get("mode") or "ml_signal_shadow") != "ml_signal_shadow":
                continue
            rows.append(dict(payload))
    return rows


def _bool_value(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _load_bars_by_symbol(data_dir: Path, timestamp_col: str) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    if not data_dir.is_dir():
        return bars
    for csv_path in sorted(data_dir.glob("*.csv")):
        symbol = csv_path.stem.upper()
        try:
            frame, _report = load_historical_bars(csv_path, timestamp_col=timestamp_col)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "ML_SHADOW_REPORT_BAR_LOAD_FAILED",
                extra={"symbol": symbol, "path": str(csv_path), "error": str(exc)},
            )
            continue
        if isinstance(frame.index, pd.DatetimeIndex) and not frame.empty:
            bars[symbol] = frame.sort_index(kind="stable")
    return bars


def _row_timestamp(row: Mapping[str, Any]) -> pd.Timestamp | None:
    market = row.get("market")
    raw_ts: Any = None
    if isinstance(market, Mapping):
        raw_ts = market.get("bar_timestamp") or market.get("quote_timestamp")
    if raw_ts in (None, ""):
        raw_ts = row.get("ts")
    try:
        parsed = pd.to_datetime(raw_ts, errors="coerce", utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return cast(pd.Timestamp, parsed)


def _entry_close(row: Mapping[str, Any]) -> float | None:
    market = row.get("market")
    if isinstance(market, Mapping):
        return _finite_float(market.get("entry_close"))
    return None


def _net_markout_bps(
    row: Mapping[str, Any],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    horizon_bars: int,
    fee_bps: float,
    slippage_bps: float,
) -> float | None:
    symbol = str(row.get("symbol") or "").strip().upper()
    frame = bars_by_symbol.get(symbol)
    if frame is None or frame.empty:
        return None
    timestamp = _row_timestamp(row)
    if timestamp is None:
        return None
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        return None
    position = int(index.searchsorted(timestamp, side="left"))
    if position >= len(frame):
        return None
    future_position = position + max(1, int(horizon_bars))
    if future_position >= len(frame):
        return None
    entry = _entry_close(row)
    if entry is None or entry <= 0.0:
        entry = _finite_float(frame["close"].iloc[position])
    future = _finite_float(frame["close"].iloc[future_position])
    if entry is None or future is None or entry <= 0.0:
        return None
    gross = ((future / entry) - 1.0) * 10000.0
    costs = (2.0 * max(0.0, float(fee_bps))) + (2.0 * max(0.0, float(slippage_bps)))
    return float(gross - costs)


def _decision_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    champion_trade = sum(1 for row in rows if _bool_value(row, "champion_would_trade"))
    challenger_trade = sum(1 for row in rows if _bool_value(row, "challenger_would_trade"))
    both_trade = sum(
        1
        for row in rows
        if _bool_value(row, "champion_would_trade")
        and _bool_value(row, "challenger_would_trade")
    )
    champion_only = sum(
        1
        for row in rows
        if _bool_value(row, "champion_would_trade")
        and not _bool_value(row, "challenger_would_trade")
    )
    challenger_only = sum(
        1
        for row in rows
        if _bool_value(row, "challenger_would_trade")
        and not _bool_value(row, "champion_would_trade")
    )
    neither = total - both_trade - champion_only - challenger_only
    agreement = sum(
        1
        for row in rows
        if _bool_value(row, "champion_would_trade")
        == _bool_value(row, "challenger_would_trade")
    )
    deltas = [
        value
        for row in rows
        if (value := _finite_float(row.get("probability_delta"))) is not None
    ]
    spreads = [
        value
        for row in rows
        if isinstance(row.get("market"), Mapping)
        and (value := _finite_float(cast(Mapping[str, Any], row["market"]).get("spread_bps")))
        is not None
    ]
    skew_breaches = sum(
        1
        for row in rows
        if isinstance(row.get("skew"), Mapping)
        and bool(cast(Mapping[str, Any], row["skew"]).get("breached"))
    )
    return {
        "rows": total,
        "agreement_count": agreement,
        "agreement_rate": _rate(agreement, total),
        "champion_trade_count": champion_trade,
        "challenger_trade_count": challenger_trade,
        "both_trade_count": both_trade,
        "champion_only_count": champion_only,
        "challenger_only_count": challenger_only,
        "neither_trade_count": neither,
        "mean_probability_delta": _mean(deltas),
        "mean_spread_bps": _mean(spreads),
        "skew_breach_count": skew_breaches,
        "skew_breach_rate": _rate(skew_breaches, total),
    }


def _markout_summary(
    rows: list[dict[str, Any]],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    horizon_bars: int,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    champion_markouts: list[float] = []
    challenger_markouts: list[float] = []
    shadow_only_markouts: list[float] = []
    by_symbol: dict[str, list[float]] = {}
    for row in rows:
        markout = _net_markout_bps(
            row,
            bars_by_symbol,
            horizon_bars=horizon_bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        if markout is None:
            continue
        champion_would_trade = _bool_value(row, "champion_would_trade")
        challenger_would_trade = _bool_value(row, "challenger_would_trade")
        if champion_would_trade:
            champion_markouts.append(markout)
        if challenger_would_trade:
            challenger_markouts.append(markout)
            symbol = str(row.get("symbol") or "").strip().upper() or "UNKNOWN"
            by_symbol.setdefault(symbol, []).append(markout)
        if challenger_would_trade and not champion_would_trade:
            shadow_only_markouts.append(markout)
    symbol_rows = [
        {
            "symbol": symbol,
            "samples": len(values),
            "mean_net_markout_bps": _mean(values),
            "positive_rate": _rate(sum(1 for value in values if value > 0.0), len(values)),
        }
        for symbol, values in by_symbol.items()
    ]
    def _symbol_sort_key(item: Mapping[str, Any]) -> tuple[bool, float]:
        mean_value = _finite_float(item.get("mean_net_markout_bps"))
        return (mean_value is None, mean_value if mean_value is not None else -1e9)

    symbol_rows.sort(key=_symbol_sort_key, reverse=True)
    return {
        "horizon_bars": int(horizon_bars),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "champion_samples": len(champion_markouts),
        "champion_mean_net_markout_bps": _mean(champion_markouts),
        "champion_positive_rate": _rate(
            sum(1 for value in champion_markouts if value > 0.0),
            len(champion_markouts),
        ),
        "challenger_samples": len(challenger_markouts),
        "challenger_mean_net_markout_bps": _mean(challenger_markouts),
        "challenger_positive_rate": _rate(
            sum(1 for value in challenger_markouts if value > 0.0),
            len(challenger_markouts),
        ),
        "shadow_only_samples": len(shadow_only_markouts),
        "shadow_only_mean_net_markout_bps": _mean(shadow_only_markouts),
        "shadow_only_positive_rate": _rate(
            sum(1 for value in shadow_only_markouts if value > 0.0),
            len(shadow_only_markouts),
        ),
        "best_symbols": symbol_rows[:15],
        "worst_symbols": list(reversed(symbol_rows[-15:])),
    }


def build_shadow_report(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_jsonl)
    rows = _load_shadow_rows(input_path)
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    if args.data_dir:
        bars_by_symbol = _load_bars_by_symbol(Path(args.data_dir), str(args.timestamp_col))
    symbols = Counter(str(row.get("symbol") or "").strip().upper() for row in rows)
    report = {
        "schema_version": "1.0.0",
        "artifact_type": "ml_shadow_report",
        "generated_at": datetime.now(UTC).isoformat(),
        "input_jsonl": str(input_path),
        "decision_summary": _decision_summary(rows),
        "markout_summary": _markout_summary(
            rows,
            bars_by_symbol,
            horizon_bars=int(args.horizon_bars),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        if bars_by_symbol
        else None,
        "top_symbols_by_rows": [
            {"symbol": symbol, "rows": int(count)}
            for symbol, count in symbols.most_common(20)
            if symbol
        ],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "ML_SHADOW_REPORT_WRITTEN",
        extra={"path": str(output_path), "rows": int(len(rows))},
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an ML shadow evaluation report from runtime JSONL telemetry."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    build_shadow_report(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
