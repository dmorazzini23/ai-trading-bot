from __future__ import annotations

import argparse
from collections import Counter
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.replay.event_loop import ReplayEventLoop

logger = get_logger(__name__)

_REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass(frozen=True)
class ReplayConfig:
    confidence_threshold: float
    entry_score_threshold: float
    allow_shorts: bool
    min_hold_bars: int
    max_hold_bars: int
    stop_loss_bps: float
    take_profit_bps: float
    trailing_stop_bps: float
    fee_bps: float
    slippage_bps: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline replay using local bars to evaluate churn and hold quality."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Path to a single OHLCV CSV file.")
    source.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing <SYMBOL>.csv files.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="",
        help="Explicit symbol name for --csv input. Defaults to CSV filename stem.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols to include when using --data-dir.",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column name. Falls back to first parseable datetime column.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.52)
    parser.add_argument("--entry-score-threshold", type=float, default=0.15)
    parser.add_argument(
        "--allow-shorts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow short entries during replay.",
    )
    parser.add_argument("--min-hold-bars", type=int, default=10)
    parser.add_argument("--max-hold-bars", type=int, default=120)
    parser.add_argument("--stop-loss-bps", type=float, default=60.0)
    parser.add_argument("--take-profit-bps", type=float, default=160.0)
    parser.add_argument("--trailing-stop-bps", type=float, default=90.0)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument(
        "--simulation-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run parity replay using async simulated broker events.",
    )
    parser.add_argument(
        "--replay-seed",
        type=int,
        default=42,
        help="Deterministic seed used by simulation replay mode.",
    )
    parser.add_argument(
        "--max-symbol-notional",
        type=float,
        default=None,
        help="Optional replay per-symbol notional cap for invariant checks.",
    )
    parser.add_argument(
        "--max-gross-notional",
        type=float,
        default=None,
        help="Optional replay gross notional cap for invariant checks.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def _load_frame(csv_path: Path, timestamp_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty")

    idx: pd.Index | None = None
    lower_map = {col.lower(): col for col in df.columns}
    ts_col = lower_map.get(timestamp_col.lower(), None)
    if ts_col is None:
        ts_col = lower_map.get("timestamp", None)

    if ts_col is not None:
        idx = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df = df.drop(columns=[ts_col])
    else:
        first_col = df.columns[0]
        first_series = df[first_col]
        if pd.api.types.is_numeric_dtype(first_series):
            idx = pd.RangeIndex(start=0, stop=len(df), step=1)
        else:
            candidate = pd.to_datetime(first_series, errors="coerce", utc=True)
            parse_ratio = float(candidate.notna().mean())
            if parse_ratio >= 0.95:
                idx = candidate
                df = df.drop(columns=[first_col])
            else:
                idx = pd.RangeIndex(start=0, stop=len(df), step=1)

    rename_map = {col: col.lower() for col in df.columns}
    df = df.rename(columns=rename_map)
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    out = df[list(_REQUIRED_COLUMNS) + ["volume"]].copy()
    out.index = idx
    for col in list(_REQUIRED_COLUMNS) + ["volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=list(_REQUIRED_COLUMNS))
    if out.empty:
        raise ValueError(f"{csv_path} has no valid OHLC rows after cleanup")
    return out.sort_index()


def _compute_signal(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    trend = ((ema_fast - ema_slow) / close.replace(0.0, np.nan)).fillna(0.0)
    momentum = close.pct_change(5).fillna(0.0)
    raw = (trend * 12.0) + (momentum * 32.0)
    score = np.tanh(raw).clip(-1.0, 1.0)
    confidence = (trend.abs() * 10.0 + momentum.abs() * 30.0).clip(0.0, 1.0)
    return score.astype(float), confidence.astype(float)


def _entry_price(close: float, side: int, slippage_bps: float) -> float:
    slip = slippage_bps / 10000.0
    if side > 0:
        return close * (1.0 + slip)
    return close * (1.0 - slip)


def _exit_price(close: float, side: int, slippage_bps: float) -> float:
    slip = slippage_bps / 10000.0
    if side > 0:
        return close * (1.0 - slip)
    return close * (1.0 + slip)


def _profit_factor(wins: np.ndarray, losses: np.ndarray) -> float | None:
    if losses.size == 0:
        if wins.size == 0:
            return 0.0
        return None
    return float(wins.sum() / abs(losses.sum()))


def _max_drawdown_bps(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    values = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(peaks > 0.0, values / peaks - 1.0, 0.0)
    return float(abs(drawdowns.min()) * 10000.0)


def _simulate_symbol(symbol: str, df: pd.DataFrame, cfg: ReplayConfig) -> dict[str, Any]:
    score, confidence = _compute_signal(df)
    trades: list[dict[str, Any]] = []

    side = 0
    entry_price = 0.0
    entry_bar = -1
    entry_ts: str | None = None
    best_price = 0.0
    position_bars = 0
    equity = 1.0
    equity_curve: list[float] = [equity]

    for i, (ts, row) in enumerate(df.iterrows()):
        close = float(row["close"])
        if close <= 0.0:
            continue
        s = float(score.iloc[i])
        conf = float(confidence.iloc[i])

        if side != 0:
            hold_bars = i - entry_bar
            position_bars += 1
            if side > 0:
                best_price = max(best_price, close)
                adverse_from_best_bps = (close / best_price - 1.0) * 10000.0
            else:
                best_price = min(best_price, close)
                adverse_from_best_bps = (best_price / close - 1.0) * 10000.0
            pnl_bps_live = ((close / entry_price) - 1.0) * 10000.0 * side

            exit_reason: str | None = None
            if pnl_bps_live <= -cfg.stop_loss_bps:
                exit_reason = "stop_loss"
            elif hold_bars >= cfg.max_hold_bars:
                exit_reason = "max_hold"
            elif hold_bars >= cfg.min_hold_bars and pnl_bps_live >= cfg.take_profit_bps:
                exit_reason = "take_profit"
            elif (
                hold_bars >= cfg.min_hold_bars
                and pnl_bps_live > 0.0
                and adverse_from_best_bps <= -cfg.trailing_stop_bps
            ):
                exit_reason = "trailing_stop"
            elif hold_bars >= cfg.min_hold_bars:
                long_flip = side > 0 and s <= -cfg.entry_score_threshold
                short_flip = side < 0 and s >= cfg.entry_score_threshold
                if (long_flip or short_flip) and conf >= cfg.confidence_threshold:
                    exit_reason = "signal_flip"

            if exit_reason is not None:
                fill_exit = _exit_price(close, side, cfg.slippage_bps)
                pnl_bps = ((fill_exit / entry_price) - 1.0) * 10000.0 * side
                pnl_bps -= 2.0 * cfg.fee_bps
                equity *= 1.0 + (pnl_bps / 10000.0)
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_ts": entry_ts,
                        "exit_ts": str(ts),
                        "side": "long" if side > 0 else "short",
                        "hold_bars": hold_bars,
                        "pnl_bps": float(pnl_bps),
                        "exit_reason": exit_reason,
                    }
                )
                side = 0
                entry_price = 0.0
                entry_bar = -1
                entry_ts = None
                best_price = 0.0

        if side == 0:
            open_long = conf >= cfg.confidence_threshold and s >= cfg.entry_score_threshold
            open_short = (
                cfg.allow_shorts
                and conf >= cfg.confidence_threshold
                and s <= -cfg.entry_score_threshold
            )
            if open_long:
                side = 1
            elif open_short:
                side = -1

            if side != 0:
                entry_price = _entry_price(close, side, cfg.slippage_bps)
                entry_bar = i
                entry_ts = str(ts)
                best_price = close

        equity_curve.append(equity)

    if side != 0 and entry_bar >= 0:
        close = float(df["close"].iloc[-1])
        fill_exit = _exit_price(close, side, cfg.slippage_bps)
        pnl_bps = ((fill_exit / entry_price) - 1.0) * 10000.0 * side
        pnl_bps -= 2.0 * cfg.fee_bps
        hold_bars = max(0, len(df) - 1 - entry_bar)
        equity *= 1.0 + (pnl_bps / 10000.0)
        trades.append(
            {
                "symbol": symbol,
                "entry_ts": entry_ts,
                "exit_ts": str(df.index[-1]),
                "side": "long" if side > 0 else "short",
                "hold_bars": hold_bars,
                "pnl_bps": float(pnl_bps),
                "exit_reason": "end_of_data",
            }
        )
        equity_curve.append(equity)

    pnl = np.asarray([float(t["pnl_bps"]) for t in trades], dtype=float)
    holds = np.asarray([float(t["hold_bars"]) for t in trades], dtype=float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    trade_count = int(pnl.size)

    summary: dict[str, Any] = {
        "symbol": symbol,
        "bars": int(len(df)),
        "trades": trade_count,
        "win_rate": float((wins.size / trade_count) if trade_count else 0.0),
        "avg_win_bps": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_bps": float(abs(losses.mean())) if losses.size else 0.0,
        "profit_factor": _profit_factor(wins, losses),
        "expectancy_bps": float(pnl.mean()) if trade_count else 0.0,
        "net_pnl_bps": float(pnl.sum()) if trade_count else 0.0,
        "median_hold_bars": float(np.median(holds)) if holds.size else 0.0,
        "churn_trades_per_100_bars": float((trade_count / len(df)) * 100.0),
        "exposure_ratio": float(position_bars / len(df)),
        "max_drawdown_bps": _max_drawdown_bps(equity_curve),
        "trades_detail": trades,
    }
    return summary


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    if args.csv is not None:
        symbol = args.symbol.strip().upper() if args.symbol else args.csv.stem.upper()
        return {symbol: args.csv}

    assert args.data_dir is not None
    chosen = {item.strip().upper() for item in args.symbols.split(",") if item.strip()}
    paths: dict[str, Path] = {}
    for csv_path in sorted(args.data_dir.glob("*.csv")):
        symbol = csv_path.stem.upper()
        if chosen and symbol not in chosen:
            continue
        paths[symbol] = csv_path
    if not paths:
        raise ValueError("No matching CSV files found for replay")
    return paths


def _normalize_bar_ts(value: Any, fallback_index: int) -> str:
    """Return UTC ISO timestamp for replay bars."""

    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.Timestamp("1970-01-01T00:00:00Z") + pd.Timedelta(minutes=fallback_index)
    return ts.isoformat()


def _run_parity_simulation(
    *,
    args: argparse.Namespace,
    cfg: ReplayConfig,
    symbol_paths: dict[str, Path],
) -> dict[str, Any]:
    """Run deterministic replay parity simulation with async fill events."""

    bars: list[dict[str, Any]] = []
    signal_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    per_symbol: list[dict[str, Any]] = []
    synthetic_index = 0
    for symbol, csv_path in symbol_paths.items():
        frame = _load_frame(csv_path, args.timestamp_col)
        score, confidence = _compute_signal(frame)
        per_symbol.append({"symbol": symbol, "bars": int(len(frame))})
        for pos, (ts, row) in enumerate(frame.iterrows()):
            close = float(row["close"])
            if not np.isfinite(close) or close <= 0.0:
                continue
            ts_iso = _normalize_bar_ts(ts, synthetic_index)
            synthetic_index += 1
            signal_lookup[(symbol, ts_iso)] = (
                float(score.iloc[pos]),
                float(confidence.iloc[pos]),
            )
            bars.append(
                {
                    "symbol": symbol,
                    "ts": ts_iso,
                    "close": close,
                }
            )

    def strategy(bar: dict[str, Any]) -> dict[str, Any] | None:
        symbol = str(bar.get("symbol", "")).upper()
        ts_iso = str(bar.get("ts", ""))
        score, confidence = signal_lookup.get((symbol, ts_iso), (0.0, 0.0))
        if confidence < cfg.confidence_threshold:
            return None
        side: str | None = None
        if score >= cfg.entry_score_threshold:
            side = "buy"
        elif cfg.allow_shorts and score <= -cfg.entry_score_threshold:
            side = "sell"
        if side is None:
            return None
        price = float(bar.get("close", 0.0) or 0.0)
        if not np.isfinite(price) or price <= 0.0:
            return None
        intent_key = f"{symbol}|{ts_iso}|{side}"
        return {
            "symbol": symbol,
            "side": side,
            "qty": 1.0,
            "type": "limit",
            "price": price,
            "limit_price": price,
            "intent_key": intent_key,
            "client_order_id": intent_key,
        }

    replay = ReplayEventLoop(
        strategy=strategy,
        seed=int(args.replay_seed),
        max_symbol_notional=args.max_symbol_notional,
        max_gross_notional=args.max_gross_notional,
    ).run(bars)

    events = list(replay.get("events", []))
    fill_events = [event for event in events if event.get("event_type") == "fill"]
    violations = list(replay.get("violations", []))
    violation_counts = Counter(
        str(item.get("code", "unknown"))
        for item in violations
    )
    total_bars = int(len(bars))
    total_trades = int(len(fill_events))
    aggregate: dict[str, Any] = {
        "simulation_mode": True,
        "replay_seed": int(args.replay_seed),
        "symbols": len(per_symbol),
        "total_bars": total_bars,
        "total_trades": total_trades,
        "win_rate": 0.0,
        "avg_win_bps": 0.0,
        "avg_loss_bps": 0.0,
        "profit_factor": 0.0,
        "expectancy_bps": 0.0,
        "net_pnl_bps": 0.0,
        "median_hold_bars": 0.0,
        "churn_trades_per_100_bars": float((total_trades / max(total_bars, 1)) * 100.0),
        "exposure_ratio": 0.0,
        "orders_submitted": int(len(replay.get("orders", []))),
        "intents_created": int(len(replay.get("intents", []))),
        "fill_events": total_trades,
        "violation_count": int(len(violations)),
        "violations_by_code": dict(sorted(violation_counts.items())),
        "config": {
            "confidence_threshold": cfg.confidence_threshold,
            "entry_score_threshold": cfg.entry_score_threshold,
            "allow_shorts": cfg.allow_shorts,
            "replay_seed": int(args.replay_seed),
            "max_symbol_notional": args.max_symbol_notional,
            "max_gross_notional": args.max_gross_notional,
        },
    }
    return {
        "aggregate": aggregate,
        "symbols": per_symbol,
        "replay": replay,
    }


def _run_replay(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ReplayConfig(
        confidence_threshold=float(args.confidence_threshold),
        entry_score_threshold=float(args.entry_score_threshold),
        allow_shorts=bool(args.allow_shorts),
        min_hold_bars=max(1, int(args.min_hold_bars)),
        max_hold_bars=max(2, int(args.max_hold_bars)),
        stop_loss_bps=max(1.0, float(args.stop_loss_bps)),
        take_profit_bps=max(1.0, float(args.take_profit_bps)),
        trailing_stop_bps=max(1.0, float(args.trailing_stop_bps)),
        fee_bps=max(0.0, float(args.fee_bps)),
        slippage_bps=max(0.0, float(args.slippage_bps)),
    )
    if cfg.max_hold_bars < cfg.min_hold_bars:
        raise ValueError("max-hold-bars must be >= min-hold-bars")

    symbol_paths = _resolve_inputs(args)
    if bool(args.simulation_mode):
        return _run_parity_simulation(args=args, cfg=cfg, symbol_paths=symbol_paths)

    per_symbol: list[dict[str, Any]] = []
    for symbol, csv_path in symbol_paths.items():
        frame = _load_frame(csv_path, args.timestamp_col)
        per_symbol.append(_simulate_symbol(symbol, frame, cfg))

    all_trades: list[dict[str, Any]] = []
    total_bars = 0
    total_position_bars = 0.0
    for item in per_symbol:
        all_trades.extend(item["trades_detail"])
        total_bars += int(item["bars"])
        total_position_bars += float(item["exposure_ratio"]) * float(item["bars"])

    pnl = np.asarray([float(t["pnl_bps"]) for t in all_trades], dtype=float)
    holds = np.asarray([float(t["hold_bars"]) for t in all_trades], dtype=float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    trade_count = int(pnl.size)
    aggregate: dict[str, Any] = {
        "symbols": len(per_symbol),
        "total_bars": total_bars,
        "total_trades": trade_count,
        "win_rate": float((wins.size / trade_count) if trade_count else 0.0),
        "avg_win_bps": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_bps": float(abs(losses.mean())) if losses.size else 0.0,
        "profit_factor": _profit_factor(wins, losses),
        "expectancy_bps": float(pnl.mean()) if trade_count else 0.0,
        "net_pnl_bps": float(pnl.sum()) if trade_count else 0.0,
        "median_hold_bars": float(np.median(holds)) if holds.size else 0.0,
        "churn_trades_per_100_bars": float((trade_count / max(total_bars, 1)) * 100.0),
        "exposure_ratio": float(total_position_bars / max(total_bars, 1)),
        "config": {
            "confidence_threshold": cfg.confidence_threshold,
            "entry_score_threshold": cfg.entry_score_threshold,
            "allow_shorts": cfg.allow_shorts,
            "min_hold_bars": cfg.min_hold_bars,
            "max_hold_bars": cfg.max_hold_bars,
            "stop_loss_bps": cfg.stop_loss_bps,
            "take_profit_bps": cfg.take_profit_bps,
            "trailing_stop_bps": cfg.trailing_stop_bps,
            "fee_bps": cfg.fee_bps,
            "slippage_bps": cfg.slippage_bps,
        },
    }
    return {"aggregate": aggregate, "symbols": per_symbol}


def run_replay(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run_replay(args)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        payload = _run_replay(args)
    except Exception as exc:
        logger.error("OFFLINE_REPLAY_FAILED", extra={"error": str(exc)}, exc_info=True)
        return 1

    aggregate = payload["aggregate"]
    logger.info(
        "OFFLINE_REPLAY_COMPLETE",
        extra={
            "symbols": aggregate["symbols"],
            "bars": aggregate["total_bars"],
            "trades": aggregate["total_trades"],
            "win_rate": aggregate["win_rate"],
            "profit_factor": aggregate["profit_factor"],
            "expectancy_bps": aggregate["expectancy_bps"],
            "median_hold_bars": aggregate["median_hold_bars"],
            "churn_100_bars": aggregate["churn_trades_per_100_bars"],
        },
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("OFFLINE_REPLAY_JSON_WRITTEN", extra={"path": str(args.output_json)})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
