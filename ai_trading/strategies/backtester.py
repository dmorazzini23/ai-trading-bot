from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import hashlib
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_trading import config
from ai_trading.data.historical_bars import (
    HistoricalBarLoadReport,
    filter_historical_bars_window,
    load_historical_bars,
)
from ai_trading.logging import get_logger
from ai_trading.oms.simulated_lifecycle import SimulatedLifecycleDriver

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


logger = get_logger(__name__)

BACKTEST_ARTIFACT_SCHEMA_VERSION = "1.0.0"
BACKTEST_MANIFEST_SCHEMA_VERSION = "1.0.0"


@dataclass
class Order:
    symbol: str
    qty: int
    side: str
    price: float
    intent_id: str | None = None
    idempotency_key: str | None = None


@dataclass
class Fill:
    order: Order
    fill_price: float
    timestamp: datetime
    commission: float = 0.0
    fill_id: str | None = None


class ExecutionModel(ABC):
    """Abstract execution model interface."""

    @abstractmethod
    def on_order(
        self,
        order: Order,
        *,
        timestamp: datetime | None = None,
    ) -> list[Fill]:
        """Handle an order and return resulting fills."""

    def on_bar(self, *, timestamp: datetime | None = None) -> list[Fill]:
        """Advance one bar and release pending fills."""

        _ = timestamp
        return []

    def reset(self) -> None:
        """Reset any internal state before starting a new deterministic run."""

        return None


class ImmediateExecutionModel(ExecutionModel):
    """Fill orders immediately at the order price."""

    def on_order(
        self,
        order: Order,
        *,
        timestamp: datetime | None = None,
    ) -> list[Fill]:
        fill_ts = timestamp if timestamp is not None else datetime.now(UTC)
        return [Fill(order=order, fill_price=order.price, timestamp=fill_ts)]


class CommissionModel(ExecutionModel):
    def __init__(self, per_share_fee: float, inner: ExecutionModel) -> None:
        self.per_share_fee = per_share_fee
        self.inner = inner

    def on_order(
        self,
        order: Order,
        *,
        timestamp: datetime | None = None,
    ) -> list[Fill]:
        fills = self.inner.on_order(order, timestamp=timestamp)
        for fill in fills:
            fill.commission += self.per_share_fee * order.qty
        return fills

    def on_bar(self, *, timestamp: datetime | None = None) -> list[Fill]:
        return self.inner.on_bar(timestamp=timestamp)

    def reset(self) -> None:
        self.inner.reset()


class SlippageModel(ExecutionModel):
    def __init__(self, pips: float, inner: ExecutionModel) -> None:
        self.pips = pips
        self.inner = inner

    def on_order(
        self,
        order: Order,
        *,
        timestamp: datetime | None = None,
    ) -> list[Fill]:
        fills = self.inner.on_order(order, timestamp=timestamp)
        adj = self.pips if order.side.lower() == "buy" else -self.pips
        for fill in fills:
            fill.fill_price += adj
        return fills

    def on_bar(self, *, timestamp: datetime | None = None) -> list[Fill]:
        return self.inner.on_bar(timestamp=timestamp)

    def reset(self) -> None:
        self.inner.reset()


class LatencyModel(ExecutionModel):
    def __init__(self, bar_delay: int, inner: ExecutionModel) -> None:
        self.bar_delay = bar_delay
        self.inner = inner
        self._queue: list[tuple[int, Fill]] = []

    def on_order(
        self,
        order: Order,
        *,
        timestamp: datetime | None = None,
    ) -> list[Fill]:
        fills = self.inner.on_order(order, timestamp=timestamp)
        for fill in fills:
            self._queue.append((self.bar_delay, fill))
        return []

    def on_bar(self, *, timestamp: datetime | None = None) -> list[Fill]:
        ready: list[Fill] = []
        new_queue: list[tuple[int, Fill]] = []
        for delay, fill in self._queue:
            if delay <= 0:
                if timestamp is not None:
                    fill.timestamp = timestamp
                ready.append(fill)
            else:
                new_queue.append((delay - 1, fill))
        self._queue = new_queue
        return ready

    def reset(self) -> None:
        self._queue = []
        self.inner.reset()


class DefaultExecutionModel(ExecutionModel):
    """Default composition: commission -> slippage -> latency."""

    def __init__(
        self,
        per_share_fee: float = 0.0,
        slippage_pips: float = 0.0,
        latency: int = 0,
    ) -> None:
        base: ExecutionModel = ImmediateExecutionModel()
        base = CommissionModel(per_share_fee, base)
        base = SlippageModel(slippage_pips, base)
        self.model = LatencyModel(latency, base)

    def on_order(
        self,
        order: Order,
        *,
        timestamp: datetime | None = None,
    ) -> list[Fill]:
        return self.model.on_order(order, timestamp=timestamp)

    def on_bar(self, *, timestamp: datetime | None = None) -> list[Fill]:
        return self.model.on_bar(timestamp=timestamp)

    def reset(self) -> None:
        self.model.reset()


@dataclass
class BacktestResult:
    trades: "pd.DataFrame"
    equity_curve: "pd.DataFrame"
    net_pnl: float
    cagr: float
    max_drawdown: float
    sharpe: float
    calmar: float
    turnover: float


def _coerce_timestamp_value(raw: Any) -> str:
    iso = getattr(raw, "isoformat", None)
    if callable(iso):
        try:
            return str(iso())
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            return str(raw)
    return str(raw)


def _resolve_output_path(path_value: str | Path) -> Path:
    target = Path(path_value).expanduser()
    if target.is_absolute():
        return target.resolve()
    return (Path.cwd() / target).resolve()


def _result_trade_records(result: BacktestResult, *, symbol: str) -> list[dict[str, Any]]:
    if getattr(result.trades, "empty", True):
        return []
    records: list[dict[str, Any]] = []
    for row in result.trades.to_dict("records"):
        payload = {str(key): value for key, value in dict(row).items()}
        payload["symbol"] = symbol
        if "timestamp" in payload:
            payload["timestamp"] = _coerce_timestamp_value(payload["timestamp"])
        records.append(payload)
    return records


def _summary_payload(
    *,
    results: dict[str, BacktestResult],
    data: dict[str, "pd.DataFrame"],
    load_reports: dict[str, HistoricalBarLoadReport],
    args: Any,
    summary_csv_path: Path,
    trades_csv_path: Path,
    summary_json_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_trades = 0
    total_bars = 0
    for symbol in sorted(results):
        result = results[symbol]
        frame = data[symbol]
        trade_count = int(len(result.trades.index)) if hasattr(result.trades, "index") else 0
        bar_count = int(len(frame.index)) if hasattr(frame, "index") else 0
        total_trades += trade_count
        total_bars += bar_count
        rows.append(
            {
                "symbol": symbol,
                "bars": bar_count,
                "trades": trade_count,
                "pnl": float(result.net_pnl),
                "net_pnl": float(result.net_pnl),
                "cagr": float(result.cagr),
                "sharpe": float(result.sharpe),
                "drawdown": float(result.max_drawdown),
                "turnover": float(result.turnover),
                "load_report": load_reports[symbol].as_dict(),
            }
        )
    return {
        "schema_version": BACKTEST_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "backtest_summary",
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "symbols_requested": [str(symbol).upper() for symbol in args.symbols],
            "symbols_loaded": [row["symbol"] for row in rows],
            "data_dir": str(args.data_dir),
            "timestamp_col": str(args.timestamp_col),
            "start": str(args.start),
            "end": str(args.end),
            "commission": float(args.commission),
            "slippage_pips": float(args.slippage_pips),
            "latency_bars": int(args.latency_bars),
            "initial_cash": float(args.initial_cash),
        },
        "aggregate": {
            "symbols": len(rows),
            "total_bars": total_bars,
            "total_trades": total_trades,
        },
        "symbols": rows,
        "inputs": {
            "symbols": {
                symbol: load_reports[symbol].as_dict()
                for symbol in sorted(load_reports)
            }
        },
        "artifacts": {
            "summary_csv": str(summary_csv_path),
            "trades_csv": str(trades_csv_path),
            "summary_json": str(summary_json_path),
            "manifest_json": str(manifest_path),
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest_payload(
    *,
    args: Any,
    load_reports: dict[str, HistoricalBarLoadReport],
    summary_csv_path: Path,
    trades_csv_path: Path,
    summary_json_path: Path,
) -> dict[str, Any]:
    source_files = {
        symbol: {
            **report.as_dict(),
            "sha256": _file_sha256(Path(report.path)),
        }
        for symbol, report in sorted(load_reports.items())
    }
    return {
        "schema_version": BACKTEST_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "backtest_run_manifest",
        "generated_at": datetime.now(UTC).isoformat(),
        "engine": {
            "name": "ai_trading.strategies.backtester",
            "strategy_id": "deterministic_local_signal_v1",
        },
        "config": {
            "symbols_requested": [str(symbol).upper() for symbol in args.symbols],
            "timestamp_col": str(args.timestamp_col),
            "start": str(args.start),
            "end": str(args.end),
            "commission": float(args.commission),
            "slippage_pips": float(args.slippage_pips),
            "latency_bars": int(args.latency_bars),
            "initial_cash": float(args.initial_cash),
        },
        "inputs": {
            "data_dir": str(args.data_dir),
            "source_files": source_files,
        },
        "outputs": {
            "summary_csv": {
                "path": str(summary_csv_path),
                "sha256": _file_sha256(summary_csv_path),
            },
            "trades_csv": {
                "path": str(trades_csv_path),
                "sha256": _file_sha256(trades_csv_path),
            },
            "summary_json": {
                "path": str(summary_json_path),
                "sha256": _file_sha256(summary_json_path),
            },
        },
    }


class BacktestEngine:
    """Historical simulator using deterministic local signal logic."""

    def __init__(
        self,
        data: dict[str, "pd.DataFrame"],
        execution_model: ExecutionModel,
        initial_cash: float = 100000.0,
    ) -> None:
        config.reload_env()
        self.data = data
        self.execution_model = execution_model
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, int] = dict.fromkeys(data, 0)
        self.trades: list[Fill] = []
        self.equity_curve: list[dict[str, float]] = []
        self._close_history: dict[str, list[float]] = {symbol: [] for symbol in data}
        self._oms_events_enabled = bool(
            config.get_env("AI_TRADING_BACKTEST_OMS_EVENTS_ENABLED", False, cast=bool)
        )
        self._oms_lifecycle = SimulatedLifecycleDriver(
            enabled=self._oms_events_enabled,
            source="backtest_engine",
        )
        self._event_counter = 0
        self._fill_counter = 0

    def reset(self) -> None:
        """Reset internal state for a new symbol run."""

        self.cash = self.initial_cash
        self.positions = dict.fromkeys(self.data, 0)
        self.trades = []
        self.equity_curve = []
        self._close_history = {symbol: [] for symbol in self.data}
        self.execution_model.reset()
        self._event_counter = 0
        self._fill_counter = 0

    @staticmethod
    def _hash_token(*parts: Any) -> str:
        material = "|".join(str(part) for part in parts if part not in (None, ""))
        if not material:
            material = "backtest-event"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _ts_text(value: Any) -> str:
        return _coerce_timestamp_value(value)

    def _emit_order_submit_lifecycle(self, order: Order, ts: Any) -> None:
        self._event_counter += 1
        token = self._hash_token(
            "backtest",
            order.symbol,
            order.side,
            order.qty,
            order.price,
            self._ts_text(ts),
            self._event_counter,
        )
        order.intent_id = f"bt-{token[:24]}"
        order.idempotency_key = token
        self._oms_lifecycle.open_submitted_intent(
            intent_id=order.intent_id,
            idempotency_key=token,
            symbol=order.symbol,
            side=order.side,
            quantity=float(order.qty),
            decision_ts=ts,
            broker_order_id=order.intent_id,
            strategy_id="backtest_engine",
            metadata={
                "price": float(order.price),
                "bar_ts": self._ts_text(ts),
            },
        )

    def _emit_fill_lifecycle(self, fill: Fill, ts: Any) -> None:
        intent_id = str(getattr(fill.order, "intent_id", "") or "").strip()
        if not intent_id:
            return
        self._fill_counter += 1
        fill.fill_id = str(fill.fill_id or f"{intent_id}-fill-{self._fill_counter}")
        self._oms_lifecycle.record_fill_and_close_intent(
            intent_id=intent_id,
            fill_qty=float(max(fill.order.qty, 0)),
            fill_price=float(fill.fill_price),
            fee=float(fill.commission),
            fill_ts=ts,
            terminal_status="FILLED",
            liquidity_flag="SIMULATED",
        )

    def _close_event_store(self) -> None:
        self._oms_lifecycle.close()

    def run_single_symbol(self, df: "pd.DataFrame", risk: Any) -> BacktestResult:
        """Run the backtest for ``df`` using local deterministic signals."""

        _ = risk
        self.data = {"symbol": df}
        self.positions = {"symbol": 0}
        self.reset()
        return self.run(["symbol"])

    def _apply_fill(self, fill: Fill, ts: "pd.Timestamp") -> None:
        qty = fill.order.qty if fill.order.side.lower() == "buy" else -fill.order.qty
        cost = fill.fill_price * qty
        if qty > 0:
            self.cash -= cost + fill.commission
        else:
            self.cash += -cost - fill.commission
        self.positions[fill.order.symbol] += qty
        self.trades.append(fill)
        self._emit_fill_lifecycle(fill, ts)

    @staticmethod
    def _close_at(df: "pd.DataFrame", ts: Any) -> float | None:
        if df is None or ts not in df.index or "close" not in df.columns:
            return None
        raw_value = df.loc[ts, "close"]
        if hasattr(raw_value, "iloc"):
            try:
                raw_value = raw_value.iloc[-1]
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        return value

    def _snapshot(self, ts: "pd.Timestamp") -> None:
        pos_val = 0.0
        for sym, qty in self.positions.items():
            df = self.data.get(sym)
            close = self._close_at(df, ts) if df is not None else None
            if close is not None:
                pos_val += qty * close
        total = self.cash + pos_val
        self.equity_curve.append(
            {
                "timestamp": ts,
                "cash": self.cash,
                "positions": pos_val,
                "total_equity": total,
            }
        )

    def _generate_orders_for_bar(self, symbol: str, close: float) -> list[Order]:
        history = self._close_history.setdefault(symbol, [])
        if len(history) < 5:
            history.append(close)
            return []

        short_mean = sum(history[-3:]) / 3.0
        long_window = history[-8:] if len(history) >= 8 else history
        long_mean = sum(long_window) / float(len(long_window))
        side: str | None = None
        if short_mean > long_mean * 1.001:
            side = "buy"
        elif short_mean < long_mean * 0.999:
            side = "sell"
        if side is None:
            history.append(close)
            return []

        current_position = int(self.positions.get(symbol, 0))
        if side == "sell" and current_position <= 0:
            history.append(close)
            return []
        if side == "buy" and self.cash < close:
            history.append(close)
            return []
        history.append(close)
        return [Order(symbol=symbol, qty=1, side=side, price=close)]

    def run(self, symbols: list[str]) -> BacktestResult:
        import pandas as pd  # heavy import; keep local

        trade_columns = ["symbol", "qty", "side", "price", "timestamp", "commission"]
        equity_columns = ["timestamp", "cash", "positions", "total_equity"]
        frames = [df for df in self.data.values() if df is not None and not getattr(df, "empty", True)]
        combined = sorted(set().union(*(df.index for df in frames)))
        if not combined:
            trades_df = pd.DataFrame(columns=trade_columns)
            eq_df = pd.DataFrame(columns=equity_columns).set_index("timestamp")
            stats = self._stats(eq_df, trades_df)
            self._close_event_store()
            return BacktestResult(trades_df, eq_df, **stats)
        for ts in combined:
            orders: list[Order] = []
            for sym in symbols:
                df = self.data.get(sym)
                if df is None:
                    continue
                close = self._close_at(df, ts)
                if close is None or close <= 0:
                    continue
                orders.extend(self._generate_orders_for_bar(sym, close))
            for order in orders:
                self._emit_order_submit_lifecycle(order, ts)
                for fill in self.execution_model.on_order(order, timestamp=ts):
                    self._apply_fill(fill, ts)
            for fill in self.execution_model.on_bar(timestamp=ts):
                self._apply_fill(fill, ts)
            self._snapshot(ts)

        trades_df = pd.DataFrame(
            [
                {
                    "symbol": fill.order.symbol,
                    "qty": fill.order.qty,
                    "side": fill.order.side,
                    "price": fill.fill_price,
                    "timestamp": fill.timestamp,
                    "commission": fill.commission,
                }
                for fill in self.trades
            ],
            columns=trade_columns,
        )
        eq_df = pd.DataFrame(self.equity_curve, columns=equity_columns).set_index("timestamp")
        stats = self._stats(eq_df, trades_df)
        self._close_event_store()
        return BacktestResult(trades_df, eq_df, **stats)

    def _stats(self, equity: "pd.DataFrame", trades: "pd.DataFrame") -> dict[str, float]:
        if equity.empty:
            return dict.fromkeys(
                ["net_pnl", "cagr", "max_drawdown", "sharpe", "calmar", "turnover"],
                0.0,
            )
        net_pnl = float(equity["total_equity"].iloc[-1] - equity["total_equity"].iloc[0])
        returns = equity["total_equity"].pct_change().dropna()
        sharpe = (
            returns.mean() / returns.std() * 252**0.5
            if returns.std()
            else float("nan")
        )
        duration_years = len(equity) / 252 if len(equity) else 0
        cagr = (
            equity["total_equity"].iloc[-1] / equity["total_equity"].iloc[0]
        ) ** (1 / max(duration_years, 1e-09)) - 1
        drawdown = (equity["total_equity"] / equity["total_equity"].cummax() - 1).min()
        turnover = 0.0
        if not trades.empty and {"qty", "price"}.issubset(set(trades.columns)):
            turnover = float(
                trades["qty"].abs().mul(trades["price"]).sum()
                / equity["total_equity"].iloc[0]
            )
        calmar = cagr / abs(drawdown) if drawdown else float("inf")
        return {
            "net_pnl": net_pnl,
            "cagr": cagr,
            "max_drawdown": abs(drawdown),
            "sharpe": sharpe,
            "calmar": calmar,
            "turnover": turnover,
        }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for running a backtest."""

    import argparse
    import pandas as pd  # heavy import; keep local

    parser = argparse.ArgumentParser(
        description="Deterministic local backtester for CSV OHLCV data.",
    )
    parser.add_argument(
        "-s",
        "--symbols",
        nargs="+",
        required=True,
        help="Tickers to backtest (e.g. AAPL MSFT GOOG).",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir",
        required=True,
        help="Directory containing <SYMBOL>.csv time series.",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column name. Falls back to the first parseable datetime column.",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission rate per trade (fraction).",
    )
    parser.add_argument(
        "--slippage-pips",
        type=float,
        default=0.0,
        help="Slippage in pips.",
    )
    parser.add_argument(
        "--latency-bars",
        type=int,
        default=0,
        help="Execution latency in bars.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Starting cash balance for the simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for backtest CSV artifacts.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON summary artifact path.",
    )
    args = parser.parse_args(argv)

    engine = BacktestEngine(
        {},
        DefaultExecutionModel(args.commission, args.slippage_pips, args.latency_bars),
        initial_cash=float(args.initial_cash),
    )
    results: dict[str, BacktestResult] = {}
    loaded_frames: dict[str, pd.DataFrame] = {}
    load_reports: dict[str, HistoricalBarLoadReport] = {}

    for raw_symbol in args.symbols:
        symbol = str(raw_symbol).upper()
        matches = sorted(Path(args.data_dir).expanduser().glob(f"**/{symbol}*.csv"))
        if not matches:
            logger.warning(
                "BACKTEST_SYMBOL_DATA_MISSING",
                extra={"symbol": symbol, "data_dir": str(args.data_dir)},
            )
            continue
        csv_path = matches[0]
        logger.info(
            "BACKTEST_SYMBOL_LOAD_START",
            extra={"symbol": symbol, "path": str(csv_path)},
        )
        try:
            frame, report = load_historical_bars(
                csv_path,
                timestamp_col=args.timestamp_col,
            )
            if report.invalid_timestamp_rows > 0:
                logger.warning(
                    "BACKTEST_INVALID_TIMESTAMPS_DROPPED",
                    extra={"path": str(csv_path), "rows": int(report.invalid_timestamp_rows)},
                )
            if report.duplicate_timestamp_rows > 0:
                logger.warning(
                    "BACKTEST_DUPLICATE_TIMESTAMPS_DEDUPED",
                    extra={"path": str(csv_path), "rows": int(report.duplicate_timestamp_rows)},
                )
            if report.non_positive_rows_dropped > 0:
                logger.warning(
                    "BACKTEST_NON_POSITIVE_ROWS_DROPPED",
                    extra={"path": str(csv_path), "rows": int(report.non_positive_rows_dropped)},
                )
            df = filter_historical_bars_window(
                frame,
                start=args.start,
                end=args.end,
            )
        except ValueError as exc:
            logger.warning(
                "BACKTEST_SYMBOL_SKIPPED",
                extra={"symbol": symbol, "path": str(csv_path), "error": str(exc)},
            )
            continue
        if df.empty:
            logger.warning(
                "BACKTEST_SYMBOL_WINDOW_EMPTY",
                extra={
                    "symbol": symbol,
                    "path": str(csv_path),
                    "start": str(args.start),
                    "end": str(args.end),
                },
            )
            continue
        loaded_frames[symbol] = df
        load_reports[symbol] = report
        engine.data = {symbol: df}
        engine.reset()
        result = engine.run([symbol])
        results[symbol] = result

    if not results:
        logger.error("No valid symbols to backtest – please check your --data-dir")
        sys.exit(1)

    output_dir = _resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "backtest_summary.csv"
    trades_csv_path = output_dir / "trades.csv"
    manifest_path = output_dir / "backtest_run_manifest.json"

    summary = [
        {
            "symbol": symbol,
            "bars": int(len(loaded_frames[symbol].index)),
            "trades": int(len(result.trades.index)) if hasattr(result.trades, "index") else 0,
            "pnl": result.net_pnl,
            "net_pnl": result.net_pnl,
            "cagr": result.cagr,
            "sharpe": result.sharpe,
            "drawdown": result.max_drawdown,
            "turnover": result.turnover,
        }
        for symbol, result in results.items()
    ]
    summary_df = pd.DataFrame(summary)
    legacy_log_summary = summary_df[["symbol", "pnl", "sharpe", "drawdown"]].copy()
    logger.info("\n%s", legacy_log_summary.to_string(index=False))
    summary_df.to_csv(summary_csv_path, index=False)

    trade_records: list[dict[str, Any]] = []
    for symbol, result in results.items():
        trade_records.extend(_result_trade_records(result, symbol=symbol))
    trades_df = pd.DataFrame(
        trade_records,
        columns=["symbol", "qty", "side", "price", "timestamp", "commission"],
    )
    trades_df.to_csv(trades_csv_path, index=False)

    output_json_path = (
        _resolve_output_path(args.output_json)
        if args.output_json is not None
        else output_dir / "backtest_summary.json"
    )
    payload = _summary_payload(
        results=results,
        data=loaded_frames,
        load_reports=load_reports,
        args=args,
        summary_csv_path=summary_csv_path,
        trades_csv_path=trades_csv_path,
        summary_json_path=output_json_path,
        manifest_path=manifest_path,
    )
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_payload = _build_manifest_payload(
        args=args,
        load_reports=load_reports,
        summary_csv_path=summary_csv_path,
        trades_csv_path=trades_csv_path,
        summary_json_path=output_json_path,
    )
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "BACKTEST_ARTIFACTS_WRITTEN",
        extra={
            "summary_csv": str(summary_csv_path),
            "trades_csv": str(trades_csv_path),
            "summary_json": str(output_json_path),
            "manifest_json": str(manifest_path),
        },
    )


if __name__ == "__main__":
    main()
