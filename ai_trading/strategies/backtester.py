from __future__ import annotations

import glob
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

from ai_trading.logging import get_logger

import pandas as pd

import config
from ai_trading import signals  # noqa: F401
# AI-AGENT-REF: Removed legacy trade_execution import as part of shim cleanup
# from ai_trading import trade_execution as execution_api  # type: ignore
import risk_engine
from ai_trading.core import bot_engine

logger = get_logger(__name__)

@dataclass
class Order:
    symbol: str
    qty: int
    side: str  # 'buy' or 'sell'
    price: float


@dataclass
class Fill:
    order: Order
    fill_price: float
    timestamp: datetime
    commission: float = 0.0


class ExecutionModel(ABC):
    """Abstract execution model interface."""

    @abstractmethod
    def on_order(self, order: Order) -> List[Fill]:
        """Handle an order and return resulting fills."""

    def on_bar(self) -> List[Fill]:
        """Advance one bar and release pending fills."""
        return []


class ImmediateExecutionModel(ExecutionModel):
    """Fill orders immediately at the order price."""

    def on_order(self, order: Order) -> List[Fill]:
        return [
            Fill(
                order=order,
                fill_price=order.price,
                timestamp=datetime.now(timezone.utc),
            )
        ]


class CommissionModel(ExecutionModel):
    def __init__(self, per_share_fee: float, inner: ExecutionModel) -> None:
        self.per_share_fee = per_share_fee
        self.inner = inner

    def on_order(self, order: Order) -> List[Fill]:
        fills = self.inner.on_order(order)
        for f in fills:
            f.commission += self.per_share_fee * order.qty
        return fills

    def on_bar(self) -> List[Fill]:
        return self.inner.on_bar()


class SlippageModel(ExecutionModel):
    def __init__(self, pips: float, inner: ExecutionModel) -> None:
        self.pips = pips
        self.inner = inner

    def on_order(self, order: Order) -> List[Fill]:
        fills = self.inner.on_order(order)
        adj = self.pips if order.side.lower() == "buy" else -self.pips
        for f in fills:
            f.fill_price += adj
        return fills

    def on_bar(self) -> List[Fill]:
        return self.inner.on_bar()


class LatencyModel(ExecutionModel):
    def __init__(self, bar_delay: int, inner: ExecutionModel) -> None:
        self.bar_delay = bar_delay
        self.inner = inner
        self._queue: List[tuple[int, Fill]] = []

    def on_order(self, order: Order) -> List[Fill]:
        fills = self.inner.on_order(order)
        for f in fills:
            self._queue.append((self.bar_delay, f))
        return []

    def on_bar(self) -> List[Fill]:
        ready: List[Fill] = []
        new_q: List[tuple[int, Fill]] = []
        for delay, f in self._queue:
            if delay <= 0:
                ready.append(f)
            else:
                new_q.append((delay - 1, f))
        self._queue = new_q
        return ready


class DefaultExecutionModel(ExecutionModel):
    """Default composition: commission → slippage → latency."""

    def __init__(self, per_share_fee: float = 0.0, slippage_pips: float = 0.0, latency: int = 0) -> None:
        base = ImmediateExecutionModel()
        base = CommissionModel(per_share_fee, base)
        base = SlippageModel(slippage_pips, base)
        self.model = LatencyModel(latency, base)

    def on_order(self, order: Order) -> List[Fill]:
        return self.model.on_order(order)

    def on_bar(self) -> List[Fill]:
        return self.model.on_bar()


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    net_pnl: float
    cagr: float
    max_drawdown: float
    sharpe: float
    calmar: float
    turnover: float


class BacktestEngine:
    """Historical simulator executing the live trading cycle."""

    def __init__(self, data: Dict[str, pd.DataFrame], execution_model: ExecutionModel, initial_cash: float = 100000.0) -> None:
        config.reload_env()
        self.data = data
        self.execution_model = execution_model
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, int] = {s: 0 for s in data}
        self.trades: List[Fill] = []
        self.equity_curve: List[Dict[str, float]] = []

    def reset(self) -> None:
        """Reset internal state for a new symbol run."""
        self.cash = self.initial_cash
        self.positions = {s: 0 for s in self.data}
        self.trades = []
        self.equity_curve = []

    def run_single_symbol(self, df: pd.DataFrame, risk: risk_engine.RiskEngine) -> BacktestResult:
        """Run the backtest for ``df`` using the live bot cycle."""
        self.data = {"symbol": df}
        self.positions = {"symbol": 0}
        self.reset()
        return self.run(["symbol"])

    def _apply_fill(self, fill: Fill, ts: pd.Timestamp) -> None:
        if hasattr(bot_engine, "apply_fill"):
            try:
                bot_engine.apply_fill(fill)
            except Exception:
                pass
        qty = fill.order.qty if fill.order.side.lower() == "buy" else -fill.order.qty
        cost = fill.fill_price * qty
        if qty > 0:
            self.cash -= cost + fill.commission
        else:
            self.cash += -cost - fill.commission
        self.positions[fill.order.symbol] += qty
        self.trades.append(fill)

    def _snapshot(self, ts: pd.Timestamp) -> None:
        pos_val = 0.0
        for sym, qty in self.positions.items():
            df = self.data.get(sym)
            if df is not None and ts in df.index:
                pos_val += qty * float(df.loc[ts, "close"])
        total = self.cash + pos_val
        self.equity_curve.append({
            "timestamp": ts,
            "cash": self.cash,
            "positions": pos_val,
            "total_equity": total,
        })

    def run(self, symbols: List[str]) -> BacktestResult:
        combined = sorted(set().union(*(df.index for df in self.data.values())))
        for ts in combined:
            for sym in symbols:
                df = self.data.get(sym)
                if df is not None and ts in df.index and hasattr(bot_engine, "update_market_data"):
                    try:
                        bot_engine.update_market_data(sym, df.loc[ts])
                    except Exception:
                        pass
            orders = []
            if hasattr(bot_engine, "next_cycle"):
                try:
                    orders = bot_engine.next_cycle()
                except Exception:
                    orders = []
            for order in orders:
                for fill in self.execution_model.on_order(order):
                    self._apply_fill(fill, ts)
            for fill in self.execution_model.on_bar():
                self._apply_fill(fill, ts)
            self._snapshot(ts)
        trades_df = pd.DataFrame([{
            "symbol": f.order.symbol,
            "qty": f.order.qty,
            "side": f.order.side,
            "price": f.fill_price,
            "timestamp": f.timestamp,
            "commission": f.commission,
        } for f in self.trades])
        eq_df = pd.DataFrame(self.equity_curve).set_index("timestamp")
        stats = self._stats(eq_df, trades_df)
        return BacktestResult(trades_df, eq_df, **stats)

    def _stats(self, equity: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, float]:
        if equity.empty:
            return {k: 0.0 for k in ["net_pnl", "cagr", "max_drawdown", "sharpe", "calmar", "turnover"]}
        net_pnl = float(equity["total_equity"].iloc[-1] - equity["total_equity"].iloc[0])
        returns = equity["total_equity"].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() else float("nan")
        duration_years = len(equity) / 252 if len(equity) else 0
        cagr = (equity["total_equity"].iloc[-1] / equity["total_equity"].iloc[0]) ** (1 / max(duration_years, 1e-9)) - 1
        drawdown = (equity["total_equity"] / equity["total_equity"].cummax() - 1).min()
        turnover = trades["qty"].abs().mul(trades["price"]).sum() / equity["total_equity"].iloc[0]
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

    parser = argparse.ArgumentParser(
        description="Full‑fidelity backtester (mirrors live bot)."
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
    args = parser.parse_args(argv)

    engine = BacktestEngine({}, DefaultExecutionModel(args.commission, args.slippage_pips, args.latency_bars))
    risk = risk_engine.RiskEngine()
    results: Dict[str, BacktestResult] = {}

    for symbol in args.symbols:
        # look for any CSV under data_dir matching SYMBOL*.csv (recursively)
        pattern = os.path.join(args.data_dir, "**", f"{symbol}*.csv")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            logger.warning(f"No CSV found for {symbol} in {args.data_dir}, skipping")
            continue
        csv_path = matches[0]
        logger.info(f"Loading backtest data from {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        df = df.loc[str(args.start) : str(args.end)]
        engine.data = {symbol: df}
        engine.reset()
        result = engine.run([symbol])
        results[symbol] = result

    if not results:
        logger.error("No valid symbols to backtest – please check your --data-dir")
        sys.exit(1)

    summary = [
        {
            "symbol": sym,
            "pnl": res.net_pnl,
            "sharpe": res.sharpe,
            "drawdown": res.max_drawdown,
        }
        for sym, res in results.items()
    ]
    summary_df = pd.DataFrame(summary)
    logger.info("\n%s", summary_df.to_string(index=False))
    summary_df.to_csv("backtest_summary.csv", index=False)

    trades_df = pd.concat(
        [res.trades.assign(symbol=sym) for sym, res in results.items()],
        ignore_index=True,
    )
    trades_df.to_csv("trades.csv", index=False)


if __name__ == "__main__":
    main()
