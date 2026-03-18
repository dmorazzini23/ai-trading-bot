"""Walk-forward validation runner backed by current research helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ai_trading.config.management import get_env
from ai_trading.config.runtime import TradingConfig
from ai_trading.core.bot_engine import DataFetcher
from ai_trading.logging import get_logger
from ai_trading.research.walk_forward import WalkForwardConfig, run_walk_forward

logger = get_logger(__name__)


def _normalize_daily_frame(frame: pd.DataFrame | None, symbol: str) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    normalized = frame.copy()
    if "timestamp" not in normalized.columns:
        if isinstance(normalized.index, pd.DatetimeIndex):
            normalized = normalized.copy()
            normalized.insert(0, "timestamp", normalized.index)
        else:
            return None
    if "close" not in normalized.columns:
        for alias in ("Close", "adj_close", "Adj Close"):
            if alias in normalized.columns:
                normalized = normalized.rename(columns={alias: "close"})
                break
    if "close" not in normalized.columns:
        return None
    normalized = normalized[["timestamp", "close"]].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", "close"])
    if normalized.empty:
        return None
    normalized["symbol"] = symbol
    return (
        normalized.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _score_fold(_train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    ordered = test_df.sort_values(["symbol", "timestamp"]).copy()
    returns = ordered.groupby("symbol")["close"].pct_change().dropna()
    if returns.empty:
        return {
            "post_cost_return": 0.0,
            "turnover": 0.0,
            "drawdown": 0.0,
            "hit_rate": 0.0,
        }
    cost_per_trade = 0.0002  # 2 bps per executed bar proxy
    post_cost_return = float(returns.sum() - (len(returns) * cost_per_trade))
    equity = (1.0 + returns).cumprod()
    drawdown = float((equity / equity.cummax() - 1.0).min())
    turnover = float(len(returns)) / float(max(len(ordered), 1))
    hit_rate = float((returns > 0).mean())
    return {
        "post_cost_return": post_cost_return,
        "turnover": turnover,
        "drawdown": drawdown,
        "hit_rate": hit_rate,
    }


def _build_walk_forward_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        train_days=int(get_env("AI_TRADING_WALK_FORWARD_TRAIN_DAYS", 180, cast=int)),
        test_days=int(get_env("AI_TRADING_WALK_FORWARD_TEST_DAYS", 30, cast=int)),
        step_days=int(get_env("AI_TRADING_WALK_FORWARD_STEP_DAYS", 30, cast=int)),
        embargo_days=int(get_env("AI_TRADING_WALK_FORWARD_EMBARGO_DAYS", 5, cast=int)),
        purge_days=int(get_env("AI_TRADING_WALK_FORWARD_PURGE_DAYS", 0, cast=int)),
    )


def _resolve_symbols(args: argparse.Namespace, config: TradingConfig) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if args.universe_file:
        path = Path(args.universe_file)
        if not path.exists():
            raise FileNotFoundError(f"Universe file not found: {path}")
        return [line.strip().upper() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]

    cfg_universe = getattr(config, "default_universe", None)
    if isinstance(cfg_universe, (list, tuple)):
        parsed = [str(sym).strip().upper() for sym in cfg_universe if str(sym).strip()]
        if parsed:
            return parsed

    env_universe = str(get_env("AI_TRADING_DEFAULT_UNIVERSE", "", cast=str) or "")
    if env_universe.strip():
        parsed = [s.strip().upper() for s in env_universe.split(",") if s.strip()]
        if parsed:
            return parsed

    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]


def run_walkforward_validation(symbols: list[str]) -> dict[str, Any]:
    logger.info("WFA_RUN_START", extra={"symbols": len(symbols)})
    fetcher = DataFetcher()
    symbol_frames: list[pd.DataFrame] = []
    loaded_symbols: list[str] = []
    for symbol in symbols:
        try:
            frame = fetcher.get_daily_df(None, symbol)
        except Exception as exc:  # pragma: no cover - defensive script guard
            logger.warning("WFA_FETCH_FAILED", extra={"symbol": symbol, "detail": str(exc)})
            continue
        normalized = _normalize_daily_frame(frame, symbol)
        if normalized is None or normalized.empty:
            logger.warning("WFA_FETCH_EMPTY", extra={"symbol": symbol})
            continue
        symbol_frames.append(normalized)
        loaded_symbols.append(symbol)

    if not symbol_frames:
        raise RuntimeError("No symbol data available for walk-forward validation")

    data = pd.concat(symbol_frames, ignore_index=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    wf_config = _build_walk_forward_config()
    result = run_walk_forward(data, score_fn=_score_fold, config=wf_config)
    result.update(
        {
            "symbols_requested": len(symbols),
            "symbols_loaded": len(loaded_symbols),
            "rows": int(len(data)),
            "loaded_symbols": loaded_symbols,
            "config": {
                "train_days": wf_config.train_days,
                "test_days": wf_config.test_days,
                "step_days": wf_config.step_days,
                "embargo_days": wf_config.embargo_days,
                "purge_days": wf_config.purge_days,
            },
        }
    )
    logger.info(
        "WFA_RUN_COMPLETE",
        extra={
            "fold_count": int(result.get("fold_count", 0) or 0),
            "symbols_loaded": len(loaded_symbols),
            "rows": int(len(data)),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols")
    parser.add_argument("--universe-file", type=str, help="Path to file containing one symbol per line")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without running")
    parser.add_argument("--json", action="store_true", help="Print JSON result payload")
    args = parser.parse_args()

    try:
        config = TradingConfig.from_env()
        symbols = _resolve_symbols(args, config)
        if not symbols:
            raise RuntimeError("No symbols resolved for walk-forward validation")
        logger.info("WFA_UNIVERSE", extra={"symbols": symbols})
        if args.dry_run:
            payload = {"status": "ok", "symbols": symbols}
            if args.json:
                print(json.dumps(payload, indent=2))
            return

        result = run_walkforward_validation(symbols)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return

        distribution = result.get("distribution", {})
        post_cost = distribution.get("post_cost_return", {})
        logger.info(
            "WFA_SUMMARY",
            extra={
                "fold_count": int(result.get("fold_count", 0) or 0),
                "symbols_loaded": int(result.get("symbols_loaded", 0) or 0),
                "post_cost_mean": float(post_cost.get("mean", 0.0) or 0.0),
            },
        )
    except KeyboardInterrupt:
        logger.info("WFA_INTERRUPTED")
        raise SystemExit(0)
    except Exception as exc:
        logger.error("WFA_FAILED", extra={"detail": str(exc)}, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
