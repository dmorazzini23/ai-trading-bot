"""Legacy halted-position risk controls extracted from ``bot_engine.py``."""

from __future__ import annotations

import importlib
import logging
from json import JSONDecodeError
from typing import Any


def _bot_engine() -> Any:
    return importlib.import_module("ai_trading.core.bot_engine")


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed


def _numeric_helper(be: Any, name: str, symbol: str, *, default: float) -> float:
    helper = getattr(be.utils, name, None)
    if not callable(helper):
        return float(default)
    try:
        return _safe_float(helper(symbol), default=default)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        be.logger.warning(
            "POSITION_RISK_HELPER_FAILED",
            extra={"symbol": symbol, "helper": name},
        )
        return float(default)


def manage_position_risk_runtime(ctx: Any, position: Any) -> None:
    """Adjust trailing stops and halted-position sizing with fail-soft guards."""

    be = _bot_engine()
    symbol = str(getattr(position, "symbol", "") or "").strip().upper()
    if not symbol:
        return
    try:
        atr = max(0.0, _numeric_helper(be, "get_rolling_atr", symbol, default=0.0))
        try:
            price_df = be.fetch_minute_df_safe(symbol)
        except be.DataFetchError:
            be.logger.critical("No minute data for %s, skipping.", symbol)
            return
        if be.logger.isEnabledFor(logging.DEBUG):
            be.logger.debug(
                "Latest rows for %s: %s",
                symbol,
                price_df.tail(3).to_dict(orient="list"),
            )

        price = 0.0
        if "close" in price_df.columns:
            price_series = price_df["close"].dropna()
            if not price_series.empty:
                price = _safe_float(price_series.iloc[-1])
                be.logger.debug("Final extracted price for %s: %s", symbol, price)
            else:
                be.logger.critical("No valid close prices found for %s, skipping.", symbol)
        else:
            be.logger.critical("Close column missing for %s, skipping.", symbol)

        if price <= 0.0:
            be.logger.critical("Invalid price computed for %s: %s", symbol, price)
            return

        qty = int(_safe_float(getattr(position, "qty", 0.0)))
        if qty == 0:
            return
        side = "long" if qty > 0 else "short"
        avg_entry_price = max(0.0, _safe_float(getattr(position, "avg_entry_price", 0.0)))
        vwap = _numeric_helper(be, "get_current_vwap", symbol, default=price)
        if side == "long":
            new_stop = avg_entry_price * (1 - min(0.01 + atr / 100.0, 0.03))
        else:
            new_stop = avg_entry_price * (1 + min(0.01 + atr / 100.0, 0.03))
        be.update_trailing_stop(ctx, symbol, price, qty, atr)
        pnl = _safe_float(getattr(position, "unrealized_plpc", 0.0))
        kelly_scale = be.compute_kelly_scale(atr, 0.0)
        be.adjust_position_size(position, kelly_scale)
        volume_factor = _numeric_helper(
            be,
            "get_volume_spike_factor",
            symbol,
            default=0.0,
        )
        ml_conf = _numeric_helper(be, "get_ml_confidence", symbol, default=0.0)
        if (
            volume_factor > float(be.CFG.volume_spike_threshold)
            and ml_conf > float(be.CFG.ml_confidence_threshold)
            and side == "long"
            and price > float(vwap)
            and pnl > 0.02
        ):
            be.pyramid_add_position(
                ctx,
                symbol,
                float(be.CFG.pyramid_levels["low"]),
                side,
            )
        be.logger.info(
            "HALT_MANAGE %s stop=%.2f vwap=%.2f vol=%.2f ml=%.2f",
            symbol,
            new_stop,
            float(vwap),
            volume_factor,
            ml_conf,
        )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:
        be.logger.warning("manage_position_risk failed for %s: %s", symbol, exc)


__all__ = ["manage_position_risk_runtime"]
