import logging
import os
import random
import warnings
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import metrics_logger

warnings.filterwarnings(
    "ignore",
    message=".*invalid escape sequence.*",
    category=SyntaxWarning,
    module="pandas_ta.*",
)

from strategies import TradeSignal
from utils import get_phase_logger

logger = get_phase_logger(__name__, "RISK_CHECK")

random.seed(42)
np.random.seed(42)
# AI-AGENT-REF: compatibility with pandas_ta expecting numpy.NaN constant
if not hasattr(np, "NaN"):
    np.NaN = np.nan


MAX_DRAWDOWN = 0.05

# Simple global risk state used by utility helpers
HARD_STOP = False
MAX_TRADES = 10
CURRENT_TRADES = 0


class RiskEngine:
    """Cross-strategy risk manager."""

    def __init__(self) -> None:
        self.global_limit = float(os.getenv("EQUITY_EXPOSURE_CAP", "2.5"))
        self.asset_limits: Dict[str, float] = {}
        self.strategy_limits: Dict[str, float] = {}
        self.exposure: Dict[str, float] = {}
        self.strategy_exposure: Dict[str, float] = {}
        self.hard_stop = False
        # AI-AGENT-REF: track returns/drawdown for adaptive exposure cap
        self._returns: list[float] = []
        self._drawdowns: list[float] = []
        self._last_portfolio_cap: float | None = None
        self._last_equity_cap: float | None = None
        # maximum acceptable drawdown (fraction between 0 and 1)
        try:
            self.max_drawdown_threshold = float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.1"))
        except Exception:
            self.max_drawdown_threshold = 0.1
        # cooldown period (in minutes) before trading resumes after a hard stop
        try:
            self.hard_stop_cooldown = float(os.getenv("HARD_STOP_COOLDOWN_MIN", "10"))
        except Exception:
            self.hard_stop_cooldown = 10.0
        # timestamp (epoch) until which hard stop remains active
        self._hard_stop_until: float | None = None

    def _dynamic_cap(
        self,
        asset_class: str,
        volatility: float | None = None,
        cash_ratio: float | None = None,
    ) -> float:
        """Return exposure cap for ``asset_class`` using adaptive rules."""
        base_cap = self.asset_limits.get(asset_class, self.global_limit)
        port_cap = self._adaptive_global_cap()
        vol = self._current_volatility()
        if (
            self._last_portfolio_cap is None
            or abs(self._last_portfolio_cap - port_cap) > 0.01
            or self._last_equity_cap is None
            or abs(self._last_equity_cap - base_cap) > 0.01
        ):
            logger.info(
                "Adaptive exposure caps: portfolio=%.1f, equity=%.1f (volatility=%.1f%%)",
                port_cap,
                base_cap,
                vol * 100,
            )
            self._last_portfolio_cap = port_cap
            self._last_equity_cap = base_cap
        return min(base_cap, port_cap)

    # AI-AGENT-REF: adaptive exposure cap based on 10-day volatility
    def _current_volatility(self) -> float:
        return float(np.std(self._returns[-10:])) if self._returns else 0.0

    def _adaptive_global_cap(self) -> float:
        vol = self._current_volatility()
        max_cap = float(os.getenv("PORTFOLIO_EXPOSURE_CAP", "2.5"))
        if vol < 0.015:
            cap = max_cap
        elif vol < 0.03:
            cap = max(2.0, max_cap - 0.5)
        else:
            cap = max(2.0, max_cap - 1.0)
        return cap

    def update_portfolio_metrics(
        self, returns: list[float] | None = None, drawdown: float | None = None
    ) -> None:
        if returns:
            self._returns.extend(list(returns))
        if drawdown is not None:
            self._drawdowns.append(float(drawdown))
            # evaluate drawdown against threshold
            self._check_drawdown_and_update_stop(float(drawdown))

    def refresh_positions(self, api) -> None:
        """Synchronize exposure with live positions."""
        try:
            positions = api.get_all_positions()
            logger.debug("Raw Alpaca positions: %s", positions)
            acct = api.get_account()
            equity = float(getattr(acct, "equity", 0) or 0)
            exposure: Dict[str, float] = {}
            for p in positions:
                asset = getattr(p, "asset_class", "equity")
                qty = float(getattr(p, "qty", 0) or 0)
                price = float(getattr(p, "avg_entry_price", 0) or 0)
                weight = abs(qty * price / equity) if equity > 0 else 0.0
                exposure[asset] = exposure.get(asset, 0.0) + weight
            self.exposure = exposure
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("refresh_positions failed: %s", exc)

    def position_exists(self, api, symbol: str) -> bool:
        """Return True if ``symbol`` exists in current Alpaca positions."""
        try:
            for p in api.get_all_positions():
                if getattr(p, "symbol", "") == symbol:
                    return True
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("position_exists failed for %s: %s", symbol, exc)
        return False

    def can_trade(
        self,
        signal: TradeSignal,
        *,
        pending: float = 0.0,
        volatility: float | None = None,
        cash_ratio: float | None = None,
        returns: list[float] | None = None,
        drawdowns: list[float] | None = None,
    ) -> bool:
        # incorporate latest metrics
        if returns:
            self._returns.extend(list(returns))
        if drawdowns:
            self._drawdowns.extend(list(drawdowns))
        # update hard stop status based on cooldown
        self._maybe_lift_hard_stop()
        if self.hard_stop:
            logger.error("TRADING_HALTED_RISK_LIMIT")
            return False
        if not isinstance(signal, TradeSignal):
            logger.error("can_trade called with invalid signal type")
            return False

        asset_exp = self.exposure.get(signal.asset_class, 0.0) + max(pending, 0.0)
        asset_cap = 1.1 * self._dynamic_cap(
            signal.asset_class, volatility, cash_ratio
        )  # slight relaxation to reduce unnecessary skips
        # apply risk scaling to the signal based on volatility and returns
        signal = self.apply_risk_scaling(signal, volatility=volatility, returns=returns)
        if asset_exp + signal.weight > asset_cap:
            logger.warning(
                "Exposure cap breach: symbol=%s qty=%s alloc=%.3f exposure=%.2f vs cap=%.2f",
                signal.symbol,
                getattr(signal, "qty", "n/a"),
                signal.weight,
                asset_exp + signal.weight,
                asset_cap,
            )
            if os.getenv("FORCE_CONTINUE_ON_EXPOSURE", "false").lower() != "true":
                return False
            logger.warning("FORCE_CONTINUE_ON_EXPOSURE enabled; overriding cap")

        strat_cap = self.strategy_limits.get(signal.strategy, self.global_limit)
        if signal.weight > strat_cap:
            logger.warning(
                "Strategy %s weight %.2f exceeds cap %.2f",
                signal.strategy,
                signal.weight,
                strat_cap,
            )
            if os.getenv("FORCE_CONTINUE_ON_EXPOSURE", "false").lower() != "true":
                return False
            logger.warning("FORCE_CONTINUE_ON_EXPOSURE enabled; overriding cap")
        return True

    def register_fill(self, signal: TradeSignal) -> None:
        if not isinstance(signal, TradeSignal):
            logger.error("register_fill called with invalid signal type")
            return

        prev = self.exposure.get(signal.asset_class, 0.0)
        delta = signal.weight if signal.side.lower() == "buy" else -signal.weight
        self.exposure[signal.asset_class] = prev + delta
        s_prev = self.strategy_exposure.get(signal.strategy, 0.0)
        self.strategy_exposure[signal.strategy] = s_prev + delta
        logger.info(
            "EXPOSURE_UPDATED",
            extra={
                "asset": signal.asset_class,
                "prev": prev,
                "new": self.exposure[signal.asset_class],
            },
        )

    # ------------------------------------------------------------------
    # Additional risk checks and scaling
    # ------------------------------------------------------------------
    def _check_drawdown_and_update_stop(self, current_drawdown: float) -> None:
        """
        Evaluate the latest drawdown and update the hard stop flag if the
        drawdown exceeds the threshold.  When triggered, trading is
        disabled until a cooldown period has elapsed.

        Parameters
        ----------
        current_drawdown : float
            The most recent drawdown measurement (0–1).
        """
        if current_drawdown >= self.max_drawdown_threshold and not self.hard_stop:
            self.hard_stop = True
            import time
            self._hard_stop_until = time.time() + self.hard_stop_cooldown * 60
            logger.error(
                "HARD_STOP_TRIGGERED",
                extra={"drawdown": current_drawdown, "threshold": self.max_drawdown_threshold},
            )

    def _maybe_lift_hard_stop(self) -> None:
        """
        Lift the hard stop if the cooldown period has expired.  This method
        should be called before evaluating new trades.
        """
        import time
        if self.hard_stop and self._hard_stop_until is not None:
            if time.time() >= self._hard_stop_until:
                self.hard_stop = False
                self._hard_stop_until = None
                logger.info("HARD_STOP_CLEARED")

    def apply_risk_scaling(
        self,
        signal: TradeSignal,
        *,
        volatility: float | None = None,
        returns: Sequence[float] | None = None,
    ) -> TradeSignal:
        """
        Adjust a signal's weight based on volatility and CVaR.  This function
        uses a simple inverse‑volatility rule and CVaR scaling to shrink
        exposures when markets become turbulent.  The original signal is
        mutated and returned for convenience.
        """
        try:
            scale = 1.0
            # apply inverse volatility scaling
            if volatility and volatility > 0:
                scale *= max(0.5, min(1.0, 0.02 / volatility))
            # apply CVaR scaling using recent returns
            if returns:
                import numpy as np
                from ai_trading.capital_scaling import cvar_scaling
                arr = np.asarray(list(returns), dtype=float)
                # ignore NaN/inf values
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    cvar_metric = cvar_scaling(arr, alpha=0.05)
                    # Only scale down when CVaR indicates heavy negative tail.
                    # cvar_scaling returns 1.0 for benign distributions and
                    # >1.0 for fat-tailed losses.  Apply shrinkage only
                    # when metric exceeds unity.
                    if cvar_metric > 1.0:
                        scale *= 1.0 / (1.0 + cvar_metric)
            signal.weight = max(0.0, signal.weight * scale)
            return signal
        except Exception as exc:
            logger.error("Risk scaling failed: %s", exc)
            return signal

    def check_max_drawdown(self, api) -> bool:
        try:
            account = api.get_account()
            pnl = float(account.equity) - float(account.last_equity)
            # Assume account returns equity strings convertible to float
            if pnl < -MAX_DRAWDOWN * float(account.last_equity):
                logger.error("HARD_STOP_MAX_DRAWDOWN", extra={"pnl": pnl})
                self.hard_stop = True
                return False
            return True
        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.error("check_max_drawdown failed: %s", exc)
            return False

    def position_size(
        self, signal: TradeSignal, cash: float, price: float, api=None
    ) -> int:
        logger.debug(
            "Pricing inputs for %s | cash: %.2f, price: %.2f, signal: %s",
            getattr(signal, "symbol", "N/A"),
            cash,
            price,
            signal,
        )
        if self.hard_stop:
            logger.error("HARD_STOP_ACTIVE")
            return 0
        if not isinstance(signal, TradeSignal):
            logger.error("position_size called with invalid signal type")
            return 0

        if api is not None and not self.check_max_drawdown(api):
            return 0

        if cash <= 0 or price <= 0:
            logger.warning(
                "Invalid cash %.2f or price %.2f for position sizing", cash, price
            )
            return 0

        if not self.can_trade(signal):
            asset_exp = self.exposure.get(signal.asset_class, 0.0)
            asset_cap = 1.1 * self._dynamic_cap(
                signal.asset_class
            )  # slight relaxation to reduce unnecessary skips
            projected = asset_exp + signal.weight
            if projected > asset_cap:
                qty_intended = int(round(cash * min(signal.weight, 1.0) / price))
                logger.warning(
                    "EXPOSURE_CAP_SKIP",
                    extra={
                        "symbol": signal.symbol,
                        "qty": qty_intended,
                        "allocation": signal.weight,
                        "exposure": projected,
                        "cap": asset_cap,
                    },
                )
            return 0

        weight = self._apply_weight_limits(signal)

        # Compute raw dollars allocated to this trade
        raw_dollars = cash * min(weight, 1.0)
        if np.isnan(raw_dollars) or np.isnan(price):
            logger.error("position_size received NaN inputs")
            return 0
        current_equity = cash
        if api is not None:
            try:
                acct = api.get_account()
                current_equity = float(getattr(acct, "equity", cash))
            except Exception:
                pass
        try:
            scaler = getattr(self, "capital_scaler", None)
            volatility = float(np.std(self._returns[-10:])) if self._returns else 0.0
            drawdown = abs(min(self._drawdowns)) if self._drawdowns else 0.0
            if scaler is not None:
                scaled_dollars = scaler.scale_position(
                    raw_dollars,
                    equity=current_equity,
                    volatility=volatility,
                    drawdown=drawdown,
                )
            else:
                scaled_dollars = raw_dollars
        except Exception as exc:
            logger.error("capital_scaler error: %s", exc)
            scaled_dollars = raw_dollars
        try:
            qty = int(round(scaled_dollars / price))
        except (ZeroDivisionError, OverflowError, TypeError, ValueError) as exc:
            logger.error("position_size division error: %s", exc)
            return 0
        return max(qty, 0)

    def _apply_weight_limits(self, sig: TradeSignal) -> float:
        """Return signal weight limited by remaining capacity."""
        # AI-AGENT-REF: use asset class key for exposure caps
        symbol = sig.asset_class
        strat = sig.strategy
        # compute how much capacity remains
        asset_cap = self.asset_limits.get(symbol, 1.0)
        strategy_cap = self.strategy_limits.get(strat, 1.0)
        used_asset = self.exposure.get(symbol, 0.0)
        used_strategy = self.strategy_exposure.get(strat, 0.0)
        max_asset = max(0.0, asset_cap - used_asset)
        max_strategy = max(0.0, strategy_cap - used_strategy)
        allowed = min(sig.weight, max_asset, max_strategy)
        return max(0.0, allowed)

    def compute_volatility(self, returns: np.ndarray) -> dict:
        if not isinstance(returns, np.ndarray) or returns.size == 0:
            logger.warning("Empty or invalid returns series—skipping risk computation")
            return {"volatility": 0.0}

        if np.isnan(returns).any() or np.isinf(returns).any():
            logger.error("Failed computing volatility: invalid values present")
            vol = 0.0
        else:
            try:
                vol = float(np.std(returns))
            except (ValueError, TypeError) as exc:
                logger.error("Failed computing volatility: %s", exc)
                vol = 0.0
        return {"volatility": vol}


def dynamic_position_size(capital: float, volatility: float, drawdown: float) -> float:
    """Return position size using volatility and drawdown aware Kelly fraction.

    The base Kelly fraction of ``0.5 / volatility`` is throttled by current
    drawdown. When drawdown exceeds 10% the fraction is scaled down by 50%.
    """

    if capital <= 0:
        return 0.0

    vol = max(volatility, 1e-6)
    kelly_fraction = 0.5 / vol
    # AI-AGENT-REF: clamp before applying drawdown adjustment
    kelly_fraction = min(max(kelly_fraction, 0.0), 1.0)
    if drawdown > 0.10:
        kelly_fraction *= 0.5
    return capital * kelly_fraction


def calculate_position_size(*args, **kwargs) -> int:
    """Convenience wrapper supporting simple and advanced usage."""
    engine = RiskEngine()
    if len(args) == 2 and not kwargs:
        cash, price = args
        dummy = TradeSignal(
            symbol="DUMMY", side="buy", confidence=1.0, strategy="default"
        )
        return engine.position_size(dummy, cash, price)
    if len(args) >= 3:
        signal, cash, price = args[:3]
        api = args[3] if len(args) > 3 else kwargs.get("api")
        return engine.position_size(signal, cash, price, api)
    raise TypeError("Invalid arguments for calculate_position_size")


def check_max_drawdown(state: Dict[str, float]) -> bool:
    """Return True if current drawdown exceeds the maximum allowed."""
    return state.get("current_drawdown", 0) > state.get("max_drawdown", 0)


def can_trade() -> bool:
    """Return False when trading should be halted."""
    return not HARD_STOP and CURRENT_TRADES < MAX_TRADES


def register_trade(size: int) -> dict | None:
    """Register a trade and increment the count if allowed."""
    global CURRENT_TRADES
    if not can_trade() or size <= 0:
        return None
    CURRENT_TRADES += 1
    return {"size": size}


def check_exposure_caps(portfolio, exposure, cap):
    for sym, pos in portfolio.positions.items():
        if pos.quantity > 0 and exposure[sym] > cap:
            logger.warning(
                "Exposure cap triggered, blocking new orders for %s", sym
            )
            return False
    # Original exposure logic continues here...


import pandas_ta as ta


def apply_trailing_atr_stop(
    df: pd.DataFrame,
    entry_price: float,
    *,
    ctx: Any | None = None,
    symbol: str = "SYMBOL",
    qty: int | None = None,
) -> None:
    """Exit ``qty`` at market if the trailing stop is triggered."""
    try:
        if entry_price <= 0:
            logger.warning(
                "apply_trailing_atr_stop invalid entry price: %.2f", entry_price
            )
            return
        atr = df.ta.atr()
        trailing_stop = entry_price - 2 * atr
        last_valid_close = df["Close"].dropna()
        if not last_valid_close.empty:
            price = last_valid_close.iloc[-1]
        else:
            logger.critical("All NaNs in close column for ATR stop")
            price = 0.0
        logger.debug("Latest 5 rows for ATR stop:\n%s", df.tail(5))
        logger.debug("Computed price for ATR stop: %s", price)
        if price <= 0 or pd.isna(price):
            logger.critical("Invalid price computed for ATR stop: %s", price)
            return
        if price < trailing_stop.iloc[-1]:
            logger.info(
                "ATR stop hit: price=%s vs stop=%s", price, trailing_stop.iloc[-1]
            )
            if ctx is not None and qty:
                try:
                    if hasattr(ctx, "risk_engine") and not ctx.risk_engine.position_exists(ctx.api, symbol):
                        logger.info("No position to sell for %s, skipping.", symbol)
                        return
                    from bot_engine import send_exit_order

                    send_exit_order(ctx, symbol, abs(int(qty)), price, "atr_stop")
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error("ATR stop exit failed: %s", exc)
            else:
                logger.warning("ATR stop triggered but no context/qty provided")
            schedule_reentry_check(symbol, lookahead_days=2)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("ATR stop error: %s", e)


def schedule_reentry_check(symbol: str, lookahead_days: int) -> None:
    """Log a re-entry check after a stop out."""
    logger.info("Scheduling reentry for %s in %s days", symbol, lookahead_days)


# AI-AGENT-REF: new risk helpers
def calculate_atr_stop(
    entry_price: float, atr: float, multiplier: float = 1.5, direction: str = "long"
) -> float:
    """Return ATR-based stop price."""
    stop = (
        entry_price - multiplier * atr
        if direction == "long"
        else entry_price + multiplier * atr
    )
    metrics_logger.log_atr_stop(symbol="generic", stop=stop)
    return stop


def calculate_bollinger_stop(
    price: float, upper_band: float, lower_band: float, direction: str = "long"
) -> float:
    """Return stop price using Bollinger band width."""
    mid = (upper_band + lower_band) / 2
    if direction == "long":
        stop = min(price, mid)
    else:
        stop = max(price, mid)
    metrics_logger.log_atr_stop(symbol="bb", stop=stop)
    return stop


def dynamic_stop_price(
    entry_price: float,
    atr: float | None = None,
    upper_band: float | None = None,
    lower_band: float | None = None,
    percent: float | None = None,
    direction: str = "long",
) -> float:
    """Return the tightest stop price based on ATR, Bollinger width or percent."""
    stops: list[float] = []
    if atr is not None:
        stops.append(calculate_atr_stop(entry_price, atr, direction=direction))
    if upper_band is not None and lower_band is not None:
        stops.append(
            calculate_bollinger_stop(
                entry_price, upper_band, lower_band, direction=direction
            )
        )
    if percent is not None:
        pct_stop = (
            entry_price * (1 - percent)
            if direction == "long"
            else entry_price * (1 + percent)
        )
        stops.append(pct_stop)
    if not stops:
        return entry_price
    return max(stops) if direction == "long" else min(stops)


# --- new risk management helpers -----------------------------------------

def compute_stop_levels(entry_price: float, atr: float, take_mult: float = 2.0) -> tuple[float, float]:
    """Return stop-loss and take-profit levels using ATR."""
    stop = entry_price - atr
    take = entry_price + take_mult * atr
    return stop, take


def correlation_position_weights(corr: pd.DataFrame, base: dict[str, float]) -> dict[str, float]:
    """Scale weights inversely proportional to asset correlations."""
    weights = {}
    for sym, w in base.items():
        if sym in corr.columns:
            c = corr[sym].abs().mean()
            scale = 1.0 / (1.0 + c)
            weights[sym] = w * scale
        else:
            weights[sym] = w
    return weights


def drawdown_circuit(drawdowns: Sequence[float], limit: float = 0.2) -> bool:
    """Return True if cumulative drawdown exceeds ``limit``."""
    dd = abs(min(0.0, min(drawdowns))) if drawdowns else 0.0
    return dd > limit


def volatility_filter(atr: float, sma: float, threshold: float = 0.05) -> bool:
    """Return True when volatility below ``threshold`` relative to SMA."""
    if sma == 0:
        return True
    return atr / sma < threshold
