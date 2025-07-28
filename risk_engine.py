import logging
import os
import random
import warnings
from typing import Any, Dict, Sequence
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import metrics_logger
import config

warnings.filterwarnings(
    "ignore",
    message=".*invalid escape sequence.*",
    category=SyntaxWarning,
    module="pandas_ta.*",
)

from strategies import TradeSignal
from utils import get_phase_logger

logger = get_phase_logger(__name__, "RISK_CHECK")

# Set deterministic seed from configuration
random.seed(config.SEED)
np.random.seed(config.SEED)
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

    def __init__(self, cfg: config.TradingConfig | None = None) -> None:
        """Initialize the engine with an optional trading config."""
        # AI-AGENT-REF: fix param shadowing bug when ``config`` is None
        self.config = cfg if cfg is not None else config.CONFIG
        
        # AI-AGENT-REF: Add comprehensive validation for risk parameters
        try:
            exposure_cap = getattr(self.config, 'exposure_cap_aggressive', 0.8)
            if not isinstance(exposure_cap, (int, float)) or not (0 < exposure_cap <= 1.0):
                logger.warning("Invalid exposure_cap_aggressive %s, using default 0.8", exposure_cap)
                exposure_cap = 0.8
            self.global_limit = exposure_cap
        except Exception as e:
            logger.error("Error validating exposure_cap_aggressive: %s, using default", e)
            self.global_limit = 0.8
        
        self.asset_limits: Dict[str, float] = {}
        self.strategy_limits: Dict[str, float] = {}
        self.exposure: Dict[str, float] = {}
        self.strategy_exposure: Dict[str, float] = {}
        self._positions: Dict[str, int] = {}
        self._atr_cache: Dict[str, tuple] = {}
        self._volatility_cache: Dict[str, tuple] = {}
        try:
            from data_client import DataClient
        except Exception:  # pragma: no cover - optional dependency
            self.data_client = None
        else:
            self.data_client = DataClient()
        self.hard_stop = False
        # AI-AGENT-REF: track returns/drawdown for adaptive exposure cap
        self._returns: list[float] = []
        self._drawdowns: list[float] = []
        self._last_portfolio_cap: float | None = None
        self._last_equity_cap: float | None = None
        # AI-AGENT-REF: signal exposure updates to trading loop
        from threading import Event
        self._update_event = Event()
        self._last_update = 0.0
        
        # Validate maximum acceptable drawdown (fraction between 0 and 1)
        try:
            max_drawdown = float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.1"))
            if not (0 < max_drawdown <= 1.0):
                logger.warning("Invalid MAX_DRAWDOWN_THRESHOLD %s, using default 0.1", max_drawdown)
                max_drawdown = 0.1
            self.max_drawdown_threshold = max_drawdown
        except (ValueError, TypeError) as e:
            logger.error("Error parsing MAX_DRAWDOWN_THRESHOLD: %s, using default 0.1", e)
            self.max_drawdown_threshold = 0.1
            
        # Validate cooldown period (in minutes) before trading resumes after a hard stop
        try:
            cooldown = float(os.getenv("HARD_STOP_COOLDOWN_MIN", "10"))
            if cooldown < 0:
                logger.warning("Invalid HARD_STOP_COOLDOWN_MIN %s, using default 10", cooldown)
                cooldown = 10.0
            self.hard_stop_cooldown = cooldown
        except (ValueError, TypeError) as e:
            logger.error("Error parsing HARD_STOP_COOLDOWN_MIN: %s, using default 10", e)
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

    def _get_atr_data(self, symbol: str, lookback: int = 14) -> float | None:
        """Return ATR value for ``symbol``."""
        try:
            if symbol in self._atr_cache:
                ts, val = self._atr_cache[symbol]
                from datetime import datetime, timedelta, timezone
                if datetime.now(timezone.utc) - ts < timedelta(minutes=30):
                    return val

            if self.data_client is None:
                return None
            bars = self.data_client.get_bars(symbol, lookback + 10)
            if len(bars) < lookback:
                return None
            high = np.array([b.h for b in bars])
            low = np.array([b.l for b in bars])
            close = np.array([b.c for b in bars])
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = float(np.mean(tr[-lookback:]))
            from datetime import datetime, timezone
            self._atr_cache[symbol] = (datetime.now(timezone.utc), atr)
            return atr
        except Exception as exc:
            logger.warning("ATR calculation error for %s: %s", symbol, exc)
            return None

    def _adaptive_global_cap(self) -> float:
        base_cap = self.global_limit

        # Handle missing config attributes gracefully
        volatility_lookback_days = getattr(self.config, 'volatility_lookback_days', 10)
        exposure_cap_conservative = getattr(self.config, 'exposure_cap_conservative', 1.0)

        # If no historical data, just use the base cap without conservative scaling
        if len(self._returns) < 3:  # Need minimum data for meaningful statistics
            return base_cap

        recent_returns = np.array(self._returns[-volatility_lookback_days:])

        mean_return = np.mean(recent_returns)
        vol = np.std(recent_returns) if np.std(recent_returns) > 0 else 0.01
        sharpe_proxy = mean_return / vol

        cumulative = np.cumprod(1 + recent_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

        if sharpe_proxy > 0.5 and max_dd < 0.05:
            multiplier = min(1.2, 1 + sharpe_proxy * 0.3)
        elif sharpe_proxy < -0.3 or max_dd > 0.1:
            multiplier = max(0.3, 1 - max_dd * 2)
        else:
            multiplier = 1.0

        adaptive_cap = base_cap * multiplier
        return np.clip(adaptive_cap, base_cap * 0.3, base_cap * 1.5)

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
                # AI-AGENT-REF: use signed quantity to track net exposure
                weight = qty * price / equity if equity > 0 else 0.0
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
        import time
        self._last_update = time.monotonic()
        self._update_event.set()

    def update_position(self, symbol: str, quantity: int, side: str) -> None:
        """Update exposure for a symbol."""
        if side == "buy":
            self._positions[symbol] = self._positions.get(symbol, 0) + quantity
        else:
            self._positions[symbol] = self._positions.get(symbol, 0) - quantity

    def update_returns(self, daily_return: float) -> None:
        """Append ``daily_return`` to history for adaptive calculations."""
        self._returns.append(daily_return)
        self._returns = self._returns[-90:]

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

    def wait_for_exposure_update(self, timeout: float = 0.5) -> None:
        """Block until an exposure update occurs or ``timeout`` elapses."""
        self._update_event.wait(timeout)
        self._update_event.clear()

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

    def position_size(self, signal: Any, cash: float, price: float, api=None) -> int:
        if price <= 0:
            logger.warning("Invalid price %s for %s", price, signal.symbol)
            return 0

        try:
            if api:
                account = api.get_account()
                total_equity = float(getattr(account, "equity", cash))
            else:
                total_equity = cash
        except Exception as e:
            logger.warning("Error getting account equity: %s", e)
            total_equity = cash

        try:
            atr_data = self._get_atr_data(signal.symbol)
            if atr_data and atr_data > 0:
                risk_per_trade = total_equity * 0.01
                stop_distance = atr_data * self.config.atr_multiplier
                raw_qty = risk_per_trade / stop_distance
            else:
                weight = self._apply_weight_limits(signal)
                raw_qty = (total_equity * weight) / price
        except Exception as exc:
            logger.warning("ATR calculation failed for %s: %s", signal.symbol, exc)
            weight = self._apply_weight_limits(signal)
            raw_qty = (total_equity * weight) / price

        min_qty = self.config.position_size_min_usd / price
        qty = max(int(raw_qty), int(min_qty)) if raw_qty >= min_qty else 0
        return qty

    def _apply_weight_limits(self, sig: TradeSignal) -> float:
        """Apply confidence-based weight limits."""
        # Use the signal's weight, respecting asset and strategy limits
        asset_limit = self.asset_limits.get(sig.asset_class, self.global_limit)
        strategy_limit = self.strategy_limits.get(sig.strategy, self.global_limit)
        max_allowed = min(asset_limit, strategy_limit)
        # Apply confidence scaling to the allowed weight, but don't cap by kelly_fraction_max for simple position sizing
        base_weight = min(sig.weight * sig.confidence, max_allowed)
        return base_weight

    def compute_volatility(self, returns: np.ndarray) -> dict:
        """Return multiple volatility estimates."""
        if len(returns) == 0:
            return {"volatility": 0.0, "mad": 0.0, "garch_vol": 0.0}

        std_vol = float(np.std(returns))
        mad = float(np.median(np.abs(returns - np.median(returns))))

        try:
            alpha, beta = 0.1, 0.85
            garch_vol = 0.0
            for i in range(1, len(returns)):
                garch_vol = alpha * returns[i - 1] ** 2 + beta * garch_vol
            garch_vol = float(np.sqrt(garch_vol))
        except Exception:
            garch_vol = std_vol

        primary_vol = mad * 1.4826 if mad > 0 else std_vol
        return {
            "volatility": primary_vol,
            "std_vol": std_vol,
            "mad": mad,
            "garch_vol": garch_vol,
        }


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
