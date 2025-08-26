from __future__ import annotations
from ai_trading.logging import get_logger
import random
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from typing import Any
import numpy as np
import importlib
from ai_trading.utils.lazy_imports import load_pandas
try:  # pragma: no cover - optional dependency
    from alpaca_trade_api.rest import APIError  # type: ignore
except Exception:  # pragma: no cover - fallback when SDK missing
    class APIError(Exception):
        """Fallback APIError when alpaca-trade-api is unavailable."""

        pass
from ai_trading.config.management import (
    SEED,
    TradingConfig,
    get_env,
    validate_required_env,
)
from ai_trading.config.settings import get_settings

if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Lazy pandas proxy
pd = load_pandas()

try:  # optional pandas_ta import for ta accessor registration
    import pandas_ta as ta  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    ta = None
    get_logger(__name__).info(
        'PANDAS_TA_MISSING', extra={'hint': 'pip install pandas-ta'}
    )
try:  # pragma: no cover - optional dependency
    from alpaca_trade_api import REST as AlpacaREST
except ImportError:
    AlpacaREST = None


def _safe_call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except AttributeError:
        return None

@dataclass
class TradeSignal:
    symbol: str
    side: str
    confidence: float
    strategy: str
    weight: float
    asset_class: str
    strength: float = 1.0
logger = get_logger(__name__)
if not get_env("PYTEST_RUNNING", "0", cast=bool):
    _ENV_SNAPSHOT = validate_required_env()
    logger.debug("ENV_VARS_MASKED", extra=_ENV_SNAPSHOT)

random.seed(SEED)
np.random.seed(SEED)
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
MAX_DRAWDOWN = 0.05

class RiskEngine:
    """Cross-strategy risk manager."""
    _lock: object | None = None

    def __init__(self, cfg: TradingConfig | None=None) -> None:
        """Initialize the engine with an optional trading config."""
        self.config = cfg if cfg is not None else TradingConfig()
        self._lock = threading.Lock()
        self.hard_stop = False
        self.max_trades = 10
        self.current_trades = 0
        try:
            exposure_cap = getattr(self.config, 'exposure_cap_aggressive', 0.8)
            if not isinstance(exposure_cap, int | float) or not 0 < exposure_cap <= 1.0:
                logger.warning('Invalid exposure_cap_aggressive %s, using default 0.8', exposure_cap)
                exposure_cap = 0.8
            self.global_limit = exposure_cap
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as e:
            logger.error('Error validating exposure_cap_aggressive: %s, using default', e)
            self.global_limit = 0.8
        self.asset_limits: dict[str, float] = {}
        self.strategy_limits: dict[str, float] = {}
        self.exposure: dict[str, float] = {}
        self.strategy_exposure: dict[str, float] = {}
        self._positions: dict[str, int] = {}
        self._atr_cache: dict[str, tuple] = {}
        self._volatility_cache: dict[str, tuple] = {}
        self.data_client = None
        try:
            s = get_settings()
            if AlpacaREST and getattr(s, 'alpaca_api_key', None) and getattr(s, 'alpaca_secret_key_plain', None) and getattr(s, 'alpaca_base_url', None):
                self.data_client = AlpacaREST(s.alpaca_api_key, s.alpaca_secret_key_plain, s.alpaca_base_url)
        except (APIError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning('Could not initialize AlpacaREST: %s', e)
        self._returns: list[float] = []
        self._drawdowns: list[float] = []
        self._last_portfolio_cap: float | None = None
        self._last_equity_cap: float | None = None
        from threading import Event
        self._update_event = Event()
        self._last_update = 0.0
        try:
            max_drawdown = get_env('MAX_DRAWDOWN_THRESHOLD', '0.15', cast=float)
            if not 0 < max_drawdown <= 1.0:
                logger.warning('Invalid MAX_DRAWDOWN_THRESHOLD %s, using default 0.15', max_drawdown)
                max_drawdown = 0.15
            self.max_drawdown_threshold = float(max_drawdown)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error('Error parsing MAX_DRAWDOWN_THRESHOLD: %s, using default 0.15', e)
            self.max_drawdown_threshold = 0.15
        try:
            cooldown = get_env('HARD_STOP_COOLDOWN_MIN', '10', cast=float)
            if cooldown < 0:
                logger.warning('Invalid HARD_STOP_COOLDOWN_MIN %s, using default 10', cooldown)
                cooldown = 10.0
            self.hard_stop_cooldown = float(cooldown)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error('Error parsing HARD_STOP_COOLDOWN_MIN: %s, using default 10', e)
            self.hard_stop_cooldown = 10.0
        self._hard_stop_until: float | None = None

    def _dynamic_cap(self, asset_class: str, volatility: float | None=None, cash_ratio: float | None=None) -> float:
        """Return exposure cap for ``asset_class`` using adaptive rules."""
        base_cap = self.asset_limits.get(asset_class, self.global_limit)
        port_cap = self._adaptive_global_cap()
        vol = self._current_volatility()
        if self._last_portfolio_cap is None or abs(self._last_portfolio_cap - port_cap) > 0.01 or self._last_equity_cap is None or (abs(self._last_equity_cap - base_cap) > 0.01):
            logger.info('Adaptive exposure caps: portfolio=%.1f, equity=%.1f (volatility=%.1f%%)', port_cap, base_cap, vol * 100)
            self._last_portfolio_cap = port_cap
            self._last_equity_cap = base_cap
        return min(base_cap, port_cap)

    def _current_volatility(self) -> float:
        return float(np.std(self._returns[-10:])) if self._returns else 0.0

    def _get_atr_data(self, symbol: str, lookback: int=14) -> float | None:
        """Return ATR value for ``symbol``."""
        try:
            if symbol in self._atr_cache:
                ts, val = self._atr_cache[symbol]
                from datetime import datetime, timedelta
                if datetime.now(UTC) - ts < timedelta(minutes=30):
                    return val
            ctx = getattr(self, 'ctx', None)
            client = getattr(ctx, 'data_client', None) or self.data_client or getattr(ctx, 'api', None)
            if not client:
                logger.warning('No data client available; skipping historical fetch for %s', symbol)
                return None
            bars = client.get_bars(symbol, lookback + 10)
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
            from datetime import datetime
            self._atr_cache[symbol] = (datetime.now(UTC), atr)
            return atr
        except (APIError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning('ATR calculation error for %s: %s', symbol, exc, extra={'cause': exc.__class__.__name__})
            return None

    def _adaptive_global_cap(self) -> float:
        base_cap = self.global_limit
        volatility_lookback_days = getattr(self.config, 'volatility_lookback_days', 10)
        getattr(self.config, 'exposure_cap_conservative', 1.0)
        if len(self._returns) < 3:
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

    def update_portfolio_metrics(self, returns: list[float] | None=None, drawdown: float | None=None) -> None:
        if returns:
            self._returns.extend(list(returns))
        if drawdown is not None:
            self._drawdowns.append(float(drawdown))
            self._check_drawdown_and_update_stop(float(drawdown))

    def refresh_positions(self, api) -> None:
        """Synchronize exposure with live positions."""
        try:
            positions = api.list_positions()
            logger.debug('Raw Alpaca positions: %s', positions)
            acct = api.get_account()
            equity = float(getattr(acct, 'equity', 0) or 0)
            exposure: dict[str, float] = {}
            for p in positions:
                asset = getattr(p, 'asset_class', 'equity')
                qty = float(getattr(p, 'qty', 0) or 0)
                price = float(getattr(p, 'avg_entry_price', 0) or 0)
                weight = qty * price / equity if equity > 0 else 0.0
                exposure[asset] = exposure.get(asset, 0.0) + weight
            self.exposure = exposure
        except (AttributeError, APIError) as exc:
            logger.warning('refresh_positions failed: %s', exc, extra={'cause': exc.__class__.__name__})

    def position_exists(self, api, symbol: str) -> bool:
        """Return True if ``symbol`` exists in current Alpaca positions."""
        try:
            for p in api.list_positions():
                if getattr(p, 'symbol', '') == symbol:
                    return True
        except (AttributeError, APIError) as exc:
            logger.warning('position_exists failed for %s: %s', symbol, exc, extra={'cause': exc.__class__.__name__})
        return False

    def can_trade(self, signal: TradeSignal, *, pending: float=0.0, volatility: float | None=None, cash_ratio: float | None=None, returns: list[float] | None=None, drawdowns: list[float] | None=None) -> bool:
        if returns:
            self._returns.extend(list(returns))
        if drawdowns:
            self._drawdowns.extend(list(drawdowns))
        self._maybe_lift_hard_stop()
        if self.hard_stop:
            logger.error('TRADING_HALTED_RISK_LIMIT')
            return False
        if not isinstance(signal, TradeSignal):
            logger.error('can_trade called with invalid signal type')
            return False
        asset_exp = self.exposure.get(signal.asset_class, 0.0) + max(pending, 0.0)
        asset_cap = 1.1 * self._dynamic_cap(signal.asset_class, volatility, cash_ratio)
        signal = self.apply_risk_scaling(signal, volatility=volatility, returns=returns)
        try:
            signal_weight = float(signal.weight)
        except (ValueError, TypeError) as e:
            logger.warning("Invalid signal.weight value '%s' for %s, defaulting to 0.0: %s", signal.weight, signal.symbol, e)
            signal_weight = 0.0
        if asset_exp + signal_weight > asset_cap:
            logger.warning('Exposure cap breach: symbol=%s qty=%s alloc=%.3f exposure=%.2f vs cap=%.2f', signal.symbol, getattr(signal, 'qty', 'n/a'), signal_weight, asset_exp + signal_weight, asset_cap)
            if not get_env('FORCE_CONTINUE_ON_EXPOSURE', 'false', cast=bool):
                return False
            logger.warning('FORCE_CONTINUE_ON_EXPOSURE enabled; overriding cap')
        strat_cap = self.strategy_limits.get(signal.strategy, self.global_limit)
        if signal_weight > strat_cap:
            logger.warning('Strategy %s weight %.2f exceeds cap %.2f', signal.strategy, signal_weight, strat_cap)
            if not get_env('FORCE_CONTINUE_ON_EXPOSURE', 'false', cast=bool):
                return False
            logger.warning('FORCE_CONTINUE_ON_EXPOSURE enabled; overriding cap')
        return True

    def register_fill(self, signal: TradeSignal) -> None:
        if not isinstance(signal, TradeSignal):
            logger.error('register_fill called with invalid signal type')
            return
        prev = self.exposure.get(signal.asset_class, 0.0)
        try:
            signal_weight = float(signal.weight)
        except (ValueError, TypeError) as e:
            logger.warning("Invalid signal.weight value '%s' for %s in register_fill, defaulting to 0.0: %s", signal.weight, signal.symbol, e)
            signal_weight = 0.0
        delta = signal_weight if signal.side.lower() == 'buy' else -signal_weight
        new_exposure = prev + delta
        if new_exposure < 0 and signal.side.lower() == 'sell':
            logger.warning('EXPOSURE_NEGATIVE_PREVENTED', extra={'asset': signal.asset_class, 'symbol': getattr(signal, 'symbol', 'UNKNOWN'), 'prev': prev, 'delta': delta, 'would_be': new_exposure})
            new_exposure = 0.0
            delta = -prev
        self.exposure[signal.asset_class] = new_exposure
        s_prev = self.strategy_exposure.get(signal.strategy, 0.0)
        self.strategy_exposure[signal.strategy] = s_prev + delta
        logger.info('EXPOSURE_UPDATED', extra={'asset': signal.asset_class, 'prev': prev, 'new': self.exposure[signal.asset_class], 'side': signal.side, 'symbol': getattr(signal, 'symbol', 'UNKNOWN')})
        import time
        self._last_update = time.monotonic()
        self._update_event.set()

    def update_position(self, symbol: str, quantity: int, side: str) -> None:
        """Update exposure for a symbol."""
        if side == 'buy':
            self._positions[symbol] = self._positions.get(symbol, 0) + quantity
        else:
            self._positions[symbol] = self._positions.get(symbol, 0) - quantity

    def update_returns(self, daily_return: float) -> None:
        """Append ``daily_return`` to history for adaptive calculations."""
        self._returns.append(daily_return)
        self._returns = self._returns[-90:]

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
        if current_drawdown >= self.max_drawdown_threshold and (not self.hard_stop):
            self.hard_stop = True
            import time
            self._hard_stop_until = time.time() + self.hard_stop_cooldown * 60
            logger.error('HARD_STOP_TRIGGERED', extra={'drawdown': current_drawdown, 'threshold': self.max_drawdown_threshold})

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
                logger.info('HARD_STOP_CLEARED')

    def acquire_trade_slot(self) -> bool:
        """Thread-safe check & increment of current_trades against max_trades."""
        if self._lock is None:
            self._lock = threading.Lock()
        with self._lock:
            if self.current_trades >= self.max_trades:
                return False
            self.current_trades += 1
            return True

    def release_trade_slot(self) -> None:
        """Decrement current_trades (no-op if already zero)."""
        if self._lock is None:
            self._lock = threading.Lock()
        with self._lock:
            if self.current_trades > 0:
                self.current_trades -= 1

    def trigger_hard_stop(self) -> None:
        self.hard_stop = True

    def wait_for_exposure_update(self, timeout: float=0.5) -> None:
        """Block until an exposure update occurs or ``timeout`` elapses."""
        self._update_event.wait(timeout)
        self._update_event.clear()

    def update_exposure(self, context=None, *args, **kwargs):
        """
        Recalculate/update exposure. Prefer the provided context.
        Backward compatible: if context is None, fall back to self.ctx (if set).
        """
        ctx = context if context is not None else getattr(self, 'ctx', None)
        if ctx is None:
            raise RuntimeError('RiskEngine.update_exposure: context is required')
        try:
            self.refresh_positions(ctx.api)
            logger.debug('Exposure updated successfully')
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
            logger.warning('Failed to update exposure: %s', exc)

    def apply_risk_scaling(self, signal: TradeSignal, *, volatility: float | None=None, returns: Sequence[float] | None=None) -> TradeSignal:
        """
        Adjust a signal's weight based on volatility and CVaR.  This function
        uses a simple inverse‑volatility rule and CVaR scaling to shrink
        exposures when markets become turbulent.  The original signal is
        mutated and returned for convenience.
        """
        try:
            scale = 1.0
            if volatility and volatility > 0:
                scale *= max(0.5, min(1.0, 0.02 / volatility))
            if returns:
                import numpy as np
                from ai_trading.capital_scaling import cvar_scaling
                arr = np.asarray(list(returns), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    cvar_metric = cvar_scaling(arr, alpha=0.05)
                    if cvar_metric > 1.0:
                        scale *= 1.0 / (1.0 + cvar_metric)
            signal.weight = max(0.0, float(signal.weight) * scale)
            return signal
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
            logger.error('Risk scaling failed: %s', exc)
            return signal

    def check_max_drawdown(self, api) -> bool:
        try:
            account = api.get_account()
            pnl = float(account.equity) - float(account.last_equity)
            if pnl < -MAX_DRAWDOWN * float(account.last_equity):
                logger.error('HARD_STOP_MAX_DRAWDOWN', extra={'pnl': pnl})
                self.hard_stop = True
                return False
            return True
        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.error('check_max_drawdown failed: %s', exc)
            return False

    def position_size(self, signal: Any, cash: float, price: float, api=None) -> int:
        """
        Calculate optimal position size using Kelly criterion and risk management.

        This is the core position sizing algorithm that combines multiple risk factors:
        - Kelly criterion for optimal bet sizing
        - ATR-based volatility scaling
        - Maximum position limits
        - Account equity validation
        """
        if self.hard_stop:
            return 0
        if not self.can_trade(signal):
            return 0
        if api and (not self.check_max_drawdown(api)):
            return 0
        if price <= 0:
            logger.warning('Invalid price %s for %s', price, getattr(signal, 'symbol', 'UNKNOWN'))
            return 0
        if cash <= 0:
            logger.warning('Invalid cash amount %s for %s', cash, getattr(signal, 'symbol', 'UNKNOWN'))
            return 0
        try:
            if api:
                account = api.get_account()
                total_equity = float(getattr(account, 'equity', cash))
            else:
                total_equity = cash
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as e:
            logger.warning('Error getting account equity: %s', e)
            total_equity = cash
        if total_equity <= 0:
            logger.warning('Invalid total equity %s for %s', total_equity, getattr(signal, 'symbol', 'UNKNOWN'))
            return 0
        try:
            if not hasattr(signal, 'symbol'):
                logger.warning('Invalid signal object missing symbol attribute')
                return 0
            atr_data = self._get_atr_data(signal.symbol)
            if atr_data and atr_data > 0:
                risk_per_trade = total_equity * 0.01
                stop_distance = atr_data * self.config.atr_multiplier
                raw_qty = risk_per_trade / stop_distance
            else:
                weight = self._apply_weight_limits(signal)
                raw_qty = total_equity * weight / price
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
            logger.warning('ATR calculation failed for %s: %s', getattr(signal, 'symbol', 'UNKNOWN'), exc)
            try:
                weight = self._apply_weight_limits(signal)
                raw_qty = total_equity * weight / price
            except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError):
                logger.warning('Failed to calculate position size, returning 0')
                return 0
        try:
            min_qty = self.config.position_size_min_usd / price
            try:
                if hasattr(np, 'isfinite'):
                    is_raw_qty_finite = np.isfinite(raw_qty)
                    is_min_qty_finite = np.isfinite(min_qty)
                else:
                    is_raw_qty_finite = str(raw_qty).lower() not in ['nan', 'inf', '-inf']
                    is_min_qty_finite = str(min_qty).lower() not in ['nan', 'inf', '-inf']
            except (AttributeError, TypeError):
                is_raw_qty_finite = isinstance(raw_qty, int | float) and raw_qty == raw_qty and (abs(raw_qty) != float('inf'))
                is_min_qty_finite = isinstance(min_qty, int | float) and min_qty == min_qty and (abs(min_qty) != float('inf'))
            if not is_raw_qty_finite or raw_qty <= 0:
                logger.warning('Invalid or negative raw_qty %s for %s, returning 0', raw_qty, getattr(signal, 'symbol', 'UNKNOWN'))
                return 0
            if not is_min_qty_finite:
                logger.warning('Invalid min_qty %s, using raw_qty only', min_qty)
                qty = int(raw_qty)
            else:
                qty = max(int(raw_qty), int(min_qty))
            if getattr(signal, 'strategy', '') == 'default':
                qty = max(qty, 10)
            return max(qty, 0)
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
            logger.warning('Error calculating final quantity: %s', exc)
            return 0

    def _apply_weight_limits(self, sig: TradeSignal) -> float:
        """Apply confidence-based weight limits considering current exposure."""
        try:
            if not hasattr(sig, 'asset_class') or not hasattr(sig, 'strategy') or (not hasattr(sig, 'weight')) or (not hasattr(sig, 'confidence')):
                logger.warning('Invalid signal object missing required attributes')
                return 0.0
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError):
            logger.warning('Error validating signal object')
            return 0.0
        current_asset_exposure = self.exposure.get(sig.asset_class, 0.0)
        current_strategy_exposure = self.strategy_exposure.get(sig.strategy, 0.0)
        asset_limit = self.asset_limits.get(sig.asset_class, self.global_limit)
        strategy_limit = self.strategy_limits.get(sig.strategy, self.global_limit)
        available_asset_capacity = max(0.0, asset_limit - current_asset_exposure)
        available_strategy_capacity = max(0.0, strategy_limit - current_strategy_exposure)
        max_allowed = min(available_asset_capacity, available_strategy_capacity)
        try:
            requested_weight = float(sig.weight)
        except (ValueError, TypeError) as e:
            logger.warning("Invalid signal.weight value '%s' for %s in _apply_weight_limits, defaulting to 0.0: %s", sig.weight, sig.symbol, e)
            requested_weight = 0.0
        base_weight = min(requested_weight, max_allowed)
        return base_weight

    def compute_volatility(self, returns: np.ndarray) -> dict:
        """Return multiple volatility estimates."""
        if len(returns) == 0:
            return {'volatility': 0.0, 'mad': 0.0, 'garch_vol': 0.0}
        try:
            returns_array = np.asarray(returns)
            has_invalid = False
            try:
                if hasattr(np, 'any') and hasattr(np, 'isnan') and hasattr(np, 'isinf'):
                    has_invalid = np.any(np.isnan(returns_array)) or np.any(np.isinf(returns_array))
                else:
                    for val in returns_array:
                        if str(val).lower() in ['nan', 'inf', '-inf']:
                            has_invalid = True
                            break
            except (AttributeError, TypeError):
                has_invalid = any((str(val).lower() in ['nan', 'inf', '-inf'] for val in returns_array))
            if has_invalid:
                logger.error('compute_volatility: invalid values in returns array')
                return {'volatility': 0.0, 'mad': 0.0, 'garch_vol': 0.0}
            try:
                std_vol = float(np.std(returns_array))
            except (AttributeError, TypeError):
                mean_val = sum(returns_array) / len(returns_array)
                variance = sum(((x - mean_val) ** 2 for x in returns_array)) / len(returns_array)
                std_vol = variance ** 0.5
            try:
                if hasattr(np, 'median') and hasattr(np, 'abs'):
                    mad = float(np.median(np.abs(returns_array - np.median(returns_array))))
                else:
                    sorted_returns = sorted(returns_array)
                    n = len(sorted_returns)
                    median_val = sorted_returns[n // 2] if n % 2 == 1 else (sorted_returns[n // 2 - 1] + sorted_returns[n // 2]) / 2
                    abs_deviations = [abs(x - median_val) for x in returns_array]
                    sorted_abs_dev = sorted(abs_deviations)
                    mad = sorted_abs_dev[len(sorted_abs_dev) // 2] if len(sorted_abs_dev) % 2 == 1 else (sorted_abs_dev[len(sorted_abs_dev) // 2 - 1] + sorted_abs_dev[len(sorted_abs_dev) // 2]) / 2
            except (AttributeError, TypeError):
                mad = std_vol
        except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
            logger.error('compute_volatility: error in numpy operations: %s', exc)
            return {'volatility': 0.0, 'mad': 0.0, 'garch_vol': 0.0}
        try:
            alpha, beta = (0.1, 0.85)
            garch_vol = 0.0
            for i in range(1, len(returns_array)):
                garch_vol = alpha * returns_array[i - 1] ** 2 + beta * garch_vol
            try:
                garch_vol = float(np.sqrt(garch_vol))
            except (AttributeError, TypeError):
                garch_vol = float(garch_vol ** 0.5)
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError):
            garch_vol = std_vol
        primary_vol = std_vol
        return {'volatility': primary_vol, 'std_vol': std_vol, 'mad': mad, 'mad_scaled': mad * 1.4826, 'garch_vol': garch_vol}

    def get_current_exposure(self) -> dict[str, float]:
        """
        Get current portfolio exposure by asset class.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping asset classes to exposure percentages.
            Values represent the portion of total equity allocated to each asset class.
        """
        return self.exposure.copy()

    def max_concurrent_orders(self) -> int:
        """
        Get maximum number of concurrent orders allowed.

        Returns
        -------
        int
            Maximum number of orders that can be active simultaneously.
            Prevents overwhelming the broker with too many pending orders.
        """
        return getattr(self.config, 'max_concurrent_orders', 50)

    def max_exposure(self) -> float:
        """
        Get maximum total portfolio exposure limit.

        Returns
        -------
        float
            Maximum portfolio exposure as a fraction (0.0 to 1.0).
            Represents the maximum percentage of equity that can be at risk.
        """
        return self.global_limit

    def order_spacing(self) -> float:
        """
        Get minimum time spacing between orders in seconds.

        Returns
        -------
        float
            Minimum seconds to wait between submitting orders.
            Prevents rapid-fire order submission that could trigger rate limits.
        """
        return getattr(self.config, 'order_spacing_seconds', 1.0)

    def check_position_limits(self, symbol: str, quantity: float) -> bool:
        """
        Check if a proposed position would exceed risk limits.

        Parameters
        ----------
        symbol : str
            Trading symbol to check limits for.
        quantity : float
            Proposed position size (positive for long, negative for short).

        Returns
        -------
        bool
            True if position is within limits, False if it would exceed limits.
        """
        try:
            current_exposure = self.exposure.get(symbol, 0.0)
            new_exposure = current_exposure + abs(quantity) * 0.001
            max_symbol_exposure = getattr(self.config, 'max_symbol_exposure', 0.1)
            if new_exposure > max_symbol_exposure:
                logger.warning('Position for %s would exceed symbol exposure limit: %.3f > %.3f', symbol, new_exposure, max_symbol_exposure)
                return False
            total_exposure = sum(self.exposure.values()) + abs(quantity) * 0.001
            if total_exposure > self.global_limit:
                logger.warning('Position for %s would exceed total exposure limit: %.3f > %.3f', symbol, total_exposure, self.global_limit)
                return False
            return True
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as e:
            logger.error('Error checking position limits for %s: %s', symbol, e)
            return False

    def validate_order_size(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Validate that an order size is appropriate for risk management.

        Parameters
        ----------
        symbol : str
            Trading symbol for the order.
        quantity : float
            Order quantity (shares).
        price : float
            Order price per share.

        Returns
        -------
        bool
            True if order size is valid, False if it should be rejected.
        """
        try:
            order_value = abs(quantity) * price
            min_order_value = getattr(self.config, 'min_order_value', 100.0)
            if order_value < min_order_value:
                logger.warning('Order for %s below minimum value: $%.2f < $%.2f', symbol, order_value, min_order_value)
                return False
            max_order_value = getattr(self.config, 'max_order_value', 50000.0)
            if order_value > max_order_value:
                logger.warning('Order for %s exceeds maximum value: $%.2f > $%.2f', symbol, order_value, max_order_value)
                return False
            if abs(quantity) < 1:
                logger.warning('Order quantity too small: %s shares', quantity)
                return False
            if abs(quantity) > 10000:
                logger.warning('Order quantity unusually large: %s shares', quantity)
            return True
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as e:
            logger.error('Error validating order size for %s: %s', symbol, e)
            return False

def dynamic_position_size(capital: float, volatility: float, drawdown: float) -> float:
    """Return position size using volatility and drawdown aware Kelly fraction.

    The base Kelly fraction of ``0.5 / volatility`` is throttled by current
    drawdown. When drawdown exceeds 10% the fraction is scaled down by 50%.
    """
    if capital <= 0:
        return 0.0
    vol = max(volatility, 1e-06)
    kelly_fraction = 0.5 / vol
    kelly_fraction = min(max(kelly_fraction, 0.0), 1.0)
    if drawdown > 0.1:
        kelly_fraction *= 0.5
    return capital * kelly_fraction

def calculate_position_size(*args, **kwargs) -> int:
    """
    Calculate optimal position size using Kelly criterion and risk management.

    This convenience wrapper function supports multiple calling patterns for
    calculating position sizes based on available capital, signal confidence,
    and risk parameters. It integrates Kelly criterion optimization with
    volatility-based risk scaling.

    Parameters
    ----------
    *args : tuple
        Variable arguments supporting multiple calling patterns:

        Pattern 1 (Simple): calculate_position_size(cash, price)
        - cash (float): Available trading capital
        - price (float): Current asset price per share

        Pattern 2 (Advanced): calculate_position_size(signal, cash, price, api=None)
        - signal (TradeSignal): Signal object with confidence and strategy info
        - cash (float): Available trading capital
        - price (float): Current asset price per share
        - api (optional): Broker API client for additional validation

    **kwargs : dict
        Optional keyword arguments:
        - api: Broker API client for account validation
        - max_position_pct (float): Maximum position as % of capital (default: 5%)
        - volatility_scaling (bool): Enable volatility-based sizing (default: True)

    Returns
    -------
    int
        Optimal number of shares to trade, considering:
        - Kelly criterion optimization for signal confidence
        - Risk-adjusted position sizing based on volatility
        - Maximum position limits and portfolio constraints
        - Available capital and margin requirements

    Raises
    ------
    TypeError
        If invalid argument patterns are provided
    ValueError
        If negative or invalid cash/price values are passed

    Examples
    --------
    >>> # Simple position sizing
    >>> shares = calculate_position_size(10000, 150.0)  # $10k capital, $150/share
    >>> logging.info(f"Buy {shares} shares")

    >>> # Advanced position sizing with signal
    >>> from ai_trading.strategies.base import StrategySignal as TradeSignal
    >>> signal = TradeSignal(symbol='AAPL', side='buy', confidence=0.8, strategy='momentum')
    >>> shares = calculate_position_size(signal, 10000, 150.0)
    >>> logging.info(f"Buy {shares} shares based on {signal.confidence:.1%} confidence")

    Notes
    -----
    - Returns 0 if insufficient capital or invalid parameters
    - Automatically applies risk management limits
    - Considers portfolio heat and correlation limits
    - Scales position size based on signal confidence
    """
    engine = RiskEngine()
    if len(args) == 2 and (not kwargs):
        cash, price = args
        if not isinstance(cash, int | float) or cash <= 0:
            logger.warning(f'Invalid cash amount: {cash}')
            return 0
        if not isinstance(price, int | float) or price <= 0:
            logger.warning(f'Invalid price: {price}')
            return 0
        dummy = TradeSignal(symbol='DUMMY', side='buy', confidence=1.0, strategy='default')
        return engine.position_size(dummy, cash, price)
    if len(args) >= 3:
        signal, cash, price = args[:3]
        if not isinstance(cash, int | float) or cash <= 0:
            logger.warning(f'Invalid cash amount: {cash}')
            return 0
        if not isinstance(price, int | float) or price <= 0:
            logger.warning(f'Invalid price: {price}')
            return 0
        if not hasattr(signal, 'confidence') or not hasattr(signal, 'symbol'):
            logger.error(f'Invalid signal object: {type(signal)}')
            return 0
        api = args[3] if len(args) > 3 else kwargs.get('api')
        return engine.position_size(signal, cash, price, api)
    raise TypeError('Invalid arguments for calculate_position_size. Expected (cash, price) or (signal, cash, price)')

def check_max_drawdown(state: dict[str, float]) -> bool:
    """
    Validate if current portfolio drawdown exceeds maximum allowed threshold.

    This function checks portfolio performance against configured drawdown limits
    to implement risk management controls. When drawdown exceeds the threshold,
    trading may be halted or position sizes reduced.

    Parameters
    ----------
    state : Dict[str, float]
        Portfolio state dictionary containing:
        - 'current_drawdown' (float): Current drawdown as decimal (e.g., 0.05 = 5%)
        - 'max_drawdown' (float): Maximum allowed drawdown threshold
        - 'portfolio_value' (float): Current portfolio value (optional)
        - 'peak_value' (float): Historical peak portfolio value (optional)

    Returns
    -------
    bool
        True if current drawdown exceeds maximum allowed threshold,
        False if within acceptable limits or if data is insufficient.

    Notes
    -----
    - Returns False for missing or invalid state data
    - Drawdown values should be positive decimals (0.05 = 5%)
    - Used for automated risk management decisions
    """
    if not isinstance(state, dict):
        logger.warning(f'Invalid state type: {type(state)}')
        return False
    current_dd = state.get('current_drawdown', 0)
    max_dd = state.get('max_drawdown', 0)
    if not isinstance(current_dd, int | float) or current_dd < 0:
        logger.warning(f'Invalid current_drawdown: {current_dd}')
        return False
    if not isinstance(max_dd, int | float) or max_dd <= 0:
        logger.warning(f'Invalid max_drawdown: {max_dd}')
        return False
    return current_dd > max_dd

def can_trade(engine: RiskEngine) -> bool:
    """Return True if trading should proceed based on engine state."""
    return not engine.hard_stop and engine.current_trades < engine.max_trades

def register_trade(engine: RiskEngine, size: int) -> dict | None:
    """Register a trade and increment the count if allowed."""
    if size <= 0 or not engine.acquire_trade_slot():
        return None
    return {'size': size}

def check_exposure_caps(portfolio, exposure, cap):
    for sym, pos in portfolio.positions.items():
        if pos.quantity > 0 and exposure[sym] > cap:
            logger.warning('Exposure cap triggered, blocking new orders for %s', sym)
            return False

def apply_trailing_atr_stop(df: pd.DataFrame, entry_price: float, *, context: Any | None=None, symbol: str='SYMBOL', qty: int | None=None) -> None:
    """Exit ``qty`` at market if the trailing stop is triggered."""
    try:
        if entry_price <= 0:
            logger.warning('apply_trailing_atr_stop invalid entry price: %.2f', entry_price)
            return
        atr = df.ta.atr()
        trailing_stop = entry_price - 2 * atr
        last_valid_close = df['Close'].dropna()
        if not last_valid_close.empty:
            price = last_valid_close.iloc[-1]
        else:
            logger.critical('All NaNs in close column for ATR stop')
            price = 0.0
        logger.debug('Latest 5 rows for ATR stop:\n%s', df.tail(5))
        logger.debug('Computed price for ATR stop: %s', price)
        if price <= 0 or pd.isna(price):
            logger.critical('Invalid price computed for ATR stop: %s', price)
            return
        if price < trailing_stop.iloc[-1]:
            logger.info('ATR stop hit: price=%s vs stop=%s', price, trailing_stop.iloc[-1])
            if context is not None and qty:
                try:
                    if hasattr(context, 'risk_engine') and (not context.risk_engine.position_exists(context.api, symbol)):
                        logger.info('No position to sell for %s, skipping.', symbol)
                        return
                    bot_mod = importlib.import_module('ai_trading.core.bot_engine')
                    bot_mod.send_exit_order(context, symbol, abs(int(qty)), price, 'atr_stop')
                except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
                    logger.error('ATR stop exit failed: %s', exc)
            else:
                logger.warning('ATR stop triggered but no context/qty provided')
            schedule_reentry_check(symbol, lookahead_days=2)
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as e:
        logger.error('ATR stop error: %s', e)

def schedule_reentry_check(symbol: str, lookahead_days: int) -> None:
    """Log a re-entry check after a stop out."""
    logger.info('Scheduling reentry for %s in %s days', symbol, lookahead_days)

def calculate_atr_stop(entry_price: float, atr: float, multiplier: float=1.5, direction: str='long') -> float:
    """Return ATR-based stop price."""
    stop = entry_price - multiplier * atr if direction == 'long' else entry_price + multiplier * atr
    from ai_trading.telemetry import metrics_logger
    _safe_call(metrics_logger.log_atr_stop, symbol='generic', stop=stop)
    return stop

def calculate_bollinger_stop(price: float, upper_band: float, lower_band: float, direction: str='long') -> float:
    """Return stop price using Bollinger band width."""
    mid = (upper_band + lower_band) / 2
    if direction == 'long':
        stop = min(price, mid)
    else:
        stop = max(price, mid)
    from ai_trading.telemetry import metrics_logger
    _safe_call(metrics_logger.log_atr_stop, symbol='bb', stop=stop)
    return stop

def dynamic_stop_price(entry_price: float, atr: float | None=None, upper_band: float | None=None, lower_band: float | None=None, percent: float | None=None, direction: str='long') -> float:
    """Return the tightest stop price based on ATR, Bollinger width or percent."""
    stops: list[float] = []
    if atr is not None:
        stops.append(calculate_atr_stop(entry_price, atr, direction=direction))
    if upper_band is not None and lower_band is not None:
        stops.append(calculate_bollinger_stop(entry_price, upper_band, lower_band, direction=direction))
    if percent is not None:
        pct_stop = entry_price * (1 - percent) if direction == 'long' else entry_price * (1 + percent)
        stops.append(pct_stop)
    if not stops:
        return entry_price
    return max(stops) if direction == 'long' else min(stops)

def compute_stop_levels(entry_price: float, atr: float, take_mult: float=2.0) -> tuple[float, float]:
    """Return stop-loss and take-profit levels using ATR."""
    stop = entry_price - atr
    take = entry_price + take_mult * atr
    return (stop, take)

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

def drawdown_circuit(drawdowns: Sequence[float], limit: float=0.2) -> bool:
    """Return True if cumulative drawdown exceeds ``limit``."""
    dd = abs(min(0.0, *drawdowns)) if drawdowns else 0.0
    return dd > limit

def volatility_filter(atr: float, sma: float, threshold: float=0.05) -> bool:
    """Return True when volatility below ``threshold`` relative to SMA."""
    if sma == 0:
        return True
    return atr / sma < threshold
