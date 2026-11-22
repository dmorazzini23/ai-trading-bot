from __future__ import annotations
'Circuit breakers and safety mechanisms for production trading.'
import json
import math
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID
from ai_trading.logging import logger
from ai_trading.logging.emit_once import emit_once
from json import JSONDecodeError
from ai_trading.exc import RequestException
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.utils.time import safe_utcnow

COMMON_EXC = (TypeError, ValueError, KeyError, JSONDecodeError, RequestException, TimeoutError, ImportError)
from ..core.constants import PERFORMANCE_THRESHOLDS

# A tripped drawdown breaker resets only once drawdown is back at or below this
# fraction of the trip level (0.8 x 8% = 6.4%), so equity noise around either
# level cannot trip and reset it on consecutive readings.
DRAWDOWN_RESET_FRACTION = 0.8

# Drawdown breaker state that survives restarts (peak equity, open/closed),
# resolved under the runtime data dir like the position backstop state. A
# breaker built without ``state_path`` keeps everything in memory.
DRAWDOWN_BREAKER_STATE_PATH = 'runtime/drawdown_breaker_state.json'
DRAWDOWN_BREAKER_STATE_SCHEMA_VERSION = 1
# A stored timestamp later than now by more than this is implausible.
DRAWDOWN_BREAKER_STATE_MAX_CLOCK_SKEW = timedelta(minutes=5)
# No maximum age: the limit is peak-to-trough across restarts, so an old peak
# still counts. Delete the file with the service stopped to re-base the peak.


class _StateRejected(ValueError):
    """A stored breaker state that must not be restored; ``reason`` is logged."""

    def __init__(self, reason: str, detail: str = '') -> None:
        super().__init__(detail or reason)
        self.reason = reason


def _parse_state_ts(value: Any, field: str, now: datetime) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise _StateRejected('timestamp_invalid', f'{field}: {exc}') from exc
    if parsed.tzinfo is None:
        raise _StateRejected('timestamp_invalid', f'{field}: no timezone')
    parsed = parsed.astimezone(UTC)
    if parsed > now + DRAWDOWN_BREAKER_STATE_MAX_CLOCK_SKEW:
        raise _StateRejected('future_timestamp', f'{field}: {parsed.isoformat()}')
    return parsed


def _snapshot_account_id(account_snapshot: Any) -> str | None:
    """The broker account id already on the cycle's snapshot, if any."""
    if isinstance(account_snapshot, Mapping):
        raw = account_snapshot.get('id', account_snapshot.get('account_id'))
    else:
        raw = getattr(account_snapshot, 'id', None)
    if isinstance(raw, UUID | str):
        return str(raw).strip() or None
    return None

class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'

class SafetyLevel(Enum):
    """Safety alert level enumeration."""
    INFO = 'info'
    WARNING = 'warning'
    CRITICAL = 'critical'
    EMERGENCY = 'emergency'

class DrawdownCircuitBreaker:
    """
    Drawdown-based circuit breaker for portfolio protection.

    Monitors portfolio drawdown and automatically halts trading
    when predefined thresholds are exceeded.
    """

    def __init__(
        self,
        max_drawdown: float=None,
        recovery_threshold: float=DRAWDOWN_RESET_FRACTION,
        state_path: Path | None = None,
    ):
        """Initialize drawdown circuit breaker.

        ``recovery_threshold`` is the reset level as a fraction of the trip
        level: trip at drawdown >= ``max_drawdown``, reset only at drawdown <=
        ``max_drawdown * recovery_threshold``. The peak only ever rises; neither
        a trip nor a reset moves it.

        With ``state_path`` the peak and open/closed state are restored from
        that file on the first equity reading and rewritten when they change.
        Without it nothing touches the filesystem.
        """
        self.max_drawdown = max_drawdown or PERFORMANCE_THRESHOLDS['MAX_DRAWDOWN']
        self.recovery_threshold = recovery_threshold
        self.state = CircuitBreakerState.CLOSED
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.last_equity: float | None = None
        self.halt_timestamp: datetime | None = None
        self.peak_set_at: datetime | None = None
        self.state_path = Path(state_path) if state_path is not None else None
        self._account_id: str | None = None
        self._state_loaded = self.state_path is None
        self._saved_key: tuple[Any, ...] | None = None
        self.reset_callbacks: list[Callable[[str, dict[str, Any]], None]] = []
        emit_once(logger, 'DRAWDOWN_CB_INIT', 'info', 'DrawdownCircuitBreaker initialized', max_drawdown=self._safe_format_percentage(self.max_drawdown))

    def _recovery_drawdown_threshold(self) -> float:
        """Return the drawdown at or below which a tripped breaker resets.

        A fraction outside (0, 1) would leave no reset margin below the trip
        level, so ``DRAWDOWN_RESET_FRACTION`` is used in its place.
        """
        try:
            max_drawdown = float(self.max_drawdown)
            fraction = float(self.recovery_threshold)
        except (TypeError, ValueError):
            return 0.0
        if not (math.isfinite(fraction) and 0.0 < fraction < 1.0):
            fraction = DRAWDOWN_RESET_FRACTION
        return max(0.0, max_drawdown * fraction)

    def _transition_extra(self, reason: str) -> dict[str, Any]:
        """Structured fields shared by the TRIPPED and RESET log lines."""
        return {
            'reason': reason,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'trip_level': self.max_drawdown,
            'reset_level': self._recovery_drawdown_threshold(),
            'peak_equity': self.peak_equity,
            'peak_set_at': self.peak_set_at.isoformat() if self.peak_set_at else None,
            'equity': self.last_equity,
        }

    # -- persistence (only with ``state_path``) --
    def _state_key(self) -> tuple[Any, ...]:
        return (
            self.peak_equity,
            self.peak_set_at,
            self.state.value,
            self.halt_timestamp,
            self._account_id,
        )

    def _state_payload(self) -> dict[str, Any]:
        return {
            'schema_version': DRAWDOWN_BREAKER_STATE_SCHEMA_VERSION,
            'account_id': self._account_id,
            'peak_equity': self.peak_equity,
            'peak_set_at': self.peak_set_at.isoformat() if self.peak_set_at else None,
            'state': self.state.value,
            'tripped_at': self.halt_timestamp.isoformat() if self.halt_timestamp else None,
            'saved_at': safe_utcnow().isoformat(),
        }

    def _validated_state(self, payload: Any, account_id: str | None) -> dict[str, Any]:
        """Return the fields to restore; raise ``_StateRejected`` otherwise."""
        if not isinstance(payload, Mapping):
            raise _StateRejected('corrupt', 'payload is not an object')
        version = payload.get('schema_version')
        if isinstance(version, bool) or version != DRAWDOWN_BREAKER_STATE_SCHEMA_VERSION:
            raise _StateRejected('schema_mismatch', f'schema_version={version!r}')
        stored_account = payload.get('account_id')
        if stored_account is not None and not isinstance(stored_account, str):
            raise _StateRejected('corrupt', 'account_id is not a string')
        if stored_account and account_id and stored_account != account_id:
            raise _StateRejected('account_mismatch')
        raw_peak = payload.get('peak_equity')
        if isinstance(raw_peak, bool) or not isinstance(raw_peak, int | float):
            raise _StateRejected('peak_invalid', f'peak_equity={raw_peak!r}')
        peak = float(raw_peak)
        if not math.isfinite(peak) or peak <= 0.0:
            raise _StateRejected('peak_invalid', f'peak_equity={raw_peak!r}')
        now = safe_utcnow()
        peak_set_at = _parse_state_ts(payload.get('peak_set_at'), 'peak_set_at', now)
        try:
            state = CircuitBreakerState(payload.get('state'))
        except ValueError as exc:
            raise _StateRejected('state_invalid', str(exc)) from exc
        if state is CircuitBreakerState.HALF_OPEN:
            raise _StateRejected('state_invalid', 'half_open')
        tripped_at = None
        if state is CircuitBreakerState.OPEN:
            tripped_at = _parse_state_ts(payload.get('tripped_at'), 'tripped_at', now)
        return {
            'account_id': stored_account or None,
            'peak_equity': peak,
            'peak_set_at': peak_set_at,
            'state': state,
            'tripped_at': tripped_at,
        }

    def _ensure_state_loaded(self, account_id: str | None = None) -> None:
        """Restore the stored state once, before the first equity reading.

        A missing, unreadable or implausible file starts fresh with one
        ``DRAWDOWN_BREAKER_STATE_LOAD_FAILED`` warning; it never raises.
        """
        if account_id:
            self._account_id = account_id
        if self._state_loaded or self.state_path is None:
            return
        self._state_loaded = True
        path = self.state_path
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            restored = self._validated_state(payload, account_id)
        except FileNotFoundError:
            logger.warning('DRAWDOWN_BREAKER_STATE_LOAD_FAILED', extra={'path': str(path), 'reason': 'missing'})
            return
        except _StateRejected as exc:
            logger.warning(
                'DRAWDOWN_BREAKER_STATE_LOAD_FAILED',
                extra={'path': str(path), 'reason': exc.reason, 'error': str(exc)},
            )
            return
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                'DRAWDOWN_BREAKER_STATE_LOAD_FAILED',
                extra={'path': str(path), 'reason': 'corrupt', 'error': str(exc)},
            )
            return
        self.peak_equity = restored['peak_equity']
        self.peak_set_at = restored['peak_set_at']
        self.state = restored['state']
        self.halt_timestamp = restored['tripped_at']
        current_account = self._account_id
        self._account_id = restored['account_id']
        self._saved_key = self._state_key()  # what the file holds
        if current_account:
            self._account_id = current_account
        age = safe_utcnow() - restored['peak_set_at']
        log = logger.warning if self.state is CircuitBreakerState.OPEN else logger.info
        log(
            'DRAWDOWN_BREAKER_STATE_RESTORED',
            extra={
                'path': str(path),
                'peak_equity': self.peak_equity,
                'peak_set_at': self.peak_set_at.isoformat(),
                'peak_age_days': round(age.total_seconds() / 86400.0, 2),
                'breaker_state': self.state.value,
                'tripped_at': self.halt_timestamp.isoformat() if self.halt_timestamp else None,
            },
        )

    def _persist_if_changed(self) -> None:
        """Atomically rewrite the state file when peak, state or account changed."""
        if self.state_path is None or not self._state_loaded:
            return
        key = self._state_key()
        if key == self._saved_key:
            return
        try:
            atomic_write_text(
                self.state_path,
                json.dumps(self._state_payload(), sort_keys=True, indent=1) + '\n',
            )
        except (OSError, TypeError, ValueError) as exc:
            # The in-memory breaker still decides; the write retries next change.
            logger.warning(
                'DRAWDOWN_BREAKER_STATE_WRITE_FAILED',
                extra={'path': str(self.state_path), 'error': str(exc)},
            )
            return
        self._saved_key = key

    def _safe_format_percentage(self, value) -> str:
        """
        Safely format a value as a percentage, handling MagicMock objects during testing.

        AI-AGENT-REF: Defensive programming for test compatibility
        """
        try:
            if hasattr(value, '_mock_name'):
                return f"<Mock: {value._mock_name or 'percentage'}>"
            return f'{value:.2%}'
        except (TypeError, ValueError, AttributeError):
            return f'<{type(value).__name__}: {value}>'

    def update_equity(self, current_equity: float) -> bool:
        """
        Update current equity and check circuit breaker status.

        Args:
            current_equity: Current portfolio equity value

        Returns:
            True if trading is allowed, False if halted
        """
        try:
            if current_equity is None or not isinstance(current_equity, int | float):
                logger.warning(f'Invalid equity value: {current_equity} (type: {type(current_equity)})')
                return False
            if current_equity < 0:
                logger.warning(f'Negative equity detected: {current_equity}')
                return False
            self._ensure_state_loaded()
            self.last_equity = float(current_equity)
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
                self.peak_set_at = safe_utcnow()
            if self.peak_equity > 0:
                self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            else:
                self.current_drawdown = 0.0
            if self.current_drawdown < 0:
                logger.debug(f'Negative drawdown detected: {self.current_drawdown} (equity increased above peak)')
                self.current_drawdown = 0.0
            if self.state == CircuitBreakerState.CLOSED:
                if self.current_drawdown >= self.max_drawdown:
                    self._trigger_halt('Maximum drawdown exceeded')
                    return False
            elif self.state == CircuitBreakerState.OPEN:
                if self.current_drawdown <= self._recovery_drawdown_threshold():
                    self._reset_breaker('Drawdown recovery achieved')
                    return True
                return False
            self._persist_if_changed()
            return self.state == CircuitBreakerState.CLOSED
        except COMMON_EXC as e:
            logger.error(f'Error updating drawdown circuit breaker: {e}', exc_info=True)
            return False

    def _trigger_halt(self, reason: str):
        """Trigger circuit breaker halt."""
        try:
            self.state = CircuitBreakerState.OPEN
            self.halt_timestamp = safe_utcnow()
            logger.critical('DRAWDOWN_BREAKER_TRIPPED', extra=self._transition_extra(reason))
            self._persist_if_changed()
            for callback in self.reset_callbacks:
                try:
                    callback('halt', {'reason': reason, 'drawdown': self.current_drawdown, 'timestamp': self.halt_timestamp})
                except COMMON_EXC as e:
                    logger.error(f'Error in circuit breaker callback: {e}')
        except COMMON_EXC as e:
            logger.error(f'Error triggering drawdown halt: {e}')

    def _reset_breaker(self, reason: str):
        """Reset circuit breaker to normal operation."""
        try:
            self.state = CircuitBreakerState.CLOSED
            self.halt_timestamp = None
            logger.warning('DRAWDOWN_BREAKER_RESET', extra=self._transition_extra(reason))
            self._persist_if_changed()
            for callback in self.reset_callbacks:
                try:
                    callback('reset', {'reason': reason, 'drawdown': self.current_drawdown, 'timestamp': safe_utcnow()})
                except COMMON_EXC as e:
                    logger.error(f'Error in circuit breaker callback: {e}')
        except COMMON_EXC as e:
            logger.error(f'Error resetting drawdown breaker: {e}')

    def score_account_snapshot(self, account_snapshot: Any) -> dict[str, Any]:
        """Feed one account snapshot's equity; return the trading cycle's gate payload.

        Equity that is missing, non-numeric, non-finite or not positive is a
        failed reading, not a loss: scoring it would register a false drawdown.
        Such a reading is logged and the breaker is left exactly as it was.
        """
        raw_equity = (
            account_snapshot.get('equity')
            if isinstance(account_snapshot, Mapping)
            else getattr(account_snapshot, 'equity', None)
        )
        equity = 0.0
        skip_reason: str | None = None
        if account_snapshot is None:
            skip_reason = 'account_snapshot_unavailable'
        elif raw_equity is None:
            skip_reason = 'account_equity_missing'
        else:
            try:
                equity = float(raw_equity)
            except (TypeError, ValueError):
                skip_reason = 'account_equity_invalid'
            else:
                if isinstance(raw_equity, bool) or not math.isfinite(equity) or equity <= 0.0:
                    skip_reason = 'account_equity_invalid'
        if skip_reason is None:
            self._ensure_state_loaded(_snapshot_account_id(account_snapshot))
            self.update_equity(equity)
        else:
            logger.warning(
                'DRAWDOWN_BREAKER_FEED_SKIPPED',
                extra={'reason': skip_reason, 'breaker_state': self.state.value},
            )
        return {
            'open': self.state is CircuitBreakerState.OPEN,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'equity': equity if skip_reason is None else None,
        }

    def add_callback(self, callback: Callable[[str, dict], None]):
        """Add callback for circuit breaker events."""
        self.reset_callbacks.append(callback)

    def manual_reset(self, current_equity: float | None = None) -> bool:
        """Reset only when current metrics show drawdown has recovered safely."""
        if current_equity is not None:
            if self.update_equity(current_equity) and self.state == CircuitBreakerState.CLOSED:
                return True
        try:
            max_drawdown = float(self.max_drawdown)
        except (TypeError, ValueError):
            max_drawdown = 0.0
        if max_drawdown > 0.0 and self.current_drawdown >= max_drawdown:
            logger.warning(
                'Drawdown reset rejected: current drawdown %s at or above max drawdown %s',
                self._safe_format_percentage(self.current_drawdown),
                self._safe_format_percentage(max_drawdown),
            )
            return False
        if self.current_drawdown > self._recovery_drawdown_threshold():
            logger.warning(
                'Drawdown reset rejected: current drawdown %s above recovery threshold %s',
                self._safe_format_percentage(self.current_drawdown),
                self._safe_format_percentage(self._recovery_drawdown_threshold()),
            )
            return False
        if self.state is not CircuitBreakerState.OPEN:
            return True
        self._reset_breaker('Manual reset after drawdown recovery')
        return True

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {'state': self.state.value, 'current_drawdown': self.current_drawdown, 'max_drawdown': self.max_drawdown, 'recovery_drawdown': self._recovery_drawdown_threshold(), 'peak_equity': self.peak_equity, 'halt_timestamp': self.halt_timestamp, 'trading_allowed': self.state == CircuitBreakerState.CLOSED}

class VolatilityCircuitBreaker:
    """
    Volatility-based circuit breaker for extreme market conditions.

    Monitors market volatility and automatically reduces or halts
    trading during periods of extreme volatility.
    """

    def __init__(self, high_vol_threshold: float=0.5, extreme_vol_threshold: float=1.0):
        """Initialize volatility circuit breaker."""
        self.high_vol_threshold = high_vol_threshold
        self.extreme_vol_threshold = extreme_vol_threshold
        self.state = CircuitBreakerState.CLOSED
        self.current_volatility = 0.0
        self.position_size_multiplier = 1.0
        self.last_volatility_update = safe_utcnow()
        logger.info(f'VolatilityCircuitBreaker initialized with thresholds: high={self._safe_format_percentage(high_vol_threshold)}, extreme={self._safe_format_percentage(extreme_vol_threshold)}')

    def _safe_format_percentage(self, value) -> str:
        """
        Safely format a value as a percentage, handling MagicMock objects during testing.

        AI-AGENT-REF: Defensive programming for test compatibility
        """
        try:
            if hasattr(value, '_mock_name'):
                return f"<Mock: {value._mock_name or 'percentage'}>"
            return f'{value:.1%}'
        except (TypeError, ValueError, AttributeError):
            return f'<{type(value).__name__}: {value}>'

    def update_volatility(self, volatility: float) -> dict[str, Any]:
        """
        Update current volatility and adjust trading parameters.

        Args:
            volatility: Current annualized volatility (e.g., 0.2 for 20%)

        Returns:
            Dictionary with trading adjustments
        """
        try:
            self.current_volatility = volatility
            self.last_volatility_update = safe_utcnow()
            if volatility >= self.extreme_vol_threshold:
                self.state = CircuitBreakerState.OPEN
                self.position_size_multiplier = 0.0
                status = 'EXTREME_VOLATILITY_HALT'
                logger.critical(f'EXTREME VOLATILITY DETECTED: {self._safe_format_percentage(volatility)} - Trading halted')
            elif volatility >= self.high_vol_threshold:
                self.state = CircuitBreakerState.HALF_OPEN
                reduction_factor = (volatility - self.high_vol_threshold) / (self.extreme_vol_threshold - self.high_vol_threshold)
                self.position_size_multiplier = max(0.1, 1.0 - reduction_factor * 0.8)
                status = 'HIGH_VOLATILITY_REDUCTION'
                logger.warning(f'HIGH VOLATILITY: {self._safe_format_percentage(volatility)} - Position sizes reduced to {self._safe_format_percentage(self.position_size_multiplier)}')
            else:
                self.state = CircuitBreakerState.CLOSED
                self.position_size_multiplier = 1.0
                status = 'NORMAL_OPERATION'
                logger.debug(f'Normal volatility: {self._safe_format_percentage(volatility)}')
            return {'status': status, 'trading_allowed': self.state != CircuitBreakerState.OPEN, 'position_size_multiplier': self.position_size_multiplier, 'volatility': volatility, 'state': self.state.value}
        except COMMON_EXC as e:
            logger.error(f'Error updating volatility circuit breaker: {e}')
            return {'status': 'ERROR', 'trading_allowed': False, 'position_size_multiplier': 0.0, 'volatility': 0.0, 'state': 'error'}

    def get_status(self) -> dict[str, Any]:
        """Get current volatility circuit breaker status."""
        return {'state': self.state.value, 'current_volatility': self.current_volatility, 'high_vol_threshold': self.high_vol_threshold, 'extreme_vol_threshold': self.extreme_vol_threshold, 'position_size_multiplier': self.position_size_multiplier, 'last_update': self.last_volatility_update, 'trading_allowed': self.state != CircuitBreakerState.OPEN}

class TradingHaltManager:
    """
    Comprehensive trading halt management system.

    Coordinates multiple circuit breakers and safety mechanisms
    to provide unified trading halt control.
    """

    def __init__(self):
        """Initialize trading halt manager."""
        self.drawdown_breaker = DrawdownCircuitBreaker()
        self.volatility_breaker = VolatilityCircuitBreaker()
        self.manual_halt = False
        self.manual_halt_reason = ''
        self.manual_halt_timestamp: datetime | None = None
        self.daily_trade_count = 0
        self.daily_loss_amount = 0.0
        self.max_daily_trades = 1000
        self.max_daily_loss = 0.05
        self.emergency_stop = False
        self.emergency_callbacks: list[Callable[[str], None]] = []
        self._lock = threading.RLock()
        logger.info('TradingHaltManager initialized')

    def _safe_format_percentage(self, value) -> str:
        """
        Safely format a value as a percentage, handling MagicMock objects during testing.

        AI-AGENT-REF: Defensive programming for test compatibility
        """
        try:
            if hasattr(value, '_mock_name'):
                return f"<Mock: {value._mock_name or 'percentage'}>"
            return f'{value:.2%}'
        except (TypeError, ValueError, AttributeError):
            return f'<{type(value).__name__}: {value}>'

    def is_trading_allowed(self) -> dict[str, Any]:
        """
        Check if trading is currently allowed.

        Returns:
            Dictionary with trading status and reasons
        """
        try:
            with self._lock:
                status: dict[str, Any] = {'trading_allowed': True, 'reasons': [], 'position_size_multiplier': 1.0, 'circuit_breakers': {}}
                if self.manual_halt:
                    status['trading_allowed'] = False
                    status['reasons'].append(f'Manual halt: {self.manual_halt_reason}')
                if self.emergency_stop:
                    status['trading_allowed'] = False
                    status['reasons'].append('Emergency stop activated')
                drawdown_status = self.drawdown_breaker.get_status()
                status['circuit_breakers']['drawdown'] = drawdown_status
                if not drawdown_status['trading_allowed']:
                    status['trading_allowed'] = False
                    status['reasons'].append(f"Drawdown limit exceeded: {self._safe_format_percentage(drawdown_status['current_drawdown'])}")
                volatility_status = self.volatility_breaker.get_status()
                status['circuit_breakers']['volatility'] = volatility_status
                if not volatility_status['trading_allowed']:
                    status['trading_allowed'] = False
                    status['reasons'].append(f"Extreme volatility: {self._safe_format_percentage(volatility_status['current_volatility'])}")
                else:
                    status['position_size_multiplier'] *= volatility_status['position_size_multiplier']
                if self.daily_trade_count >= self.max_daily_trades:
                    status['trading_allowed'] = False
                    status['reasons'].append(f'Daily trade limit exceeded: {self.daily_trade_count}')
                if self.daily_loss_amount >= self.max_daily_loss:
                    status['trading_allowed'] = False
                    status['reasons'].append(f'Daily loss limit exceeded: {self._safe_format_percentage(self.daily_loss_amount)}')
                return status
        except COMMON_EXC as e:
            logger.error(f'Error checking trading status: {e}')
            return {'trading_allowed': False, 'reasons': [f'System error: {e}'], 'position_size_multiplier': 0.0, 'circuit_breakers': {}}

    def update_equity(self, current_equity: float):
        """Update equity for drawdown monitoring."""
        try:
            with self._lock:
                self.drawdown_breaker.update_equity(current_equity)
        except COMMON_EXC as e:
            logger.error(f'Error updating equity in halt manager: {e}')

    def update_volatility(self, volatility: float):
        """Update volatility for volatility monitoring."""
        try:
            with self._lock:
                return self.volatility_breaker.update_volatility(volatility)
        except COMMON_EXC as e:
            logger.error(f'Error updating volatility in halt manager: {e}')
            return {'status': 'ERROR', 'trading_allowed': False}

    def manual_halt_trading(self, reason: str):
        """Manually halt trading with reason."""
        try:
            with self._lock:
                self.manual_halt = True
                self.manual_halt_reason = reason
                self.manual_halt_timestamp = safe_utcnow()
                logger.critical(f'MANUAL TRADING HALT: {reason}')
        except COMMON_EXC as e:
            logger.error(f'Error setting manual halt: {e}')

    def resume_trading(self, reason: str='Manual resume'):
        """Resume trading from manual halt."""
        try:
            with self._lock:
                self.manual_halt = False
                self.manual_halt_reason = ''
                self.manual_halt_timestamp = None
                logger.info(f'TRADING RESUMED: {reason}')
        except COMMON_EXC as e:
            logger.error(f'Error resuming trading: {e}')

    def emergency_stop_all(self, reason: str='Emergency stop'):
        """Activate emergency stop - highest priority halt."""
        try:
            with self._lock:
                self.emergency_stop = True
                logger.critical(f'EMERGENCY STOP ACTIVATED: {reason}')
                for callback in self.emergency_callbacks:
                    try:
                        callback(reason)
                    except COMMON_EXC as e:
                        logger.error(f'Error in emergency callback: {e}')
        except COMMON_EXC as e:
            logger.error(f'Error activating emergency stop: {e}')

    def reset_emergency_stop(self, reason: str='Manual reset'):
        """Reset emergency stop (requires manual intervention)."""
        try:
            with self._lock:
                self.emergency_stop = False
                logger.info(f'EMERGENCY STOP RESET: {reason}')
        except COMMON_EXC as e:
            logger.error(f'Error resetting emergency stop: {e}')

    def record_trade(self, trade_pnl: float=0.0):
        """Record a trade for daily limit tracking."""
        try:
            with self._lock:
                self.daily_trade_count += 1
                if trade_pnl < 0:
                    self.daily_loss_amount += abs(trade_pnl)
                logger.debug(f'Trade recorded: count={self.daily_trade_count}, daily_loss={self.daily_loss_amount:.4f}')
        except COMMON_EXC as e:
            logger.error(f'Error recording trade: {e}')

    def reset_daily_counters(self):
        """Reset daily counters (typically called at market open)."""
        try:
            with self._lock:
                self.daily_trade_count = 0
                self.daily_loss_amount = 0.0
                logger.info('Daily trading counters reset')
        except COMMON_EXC as e:
            logger.error(f'Error resetting daily counters: {e}')

    def add_emergency_callback(self, callback: Callable[[str], None]):
        """Add callback for emergency stop events."""
        self.emergency_callbacks.append(callback)

    def get_comprehensive_status(self) -> dict[str, Any]:
        """Get comprehensive status of all safety systems."""
        try:
            with self._lock:
                trading_status = self.is_trading_allowed()
                return {'timestamp': safe_utcnow(), 'trading_status': trading_status, 'manual_controls': {'manual_halt': self.manual_halt, 'manual_halt_reason': self.manual_halt_reason, 'manual_halt_timestamp': self.manual_halt_timestamp, 'emergency_stop': self.emergency_stop}, 'daily_limits': {'trade_count': self.daily_trade_count, 'max_daily_trades': self.max_daily_trades, 'daily_loss_amount': self.daily_loss_amount, 'max_daily_loss': self.max_daily_loss}, 'circuit_breakers': trading_status['circuit_breakers']}
        except COMMON_EXC as e:
            logger.error(f'Error getting comprehensive status: {e}')
            return {'error': str(e)}

class DeadMansSwitch:
    """
    Dead man's switch for automated system monitoring.

    Requires periodic heartbeat signals and automatically triggers
    emergency procedures if heartbeat is not received within timeout.
    """

    def __init__(self, timeout_seconds: int=300):
        """Initialize dead man's switch."""
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = safe_utcnow()
        self.is_active = False
        self.emergency_callbacks: list[Callable[[str], None]] = []
        self.monitoring_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        logger.info(f'DeadMansSwitch initialized with timeout={timeout_seconds}s')

    def start_monitoring(self):
        """Start the dead man's switch monitoring."""
        try:
            if self.is_active:
                logger.warning("Dead man's switch already active")
                return
            self.is_active = True
            self.last_heartbeat = safe_utcnow()
            self._stop_event.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True, name='DeadMansSwitch')
            self.monitoring_thread.start()
            logger.info("Dead man's switch monitoring started")
        except COMMON_EXC as e:
            logger.error(f"Error starting dead man's switch: {e}")

    def stop_monitoring(self):
        """Stop the dead man's switch monitoring."""
        try:
            if not self.is_active:
                return
            self.is_active = False
            self._stop_event.set()
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Dead man's switch monitoring stopped")
        except COMMON_EXC as e:
            logger.error(f"Error stopping dead man's switch: {e}")

    def heartbeat(self):
        """Send heartbeat signal to reset the timer."""
        try:
            if self.is_active:
                self.last_heartbeat = safe_utcnow()
                logger.debug("Dead man's switch heartbeat received")
        except COMMON_EXC as e:
            logger.error(f'Error processing heartbeat: {e}')

    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        try:
            while self.is_active and (not self._stop_event.is_set()):
                current_time = safe_utcnow()
                time_since_heartbeat = (current_time - self.last_heartbeat).total_seconds()
                if time_since_heartbeat > self.timeout_seconds:
                    logger.critical(f"DEAD MAN'S SWITCH TRIGGERED - No heartbeat for {time_since_heartbeat:.0f}s")
                    self._trigger_emergency()
                    break
                self._stop_event.wait(10.0)
        except COMMON_EXC as e:
            logger.error(f"Error in dead man's switch monitoring loop: {e}")

    def _trigger_emergency(self):
        """Trigger emergency procedures."""
        try:
            for callback in self.emergency_callbacks:
                try:
                    callback("Dead man's switch timeout")
                except COMMON_EXC as e:
                    logger.error(f"Error in dead man's switch callback: {e}")
            self.is_active = False
        except COMMON_EXC as e:
            logger.error(f"Error triggering dead man's switch emergency: {e}")

    def add_emergency_callback(self, callback: Callable[[str], None]):
        """Add callback for emergency trigger events."""
        self.emergency_callbacks.append(callback)

    def get_status(self) -> dict[str, Any]:
        """Get current dead man's switch status."""
        try:
            current_time = safe_utcnow()
            time_since_heartbeat = (current_time - self.last_heartbeat).total_seconds()
            return {'is_active': self.is_active, 'last_heartbeat': self.last_heartbeat, 'time_since_heartbeat': time_since_heartbeat, 'timeout_seconds': self.timeout_seconds, 'status': 'OK' if time_since_heartbeat < self.timeout_seconds else 'TIMEOUT', 'monitoring_thread_alive': self.monitoring_thread.is_alive() if self.monitoring_thread else False}
        except COMMON_EXC as e:
            logger.error(f"Error getting dead man's switch status: {e}")
            return {'error': str(e)}
