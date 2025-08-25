import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from ai_trading.logging import get_logger
try:
    from ai_trading.capital_scaling import drawdown_adjusted_kelly_alt as drawdown_adjusted_kelly
except (ValueError, TypeError):
    from ai_trading.capital_scaling import drawdown_adjusted_kelly
from ai_trading.config.settings import get_settings
from ai_trading.core.bot_engine import _fetch_intraday_bars_chunked
log = get_logger(__name__)

def should_enter_trade(price_data, signals, risk_params):
    """Determine whether a trade entry conditions are met."""
    if not isinstance(signals, dict) or not isinstance(risk_params, dict):
        log.warning('Invalid input types for trade entry evaluation')
        return False
    try:
        if price_data is None or len(price_data) < 2:
            log.debug('Insufficient price data for trade entry')
            return False
        last_price, prev_price = (float(price_data[-1]), float(price_data[-2]))
        recent_gain = (last_price - prev_price) / max(prev_price, 1e-09)
    except (ValueError, TypeError):
        log.warning('Failed to calculate recent gain from price data')
        return False
    signal_strength = signals.get('signal_strength', 0)
    max_risk = risk_params.get('max_risk', 0.02)
    try:
        signal_strength = float(signal_strength)
        max_risk = float(max_risk)
    except (ValueError, TypeError):
        log.warning('Invalid signal_strength or max_risk values')
        return False
    result = signal_strength > 0.7 and recent_gain > 0.001 and (max_risk < 0.05)
    log.debug('Trade entry evaluation: signal=%.3f, gain=%.4f, risk=%.3f, result=%s', signal_strength, recent_gain, max_risk, result)
    return result

def extract_price(data: Any) -> float:
    """Return the last price from various data structures."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        if data is None:
            logger.warning('extract_price called with None; returning fallback value')
            return 0.001
        if hasattr(data, 'iloc'):
            if 'close' in data.columns and (not data.empty):
                val = data['close'].iloc[-1]
            else:
                logger.warning("extract_price: DataFrame missing 'close' column or empty; using fallback")
                return 0.001
        elif isinstance(data, dict):
            val = data.get('close') or data.get('price')
            if val is None:
                logger.warning("extract_price: dict missing 'close'/'price'; using fallback")
                return 0.001
        elif isinstance(data, Sequence):
            if not data:
                logger.warning('extract_price: empty sequence; using fallback')
                return 0.001
            val = data[-1]
        else:
            val = float(data)
        return float(val or 0.001)
    except (ValueError, TypeError) as exc:
        logger.warning('extract_price failed: %s', exc)
        return 0.001

def compute_order_price(symbol_data):
    raw_price = extract_price(symbol_data)
    price = max(raw_price, 0.001)
    _, slipped = simulate_execution(price, 1)
    return slipped

def simulate_execution(price: float, qty: int) -> tuple[int, float]:
    """Return filled quantity and price after slippage and partial fill."""
    if qty <= 0 or price <= 0:
        return (0, price)
    slippage = random.uniform(-0.0002, 0.0002)
    fill_price = price * (1 + slippage)
    fill_ratio = random.uniform(0.9, 1.0)
    filled_qty = max(1, int(qty * fill_ratio))
    return (filled_qty, fill_price)

def pyramiding_logic(current_position: float, profit_in_atr: float, base_size: float) -> float:
    """Return new position size applying pyramiding rules."""
    if profit_in_atr > 1.0 and current_position < 2 * base_size:
        from ai_trading.telemetry import metrics_logger
        new_pos = current_position + 0.25 * base_size
        metrics_logger.log_pyramid_add('generic', new_pos)
        return new_pos
    return current_position

def execute_trade(signal: int, position_size: float, price: float, equity_peak: float, account_value: float, raw_kelly: float) -> None:
    adj_kelly = drawdown_adjusted_kelly(account_value, equity_peak, raw_kelly)
    final_size = position_size * adj_kelly
    if signal == 1:
        log.info('BUY %s at %s', final_size, price)
    elif signal == -1:
        log.info('SELL %s at %s', final_size, price)
    else:
        log.info('HOLD')

def evaluate_entries(ctx, candidates):
    """
    Compute entry signals using 1-Min data via chunked batch with fallback.
    """
    settings = get_settings()
    lookback_min = max(5, int(getattr(settings, 'intraday_lookback_minutes', 120)))
    end_ts = getattr(ctx, 'intraday_end', None) or datetime.now(UTC)
    start_ts = getattr(ctx, 'intraday_start', None) or end_ts - timedelta(minutes=lookback_min)
    frames = _fetch_intraday_bars_chunked(ctx, candidates, start=start_ts, end=end_ts, feed=getattr(ctx, 'data_feed', None))
    signals = {}
    for sym in candidates:
        df = frames.get(sym)
        if df is None or getattr(df, 'empty', False):
            continue
        try:
            sig = _compute_entry_signal(ctx, sym, df)
            if sig is not None:
                signals[sym] = sig
        except (ValueError, TypeError) as exc:
            ctx.logger.warning('Entry eval failed for %s: %s', sym, exc)
    return signals

def evaluate_exits(ctx, open_positions):
    """
    Compute exit signals using 1-Min data via chunked batch with fallback.
    """
    syms = list(open_positions) if isinstance(open_positions, list | set | tuple) else list(open_positions.keys())
    if not syms:
        return {}
    settings = get_settings()
    lookback_min = max(5, int(getattr(settings, 'intraday_lookback_minutes', 120)))
    end_ts = getattr(ctx, 'intraday_end', None) or datetime.now(UTC)
    start_ts = getattr(ctx, 'intraday_start', None) or end_ts - timedelta(minutes=lookback_min)
    frames = _fetch_intraday_bars_chunked(ctx, syms, start=start_ts, end=end_ts, feed=getattr(ctx, 'data_feed', None))
    exits = {}
    for sym in syms:
        df = frames.get(sym)
        if df is None or getattr(df, 'empty', False):
            continue
        try:
            sig = _compute_exit_signal(ctx, sym, df)
            if sig:
                exits[sym] = sig
        except (ValueError, TypeError) as exc:
            ctx.logger.warning('Exit eval failed for %s: %s', sym, exc)
    return exits

def _compute_entry_signal(ctx, symbol, df):
    """Placeholder for entry signal computation."""
    return {'buy': True}

def _compute_exit_signal(ctx, symbol, df):
    """Placeholder for exit signal computation."""
    return {'sell': True}
