from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.data.fetch import get_bars

# Lazy pandas proxy for on-demand import
pd = load_pandas()
__all__ = ['check_data_freshness', 'get_stale_symbols', 'validate_trading_data', 'emergency_data_check', 'is_valid_ohlcv', 'validate_trade_log_integrity', 'monitor_real_time_data_quality', 'MarketDataValidator', 'ValidationSeverity']
REQUIRED_PRICE_COLS = ('open', 'high', 'low', 'close', 'volume')

def is_valid_ohlcv(df: pd.DataFrame, min_rows: int=50) -> bool:
    """Return True if ``df`` has required OHLCV columns and rows."""
    if df is None or df.empty:
        return False
    if not set(REQUIRED_PRICE_COLS).issubset(df.columns):
        return False
    return len(df) >= min_rows

def check_data_freshness(df: pd.DataFrame | None, symbol: str, *, max_staleness_minutes: int=15) -> dict[str, float | str | bool]:
    """Return freshness info for ``symbol`` handling empty/naive data."""
    if df is None or getattr(df, 'empty', True):
        return {'symbol': symbol, 'is_fresh': False, 'minutes_stale': float('inf')}
    try:
        last_ts = df.index[-1]
        if not isinstance(last_ts, datetime):
            last_ts = datetime.fromtimestamp(float(last_ts), tz=UTC)
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=UTC)
        age = datetime.now(UTC) - last_ts.astimezone(UTC)
        minutes = age.total_seconds() / 60.0
        return {'symbol': symbol, 'is_fresh': minutes <= max_staleness_minutes, 'minutes_stale': minutes}
    except (ValueError, TypeError):
        return {'symbol': symbol, 'is_fresh': False, 'minutes_stale': float('inf')}

def get_stale_symbols(data_map: Mapping[str, object] | None, *, max_staleness_minutes: int=15) -> list[str]:
    """Return symbols whose data is stale.

    Accepts either ``{sym: DataFrame}`` or ``{sym: info mapping}``.
    """
    out: list[str] = []
    for sym, info in (data_map or {}).items():
        if isinstance(info, Mapping):
            fresh = bool(info.get('trading_ready', info.get('is_fresh')))
        else:
            fresh = bool(check_data_freshness(info, sym, max_staleness_minutes=max_staleness_minutes)['is_fresh'])
        if not fresh:
            out.append(sym)
    return out

def validate_trading_data(data_map: Mapping[str, pd.DataFrame] | None, *, max_staleness_minutes: int=15) -> dict[str, dict[str, object]]:
    """Return mapping of freshness and trading readiness."""
    data_map = data_map or {}
    results: dict[str, dict[str, object]] = {}
    for sym, df in data_map.items():
        info = check_data_freshness(df, sym, max_staleness_minutes=max_staleness_minutes)
        info['trading_ready'] = bool(info.get('is_fresh'))
        results[sym] = info
    return results

def emergency_data_check(symbols_or_df: Sequence[str] | str | pd.DataFrame, symbol: str | None=None, *, fetcher: Callable[[str, str, datetime, datetime], pd.DataFrame] | None=None) -> bool:
    """Return True if any symbol yields non-empty recent bars.

    Back-compat: ``emergency_data_check(df, "AAPL")`` returns ``not df.empty``.
    """
    if isinstance(symbols_or_df, pd.DataFrame) and isinstance(symbol, str):
        return not symbols_or_df.empty
    if isinstance(symbols_or_df, str | bytes):
        to_check = [symbols_or_df]
    elif isinstance(symbols_or_df, Sequence):
        to_check = list(symbols_or_df)
    else:
        to_check = [str(symbols_or_df)]
    fetch = fetcher or get_bars
    end = datetime.now(UTC)
    start = end - timedelta(minutes=1)
    for sym in to_check:
        try:
            df = fetch(sym, '1Min', start, end)
            if df is not None and (not df.empty):
                return True
        except (ValueError, TypeError):
            continue
    return False

class ValidationSeverity(Enum):
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

@dataclass
class ValidationResult:
    is_valid: bool
    data_quality_score: float
    severity: ValidationSeverity

class MarketDataValidator:
    """Minimal market data validator used in tests."""

    def validate_ohlc_data(self, df: pd.DataFrame, symbol: str) -> ValidationResult:
        if df is None or df.empty:
            return ValidationResult(False, 0.0, ValidationSeverity.ERROR)
        cond = (df['high'] >= df[['open', 'close']].max(axis=1)) & (df['low'] <= df[['open', 'close']].min(axis=1)) & (df['volume'] >= 0)
        ok = bool(cond.all())
        score = 1.0 if ok else 0.0
        sev = ValidationSeverity.INFO if ok else ValidationSeverity.ERROR
        return ValidationResult(ok, score, sev)

    @staticmethod
    def positive_prices(df: pd.DataFrame) -> pd.DataFrame:
        return df[df['close'] > 0]

def monitor_real_time_data_quality(prices: dict[str, float]) -> dict[str, Any]:
    """Check price mapping for non-positive values."""
    bad = [sym for sym, price in prices.items() if price <= 0]
    return {'data_quality_ok': not bad, 'critical_symbols': bad, 'anomalies_detected': bad}

def validate_trade_log_integrity(path: str | Path) -> dict[str, Any]:
    """Validate CSV trade log and return integrity report."""
    p = Path(path)
    report: dict[str, Any] = {'file_exists': p.exists(), 'file_readable': False, 'valid_format': False, 'data_consistent': False, 'total_trades': 0, 'integrity_score': 0.0, 'corrupted_rows': []}
    if not report['file_exists']:
        return report
    try:
        df = pd.read_csv(p)
        report['file_readable'] = True
    except (ValueError, TypeError):
        return report
    required = {'timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'pnl'}
    report['valid_format'] = required.issubset(df.columns)
    if not report['valid_format']:
        return report
    report['total_trades'] = len(df)
    corrupted: list[int] = []
    for idx, row in df.iterrows():
        try:
            pd.to_datetime(row['timestamp'])
            float(row['entry_price'])
            float(row['exit_price'])
            int(row['quantity'])
            float(row['pnl'])
        except (ValueError, TypeError):
            corrupted.append(idx)
    report['corrupted_rows'] = corrupted
    report['data_consistent'] = not corrupted
    report['integrity_score'] = 1.0 - len(corrupted) / len(df) if len(df) else 0.0
    return report