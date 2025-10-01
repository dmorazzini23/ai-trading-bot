from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from types import GeneratorType
from typing import Any, Dict, Tuple

from ai_trading.data.fetch.normalize import normalize_ohlcv_df
from ai_trading.utils.time import is_generator_stop


@dataclass(slots=True)
class _MemoEntry:
    ts: float
    key: Tuple[str, str, str, str]
    df: Any


_MEMO: Dict[Tuple[str, str, str, str], _MemoEntry] = {}
_TTL_S = 60.0


def get_daily_df_memoized(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    df_or_factory: Any,
) -> Any:
    """Return cached daily DataFrame within TTL, applying normalization once."""

    key = (symbol, timeframe, start, end)
    now = time.time()
    entry = _MEMO.get(key)
    if entry and (now - entry.ts) < _TTL_S:
        return entry.df

    last_good_df = entry.df if entry else None

    factory_callable = callable(df_or_factory)

    def _invoke_factory() -> Any:
        if not factory_callable:
            return df_or_factory
        try:
            return df_or_factory(symbol, timeframe, start, end)
        except TypeError:
            return df_or_factory()

    attempts = 0
    while True:
        attempts += 1
        stop_iteration = False
        try:
            candidate = _invoke_factory()
        except StopIteration:
            stop_iteration = True
            candidate = None
        except RuntimeError as exc:
            if is_generator_stop(exc):
                stop_iteration = True
                candidate = None
            else:
                raise

        if isinstance(candidate, GeneratorType):
            generator = candidate
            try:
                candidate = next(generator)
            except StopIteration:
                stop_iteration = True
                candidate = None
            except RuntimeError as exc:
                if is_generator_stop(exc):
                    stop_iteration = True
                    candidate = None
                else:
                    raise
            finally:
                with suppress(Exception):
                    generator.close()

        if stop_iteration:
            _MEMO.pop(key, None)
            if (not factory_callable) or attempts >= 2:
                if last_good_df is not None:
                    refreshed_at = time.time()
                    _MEMO[key] = _MemoEntry(refreshed_at, key, last_good_df)
                    return last_good_df
                return None
            continue

        df = candidate
        break

    try:
        normalized = normalize_ohlcv_df(df)
    except Exception:
        normalized = df

    refreshed_at = time.time()
    _MEMO[key] = _MemoEntry(refreshed_at, key, normalized)
    return normalized


def clear_memo() -> None:
    """Reset memoized values; primarily for tests."""

    _MEMO.clear()


__all__ = ["get_daily_df_memoized", "clear_memo"]
