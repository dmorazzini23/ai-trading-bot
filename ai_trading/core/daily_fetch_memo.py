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

    exhaustions = 0
    _RETRY = object()

    def _handle_exhaustion(*, terminal: bool = False) -> Any:
        nonlocal exhaustions
        exhaustions += 1
        _MEMO.pop(key, None)
        if terminal or (not factory_callable) or exhaustions >= 2:
            if last_good_df is not None:
                refreshed_at = time.time()
                _MEMO[key] = _MemoEntry(refreshed_at, key, last_good_df)
                return last_good_df
            return None
        return _RETRY

    def _is_empty_frame(candidate: Any) -> bool:
        if candidate is None:
            return True

        empty_attr = getattr(candidate, "empty", None)
        if isinstance(empty_attr, bool):
            return empty_attr
        if callable(empty_attr):
            try:
                empty_value = empty_attr()
            except Exception:
                empty_value = None
            if isinstance(empty_value, bool):
                return empty_value

        if hasattr(candidate, "__len__"):
            try:
                return len(candidate) == 0
            except TypeError:
                return False
        return False

    while True:
        try:
            candidate = _invoke_factory()
        except StopIteration:
            result = _handle_exhaustion()
            if result is _RETRY:
                continue
            return result
        except RuntimeError as exc:
            if is_generator_stop(exc):
                result = _handle_exhaustion()
                if result is _RETRY:
                    continue
                return result
            else:
                raise

        if isinstance(candidate, GeneratorType):
            generator = candidate
            try:
                candidate = next(generator)
            except StopIteration:
                result = _handle_exhaustion()
                if result is _RETRY:
                    continue
                return result
            except RuntimeError as exc:
                if is_generator_stop(exc):
                    result = _handle_exhaustion()
                    if result is _RETRY:
                        continue
                    return result
                else:
                    raise
            finally:
                with suppress(Exception):
                    generator.close()

        if _is_empty_frame(candidate):
            result = _handle_exhaustion(terminal=True)
            if result is _RETRY:
                continue
            return result

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
