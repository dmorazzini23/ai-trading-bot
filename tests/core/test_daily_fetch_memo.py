from __future__ import annotations

import pytest

from ai_trading.core.daily_fetch_memo import clear_memo, get_daily_df_memoized


class _Frame:
    pass


def test_daily_fetch_memo_supports_no_arg_factory() -> None:
    clear_memo()
    frame = _Frame()

    def factory() -> _Frame:
        return frame

    result = get_daily_df_memoized("AAPL", "1Day", "2026-01-01", "2026-01-02", factory)

    assert result is frame


def test_daily_fetch_memo_does_not_retry_internal_type_error_as_no_arg() -> None:
    clear_memo()

    def factory(_symbol: str, _timeframe: str, _start: str, _end: str) -> _Frame:
        raise TypeError("factory body failed")

    with pytest.raises(TypeError, match="factory body failed"):
        get_daily_df_memoized("AAPL", "1Day", "2026-01-01", "2026-01-02", factory)
