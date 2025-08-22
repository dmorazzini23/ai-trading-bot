from datetime import UTC, datetime

import pytest
from ai_trading.data.timeutils import ensure_utc_datetime


def test_callable_rejected():
    def cb():
        return datetime.now(UTC)

    with pytest.raises(TypeError):
        ensure_utc_datetime(cb)


def test_callable_allowed():
    dt = datetime(2023, 1, 1, tzinfo=UTC)

    def cb():
        return dt

    out = ensure_utc_datetime(cb, allow_callables=True)
    assert out == dt
