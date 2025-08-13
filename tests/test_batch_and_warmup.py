import pytest
from datetime import datetime, timedelta, timezone
try:
    from ai_trading.data_fetcher import get_bars_batch, warmup_cache
except Exception:
    pytest.skip("data_fetcher deps missing", allow_module_level=True)


def test_get_bars_batch_handles_empty_list():
    out = get_bars_batch([], "1D", "2024-01-01", "2024-01-31")
    assert out == {}


def test_warmup_cache_no_symbols():
    n = warmup_cache([], "1D", datetime.now(timezone.utc)-timedelta(days=1), datetime.now(timezone.utc))
    assert n == 0