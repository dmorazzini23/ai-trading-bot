from __future__ import annotations

import math
import sys
import types
import time

import pytest

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.nan = float("nan")
    numpy_stub.NaN = numpy_stub.nan
    numpy_stub.bool_ = bool
    numpy_stub.float64 = float

    def _isscalar(obj):
        return isinstance(obj, (bool, int, float, complex))

    numpy_stub.isscalar = _isscalar

    def _isfinite(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    numpy_stub.isfinite = _isfinite

    def _random_default(*_args, **_kwargs):
        return 0.5

    def _normal(loc=0.0, scale=1.0, size=None):
        if size is None:
            return loc
        return [loc] * (size if isinstance(size, int) else 1)

    def _uniform(a=0.0, b=1.0, size=None):
        midpoint = (float(a) + float(b)) / 2.0
        if size is None:
            return midpoint
        return [midpoint] * (size if isinstance(size, int) else 1)

    def _randn(*shape):
        if not shape:
            return 0.0
        count = 1
        for dim in shape:
            try:
                count *= int(dim)
            except (TypeError, ValueError):
                count = 1
                break
        return [0.0] * count

    def _exponential(scale=1.0, size=None):
        if size is None:
            return scale
        return [scale] * (size if isinstance(size, int) else 1)

    def _choice(seq, *args, **kwargs):
        if isinstance(seq, (list, tuple)) and seq:
            return seq[0]
        return None

    numpy_stub.random = types.SimpleNamespace(
        seed=lambda *_args, **_kwargs: None,
        random=_random_default,
        normal=_normal,
        uniform=_uniform,
        randn=_randn,
        exponential=_exponential,
        choice=_choice,
    )
    numpy_stub.randn = _randn
    numpy_stub.array = lambda data, *args, **kwargs: list(data)
    numpy_stub.asarray = lambda data, *args, **kwargs: list(data)
    sys.modules["numpy"] = numpy_stub

if "portalocker" not in sys.modules:
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.Lock = object
    portalocker_stub.LockException = RuntimeError
    portalocker_stub.LockingException = RuntimeError
    portalocker_stub.unlock = lambda *_args, **_kwargs: None
    portalocker_stub.lock = lambda *_args, **_kwargs: None
    sys.modules["portalocker"] = portalocker_stub

if "ai_trading.core.bot_engine" not in sys.modules:
    bot_engine_stub = types.ModuleType("ai_trading.core.bot_engine")

    def _noop(*_args, **_kwargs):
        return None

    bot_engine_stub.submit_order = _noop
    bot_engine_stub.safe_submit_order = _noop
    bot_engine_stub.execute_exit = _noop
    sys.modules["ai_trading.core.bot_engine"] = bot_engine_stub

if "bs4" not in sys.modules:
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:
        def __init__(self, *_args, **_kwargs):
            self.text = ""

        def find(self, *_args, **_kwargs):
            return None

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub


def test_poll_order_fill_status_returns_on_filled(monkeypatch):
    from ai_trading.core.execution_flow import poll_order_fill_status

    # Fake API that transitions statuses over successive calls
    class _Order:
        def __init__(self, status: str, filled_qty: str = "0"):
            self.status = status
            self.filled_qty = filled_qty

    class _API:
        def __init__(self):
            self._i = 0

        def get_order(self, _order_id):
            self._i += 1
            if self._i == 1:
                return _Order("new")
            if self._i == 2:
                return _Order("partially_filled", "10")
            return _Order("filled", "100")

    class _Ctx:
        def __init__(self):
            self.api = _API()

    ctx = _Ctx()
    t0 = time.time()
    poll_order_fill_status(ctx, "oid-1", timeout=2)
    elapsed = time.time() - t0
    # Should return quickly (well under timeout) once status becomes filled
    assert elapsed < 2


def test_poll_order_fill_status_coerces_numeric_fields():
    from ai_trading.core.execution_flow import poll_order_fill_status

    class _Order:
        def __init__(self):
            self.status = "filled"
            self.qty = "7"
            self.filled_qty = "3.5"

    class _API:
        def __init__(self):
            self.order = _Order()

        def get_order(self, _order_id):
            return self.order

    ctx = types.SimpleNamespace(api=_API())

    poll_order_fill_status(ctx, "oid-2", timeout=0.1)

    coerced_order = ctx.api.order
    assert isinstance(coerced_order.qty, float)
    assert coerced_order.qty == pytest.approx(7.0)
    assert isinstance(coerced_order.filled_qty, float)
    assert coerced_order.filled_qty == pytest.approx(3.5)


def test_poll_order_fill_status_warns_on_pending_new_timeout(caplog):
    from ai_trading.core.execution_flow import poll_order_fill_status

    class _Order:
        def __init__(self):
            self.status = "pending_new"
            self.qty = "4"
            self.filled_qty = "0"

    class _API:
        def __init__(self):
            self.calls = 0
            self.order = _Order()

        def get_order(self, _order_id):
            self.calls += 1
            return self.order

    ctx = types.SimpleNamespace(api=_API())

    with caplog.at_level("WARNING"):
        poll_order_fill_status(ctx, "oid-timeout", timeout=0.3)

    assert ctx.api.calls > 1, "Expected polling to retry until timeout"
    timeout_logs = [record for record in caplog.records if record.message == "ORDER_POLL_TIMEOUT"]
    assert timeout_logs, "Expected timeout warning to be logged"
    log_record = timeout_logs[0]
    assert log_record.status == "pending_new"
    assert log_record.order_id == "oid-timeout"
    assert log_record.timeout_s == 0.3

