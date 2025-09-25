import sys
import types
from types import SimpleNamespace

import ai_trading.logging.emit_once as emit_once_module


class _NumPyStub(types.ModuleType):
    ndarray = object
    float64 = float
    int64 = int
    nan = float("nan")
    NaN = float("nan")

    class _RandomStub:
        def seed(self, *args, **kwargs):  # noqa: D401 - compatibility shim
            return None

    random = _RandomStub()

    def array(self, *args, **kwargs):  # noqa: D401 - compatibility shim
        return args[0] if args else None

    def zeros(self, *args, **kwargs):  # noqa: D401 - compatibility shim
        return [0] * (args[0] if args else 0)

    def ones(self, *args, **kwargs):  # noqa: D401 - compatibility shim
        return [1] * (args[0] if args else 0)

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


if "numpy" not in sys.modules:
    sys.modules["numpy"] = _NumPyStub("numpy")

if "portalocker" not in sys.modules:
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1

    def _noop(*args, **kwargs):  # noqa: D401 - compatibility shim
        return None

    portalocker_stub.lock = _noop
    portalocker_stub.unlock = _noop
    sys.modules["portalocker"] = portalocker_stub

if "bs4" not in sys.modules:
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:
        def __init__(self, *args, **kwargs):  # noqa: D401 - compatibility shim
            self.args = args
            self.kwargs = kwargs

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

from ai_trading.core import bot_engine


def _reset_emit_once(monkeypatch):
    monkeypatch.setattr(emit_once_module, "_emitted", {}, raising=False)


def test_liquidity_factor_handles_missing_data_client(monkeypatch):
    _reset_emit_once(monkeypatch)
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    stub_settings = SimpleNamespace(default_liquidity_factor=0.75)
    monkeypatch.setattr(bot_engine, "get_settings", lambda: stub_settings)
    ctx = SimpleNamespace(data_client=None, volume_threshold=100_000, last_bar_by_symbol={})

    factor = bot_engine.liquidity_factor(ctx, "FAKE")

    assert abs(factor - 0.75) < 1e-6
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)


def test_liquidity_factor_falls_back_to_last_bar(monkeypatch):
    _reset_emit_once(monkeypatch)
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    stub_settings = SimpleNamespace(default_liquidity_factor=0.9)
    monkeypatch.setattr(bot_engine, "get_settings", lambda: stub_settings)

    class FailingClient:
        def get_stock_latest_quote(self, req):  # noqa: D401 - simple stub
            raise bot_engine.APIError("quote unavailable")

    last_bar = SimpleNamespace(high=102, low=98, close=100, volume=2_000_000)
    ctx = SimpleNamespace(
        data_client=FailingClient(),
        volume_threshold=100_000,
        last_bar_by_symbol={"FAKE": last_bar},
    )

    factor = bot_engine.liquidity_factor(ctx, "FAKE")

    assert 0.1 <= factor <= 0.9
    assert factor < 0.9  # Uses last-bar heuristics rather than default directly
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
