import importlib.machinery
import logging
import pathlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

if "dotenv" not in sys.modules:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    stub_origin = repo_root / ".venv" / "lib" / "python3.11" / "site-packages" / "dotenv" / "__init__.py"
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.__spec__ = importlib.machinery.ModuleSpec("dotenv", loader=None, origin=str(stub_origin))
    dotenv_stub.load_dotenv = lambda *a, **k: None
    dotenv_stub.dotenv_values = lambda *a, **k: {}
    sys.modules["dotenv"] = dotenv_stub

for missing in ("numpy", "pandas", "sklearn", "tenacity", "portalocker", "bs4"):
    if missing not in sys.modules:
        module = types.ModuleType(missing)
        if missing == "numpy":
            module.nan = float("nan")
            module.NaN = module.nan
            module.array = lambda *a, **k: list(a)
            module.random = types.SimpleNamespace(seed=lambda *_a, **_k: None)
        if missing == "tenacity":
            def _retry(*_a, **_k):  # pragma: no cover - defensive stub
                def decorator(func):
                    return func

                return decorator

            module.retry = _retry
            module.RetryError = RuntimeError
            module.stop_after_attempt = lambda *_a, **_k: None
            module.wait_fixed = lambda *_a, **_k: None
            module.wait_random_exponential = lambda *_a, **_k: None
            module.wait_exponential = lambda *_a, **_k: None
            module.retry_if_exception_type = lambda *_a, **_k: True
        if missing == "portalocker":
            class _Lock:
                def __init__(self, *_a, **_k):
                    pass

                def acquire(self, *_a, **_k):
                    return True

                def release(self):
                    return True

                def __enter__(self):
                    return self

                def __exit__(self, *_exc):
                    return False

            module.Lock = _Lock
        if missing == "bs4":
            class _Soup:
                def __init__(self, *_a, **_k):
                    self.text = ""

            module.BeautifulSoup = _Soup
        sys.modules[missing] = module

from ai_trading.core.bot_engine import _check_runtime_stops
from ai_trading.execution.production_engine import ProductionExecutionCoordinator


def test_runtime_stop_checks_use_production_coordinator(caplog):
    coordinator = ProductionExecutionCoordinator(account_equity=100000)
    coordinator.current_positions["AAPL"] = {"quantity": 10, "avg_price": 150.0}
    runtime = SimpleNamespace(exec_engine=coordinator)

    with patch.object(coordinator, "check_stops", wraps=coordinator.check_stops) as spy:
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _check_runtime_stops(runtime)

    assert spy.called, "ProductionExecutionCoordinator.check_stops was not invoked"
    assert not caplog.records, "check_stops emitted warnings when invoked via bot engine"
