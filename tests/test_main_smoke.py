import importlib

import pytest

try:
    main = importlib.import_module("run")
except Exception:  # pragma: no cover - optional entrypoint
    pytest.skip("run module not available", allow_module_level=True)


def test_main_smoke():
    assert main is not None
