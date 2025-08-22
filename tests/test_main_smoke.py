import importlib

import pytest

try:
    main = importlib.import_module("run")
# noqa: BLE001 TODO: narrow exception
except Exception:  # pragma: no cover - optional entrypoint
    pytest.skip("run module not available", allow_module_level=True)


def test_main_smoke():
    assert main is not None
