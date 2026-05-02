from __future__ import annotations

import importlib.util
from pathlib import Path


def test_smoke_backtest_uses_repo_root_import_path() -> None:
    script_path = Path("scripts/smoke_backtest.py").resolve()
    spec = importlib.util.spec_from_file_location("smoke_backtest_under_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.run_backtest_smoke_test() is True
