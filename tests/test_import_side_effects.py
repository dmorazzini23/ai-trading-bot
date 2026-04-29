import runpy
import subprocess
import sys
import warnings


def test_module_imports_without_heavy_stacks(monkeypatch):
    heavy_roots = {"torch", "gymnasium", "pandas", "pyarrow", "sklearn", "matplotlib"}
    before = set(sys.modules)
    # Running the module main should not pull heavy deps implicitly
    monkeypatch.setattr(sys, "argv", ["ai_trading"])
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    runpy.run_module("ai_trading", run_name="__main__")
    after = set(sys.modules)
    loaded = {m.split('.')[0] for m in (after - before)}
    assert heavy_roots.isdisjoint(loaded), f"Heavy modules imported at startup: {heavy_roots & loaded}"


def test_run_module_emits_no_runtime_warning(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ai_trading"])
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        runpy.run_module("ai_trading", run_name="__main__")
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings, f"RuntimeWarning emitted: {runtime_warnings}"


def test_bot_engine_import_does_not_create_trade_or_slippage_paths(tmp_path):
    code = """
import os
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
os.chdir(base)
os.environ["PYTEST_RUNNING"] = "1"
os.environ["TRADE_LOG_PATH"] = str(base / "configured" / "trades.jsonl")
os.environ["SLIPPAGE_LOG_PATH"] = str(base / "configured" / "slippage.csv")

import ai_trading.core.bot_engine  # noqa: F401

assert not (base / "logs").exists()
assert not (base / "configured").exists()
"""

    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
