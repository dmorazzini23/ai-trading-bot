import runpy
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
