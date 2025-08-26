import runpy, sys


def test_module_imports_without_heavy_stacks(monkeypatch):
    heavy_roots = {"torch","gymnasium","pandas","pyarrow","sklearn","matplotlib"}
    before = set(sys.modules)
    # Running the module main should not pull heavy deps implicitly
    runpy.run_module("ai_trading", run_name="__main__")
    after = set(sys.modules)
    loaded = {m.split('.')[0] for m in (after - before)}
    assert heavy_roots.isdisjoint(loaded), f"Heavy modules imported at startup: {heavy_roots & loaded}"
