import importlib.util
from pathlib import Path
import pytest
import sys

# AI-AGENT-REF: load helper directly to avoid heavy optional deps
spec = importlib.util.spec_from_file_location(
    "_optdeps",
    Path(__file__).resolve().parent.parent / "ai_trading" / "utils" / "optdeps.py",
)
_optdeps = importlib.util.module_from_spec(spec)
sys.modules["_optdeps"] = _optdeps
assert spec.loader is not None
spec.loader.exec_module(_optdeps)

optional_import = _optdeps.optional_import
module_ok = _optdeps.module_ok
OptionalDependencyError = _optdeps.OptionalDependencyError


def test_optional_import_present_module():
    mod = optional_import("math")
    assert mod is not None
    sqrt = optional_import("math", attr="sqrt")
    assert callable(sqrt)


def test_optional_import_absent_module_returns_none():
    mod = optional_import("totally_nonexistent_pkg_xyz")
    assert mod is None


def test_required_raises_clear_message(monkeypatch):
    # AI-AGENT-REF: ensure extras hint surfaces
    monkeypatch.setitem(sys.modules, "totally_missing_pkg", None)
    with pytest.raises(OptionalDependencyError) as ei:
        optional_import(
            "totally_missing_pkg",
            required=True,
            purpose="demo",
            extra="demo",
        )
    msg = str(ei.value)
    assert "Missing optional dependency 'totally_missing_pkg'" in msg
    assert "for demo" in msg
    assert 'Install with: pip install "ai-trading-bot[demo]"' in msg


def test_auto_derives_extra(monkeypatch):
    # AI-AGENT-REF: derive extras without explicit hint
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(OptionalDependencyError) as ei:
        optional_import("pandas", required=True)
    msg = str(ei.value)
    assert 'pip install "ai-trading-bot[pandas]"' in msg


def test_render_equity_curve_handles_missing_matplotlib(monkeypatch):
    sys.modules.pop("ai_trading.plotting.renderer", None)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "matplotlib.pyplot":
            raise ImportError("simulated missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from ai_trading.plotting import renderer

    with pytest.raises(renderer.OptionalDependencyError):
        renderer.render_equity_curve([1, 2, 3])


def test_module_ok_boolean():
    assert module_ok("math") is True
    assert module_ok(None) is False


@pytest.mark.parametrize(
    "pkg,expected_extra",
    [("pandas", "pandas"), ("matplotlib", "plot"), ("ta", "ta")],
)
def test_optional_import_extras_hint(monkeypatch, pkg, expected_extra):
    """Ensure OptionalDependencyError carries the proper extras hint."""
    def fake_import(name, *a, **k):
        if name == pkg:
            raise ImportError("simulated missing")
        return __import__(name, *a, **k)

    import builtins

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(OptionalDependencyError) as ei:
        optional_import(pkg, required=True)
    msg = str(ei.value)
    assert f'pip install "ai-trading-bot[{expected_extra}]"' in msg
