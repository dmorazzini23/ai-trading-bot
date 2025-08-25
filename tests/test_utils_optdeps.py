import pytest
import importlib.util
from pathlib import Path

# AI-AGENT-REF: load helper directly to avoid optional deps
spec = importlib.util.spec_from_file_location(
    "_optdeps",
    Path(__file__).resolve().parent.parent / "ai_trading" / "utils" / "optdeps.py",
)
_optdeps = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(_optdeps)

optional_import = _optdeps.optional_import
module_ok = _optdeps.module_ok


def test_optional_import_present_module():
    mod = optional_import("math")
    assert mod is not None
    sqrt = optional_import("math", attr="sqrt")
    assert callable(sqrt)


def test_optional_import_absent_module_returns_none():
    mod = optional_import("totally_nonexistent_pkg_xyz")
    assert mod is None


def test_optional_import_required_raises():
    with pytest.raises(ImportError):
        optional_import(
            "totally_nonexistent_pkg_xyz",
            required=True,
            install_hint="pip install totally-nonexistent",
        )


def test_module_ok_boolean():
    assert module_ok("math") is True
    assert module_ok("totally_nonexistent_pkg_xyz") is False
