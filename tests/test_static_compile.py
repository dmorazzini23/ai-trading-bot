import compileall
import pathlib


def test_package_compiles():
    """Statically compile the package to catch Syntax/Indentation errors early."""
    pkg = pathlib.Path(__file__).resolve().parents[1] / "ai_trading"
    assert compileall.compile_dir(str(pkg), quiet=1)
