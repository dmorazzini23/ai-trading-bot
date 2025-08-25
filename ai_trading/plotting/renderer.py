"""Basic plotting helpers."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

# AI-AGENT-REF: load optdeps without heavy package import
_spec = importlib.util.spec_from_file_location(
    "_optdeps", Path(__file__).resolve().parent.parent / "utils" / "optdeps.py"
)
_optdeps = importlib.util.module_from_spec(_spec)
sys.modules["_optdeps"] = _optdeps
assert _spec.loader is not None
_spec.loader.exec_module(_optdeps)
optional_import = _optdeps.optional_import
module_ok = _optdeps.module_ok
OptionalDependencyError = _optdeps.OptionalDependencyError

# AI-AGENT-REF: optional matplotlib import
plt = optional_import(
    "matplotlib.pyplot",
    purpose="plotting results",
    extra='pip install "ai-trading-bot[plot]"',
)


def render_equity_curve(series, *, title: str = "Equity") -> None:
    """Render a simple equity curve using matplotlib if available."""
    if not module_ok(plt):
        raise OptionalDependencyError(
            name="matplotlib",
            purpose="plotting",
            extra='pip install "ai-trading-bot[plot]"',
        )
    plt.figure()
    plt.plot(series)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Equity")
    plt.close()
