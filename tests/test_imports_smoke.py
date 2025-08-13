"""Basic smoke test to ensure submodules import cleanly."""

import importlib
import pkgutil

# AI-AGENT-REF: smoke test for package imports


def _safe_import(name: str) -> None:
    """Import module unless it's a known heavy orchestrator."""
    skip_prefixes = {
        "ai_trading.core.bot_engine",  # heavy orchestrator
    }
    if any(name.startswith(p) for p in skip_prefixes):
        return
    importlib.import_module(name)


def test_submodules_import() -> None:
    pkg = importlib.import_module("ai_trading")
    for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        _safe_import(modinfo.name)

