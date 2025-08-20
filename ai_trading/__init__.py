
"""ai_trading public API.

Keep imports *lazy* to avoid optional deps at import-time (e.g., Alpaca).
"""

from importlib import import_module as _import_module

__all__ = [
    "config",
    "logging",
    "utils",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy export
    if name in {"config", "logging", "utils"}:
        return _import_module(f"ai_trading.{name}")
    raise AttributeError(name)
