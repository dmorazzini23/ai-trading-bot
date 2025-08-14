"""
Compatibility facade for legacy imports:
    from features import build_features_pipeline
This forwards to the canonical implementation under ai_trading.features.*.

We intentionally probe a few common module layouts but do not import heavy
ML frameworks at import time. If nothing is found, we raise an informative error.
"""
from importlib import import_module
from typing import Any

__all__ = ["build_features_pipeline"]


def _resolve_attr(candidates: list[tuple[str, str]], *, attr_name: str) -> Any:
    for modpath, name in candidates:
        try:
            m = import_module(modpath)
            if hasattr(m, name):
                return getattr(m, name)
        except Exception:
            pass
    raise ImportError(
        f"Could not locate '{attr_name}' in any of the expected modules under ai_trading.features; "
        f"checked: {', '.join([f'{mp}.{attr_name}' for mp,_ in candidates])}"
    )


_build_candidates = [
    ("ai_trading.features", "build_features_pipeline"),
    ("ai_trading.features.pipeline", "build_features_pipeline"),
    ("ai_trading.features.build_features", "build_features_pipeline"),
    ("ai_trading.features.builder", "build_features_pipeline"),
    ("scripts.features", "build_features_pipeline"),
]

build_features_pipeline = _resolve_attr(_build_candidates, attr_name="build_features_pipeline")
