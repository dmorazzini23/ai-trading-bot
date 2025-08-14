from __future__ import annotations
import os
from typing import Optional

# AI-AGENT-REF: facade for environment validation

def debug_environment() -> dict:
    """Return a tiny dump used by tests without side effects."""
    return {"pythonpath": os.environ.get("PYTHONPATH", ""), "env": dict(os.environ)}


def validate_specific_env_var(name: str, required: bool = False) -> Optional[str]:
    val = os.environ.get(name)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val
