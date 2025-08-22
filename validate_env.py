
from __future__ import annotations

import importlib.util
from pathlib import Path

# AI-AGENT-REF: Load env_validate without importing deprecated package
_MOD_PATH = Path(__file__).parent / "ai_trading" / "tools" / "env_validate.py"
_spec = importlib.util.spec_from_file_location("_env_validate", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader  # keep mypy happy
_spec.loader.exec_module(_mod)
_main = _mod.main

if __name__ == "__main__":
    raise SystemExit(_main())
