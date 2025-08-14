from __future__ import annotations
import os
# AI-AGENT-REF: legacy env var helper for tests


def require_env_vars(names):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")
    return True


# Legacy alias so tests can import _require_env_vars if they must
_require_env_vars = require_env_vars  # noqa: N816
